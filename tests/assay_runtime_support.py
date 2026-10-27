"""Synthetic pixels/models only; real cp_measure and ALIBY adapters are exercised."""

import hashlib
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from aliby.assay import (
    AssayRequest,
    ChannelEvidence,
    ChannelInput,
    MeasurementRequest,
    ModelCapability,
    PlaneSample,
    SegmentationRequest,
    decide_assay,
)
from aliby.assay_runtime import bind_assay_pipeline
from aliby.pipe_builder import build_pipeline_steps


def fixture(
    tmp_path, monkeypatch, *, kind="cellpose", ntps=2, dtype=np.uint16, save=False
):
    pixels = np.zeros((ntps, 2, 3, 5, 9, 9), dtype=dtype)
    for tp in range(ntps):
        pixels[tp, :, 0] = np.arange(1, 6)[None, :, None, None] + tp
        pixels[tp, :, 1, 0] = 10 + tp
        pixels[tp, :, 2, 0] = 100 + tp
    log = {"constructed": [], "loaded_weights": [], "images": [], "tilers": []}
    weights = tmp_path / "synthetic-model.bin"
    weights.write_bytes(b"test stub, not real neural weights")
    identity = "sha256:" + hashlib.sha256(weights.read_bytes()).hexdigest()
    channels = tuple(
        ChannelEvidence(
            index=c,
            name=name,
            modality=modality,
            stored_z=5,
            acquired_z=tuple(range(5)) if c == 0 else (0,),
            metadata_evidence=("fixture:acquisition",),
            samples=(
                PlaneSample(
                    0,
                    ("nonzero",) * 5 if c == 0 else ("nonzero",) + ("zero",) * 4,
                    f"fixture:channel-{c}",
                ),
            ),
        )
        for c, name, modality in (
            (0, "mCherry", "fluorescence"),
            (1, "GFP", "fluorescence"),
            (2, "Brightfield", "brightfield"),
        )
    )
    request = AssayRequest(
        "fixture-site",
        "fixture:raw",
        ntps,
        channels,
        (SegmentationRequest("cell", (ChannelInput(2, (0,)),)),),
        (
            MeasurementRequest(
                "red", "cell", (ChannelInput(0, tuple(range(5)), "max"),)
            ),
            MeasurementRequest("green", "cell", (ChannelInput(1, (0,)),)),
        ),
        assay_type="synthetic_yeast",
    )
    model = ModelCapability(
        "synthetic-bf",
        identity,
        ("brightfield",),
        1,
        masks="overlapping" if kind == "nahual_baby" else "labels",
        stateful=kind == "nahual_baby",
        assays=("synthetic_yeast",),
        targets=("cell",),
    )
    decision = decide_assay(request, (model,))
    p = build_pipeline_steps(
        channels_to_segment={"cell": 2},
        channels_to_extract=[0, 1],
        features_to_extract=("intensity",),
    )
    p["ntps"] = ntps
    if not save:
        p["save"] = []
    p["steps"]["tile"]["image_kwargs"] = {"source": "fixture-only"}
    kw = {"kind": kind}
    if kind == "nahual_baby":
        kw.update(address="http://127.0.0.1:1", modelset=model.model_id)
    else:
        kw["setup_params"] = {"pretrained_model": str(weights), "gpu": False}
    p["steps"]["segment_cell"]["segmenter_kwargs"] = kw

    def masks(overlap=False):
        result = []
        for tile in range(2):
            m = np.zeros((9, 9), dtype=np.uint16)
            m[1 + tile : 3 + tile, 1 + tile : 3 + tile] = 17 + tile if overlap else 1
            result.append(m[None] if overlap else m)
        return result

    class FakeModel:
        def __init__(self, **kwargs):
            log["constructed"].append(kwargs)
            log["loaded_weights"].append(Path(kwargs["pretrained_model"]).read_bytes())

        def eval(self, images, **kwargs):
            assert kwargs["channel_axis"] == -1 and kwargs["do_3D"] is False
            assert len(images) == 2
            assert all(image.shape == (9, 9, 1) for image in images)
            log["images"].append([im.copy() for im in images])
            return masks(), None, None

    monkeypatch.setitem(
        sys.modules, "cellpose.models", SimpleNamespace(CellposeModel=FakeModel)
    )

    def load_model(address, modelset):
        log["constructed"].append((address, modelset))
        return "fixture-session"

    def process_data(data, **kwargs):
        assert kwargs["channel_to_segment"] == 0
        assert data.shape == (2, 1, 1, 9, 9)
        assert data.dtype == np.uint16
        log["images"].append(data.copy())
        return {
            "masks": masks(True),
            "metadata": [
                {"cell_label": [17 + tile], "mother_assign": []} for tile in range(2)
            ],
        }

    # Explicit fake guarded-client contract; no sockets or HTTP calls occur.
    stub = SimpleNamespace(
        load_model=load_model,
        process_data=process_data,
        _UNSAFE_SESSIONS=set(),
        _SESSION_LOCK=object(),
        _checked_get=lambda *a: None,
        HTTP_TIMEOUT=(1, 1),
        POLL_TIMEOUT=1,
    )
    import nahual.client

    monkeypatch.setitem(sys.modules, "nahual.client.baby", stub)
    monkeypatch.setattr(nahual.client, "baby", stub, raising=False)

    class Tile:
        def __init__(self):
            self.tp = -1

        def run_tp(self, tp):
            assert tp == self.tp + 1
            self.tp = tp
            return {"pixels": self.get_fczyx(tp)}

        def get_fczyx(self, tp):
            assert tp == self.tp
            return pixels[tp].copy()

    import aliby.pipe as pipe
    import aliby.pipe_baby as pipe_baby

    old_init = pipe.init_step
    old_baby = pipe_baby.init_step

    def initialize(name, params, others=None):
        if name == "tile":
            tiler = Tile()
            log["tilers"].append(tiler)
            return tiler
        return old_init(name, params, others)

    def initialize_baby(name, params, others=None):
        if name == "tile":
            return initialize(name, params, others)
        return old_baby(name, params, others)

    monkeypatch.setattr(pipe, "init_step", initialize)
    # pipe's public APIs are partials bound to the original initializer; the
    # release test patches those partials explicitly rather than starting IO.
    from functools import partial
    from aliby.pipe_core import _run_pipeline_and_post_async_impl

    monkeypatch.setattr(
        pipe,
        "run_pipeline_and_post_async",
        partial(_run_pipeline_and_post_async_impl, init_step_fn=initialize),
    )
    monkeypatch.setattr(pipe_baby, "init_step", initialize_baby)
    bound = bind_assay_pipeline(
        p, decision, model_identities={model.model_id: identity}
    )
    return SimpleNamespace(
        pipeline=p,
        bound=bound,
        decision=decision,
        model=model,
        log=log,
        pixels=pixels,
        init=initialize_baby if kind == "nahual_baby" else initialize,
        stub=stub,
        weights=weights,
    )

import asyncio
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from aliby.assay import ChannelInput, MeasurementRequest, decide_assay
from aliby.assay_runtime import bind_assay_pipeline, validate_assay_binding
from aliby.executor import PipelineStepError
from aliby.pipe_core import get_profiles_from_state, run_pipeline_return_state
from assay_runtime_support import fixture


@pytest.mark.parametrize("kind", ["cellpose", "nahual_baby"])
@pytest.mark.parametrize("backend", ["sequential", "concurrent", "async"])
def test_real_measurements_use_independent_channel_z_inputs(
    tmp_path, monkeypatch, kind, backend
):
    f = fixture(tmp_path, monkeypatch, kind=kind)
    original = deepcopy(f.bound)
    pixels = f.pixels.copy()
    state = run_pipeline_return_state(
        f.bound, None, f.init, backend=backend, max_workers=3
    )
    table = get_profiles_from_state(state, f.bound)
    assert len(table) == 4
    red = next(
        n
        for n in table.column_names
        if n.startswith("c0[z=0,1,2,3,4;max]") and "MeanIntensity" in n
    )
    green = next(
        n
        for n in table.column_names
        if n.startswith("c1[z=0;stack]") and "MeanIntensity" in n
    )
    assert table[red].to_pylist() == [5.0, 5.0, 6.0, 6.0]
    assert table[green].to_pylist() == [10.0, 10.0, 11.0, 11.0]
    assert len(f.log["constructed"]) == 1
    assert len(f.log["images"]) == 2
    if kind == "nahual_baby":
        assert set(table["metadata_label"].to_pylist()) == {17, 18}
        assert all(im.shape == (2, 1, 1, 9, 9) for im in f.log["images"])
    else:
        assert state["data"]["segment_cell"][0][0].shape == (9, 9)
        assert not np.array_equal(*state["data"]["segment_cell"][0])
    assert f.bound == original
    validate_assay_binding(f.bound)
    np.testing.assert_equal(f.pixels, pixels)


def test_negative_fluorescence_is_not_replaced_by_padded_zero_max(
    tmp_path, monkeypatch
):
    f = fixture(tmp_path, monkeypatch, ntps=1, dtype=np.float64)
    f.pixels[:, :, 1, 0] = -7
    state = run_pipeline_return_state(f.bound, None, f.init, backend="async")
    table = get_profiles_from_state(state, f.bound)
    green = next(
        n for n in table.column_names if n.startswith("c1[") and "MeanIntensity" in n
    )
    assert table[green].to_pylist() == [-7.0, -7.0]


def test_bound_cellpose_preserves_tiles_and_calls_the_declared_weights(
    tmp_path, monkeypatch
):
    f = fixture(tmp_path, monkeypatch, ntps=1)
    state = run_pipeline_return_state(f.bound, None, f.init, backend="async")
    masks = state["data"]["segment_cell"][0]
    assert len(masks) == 2 and not np.array_equal(*masks)
    assert f.log["constructed"][0]["pretrained_model"].startswith("/proc/self/fd/")
    assert f.log["constructed"][0]["gpu"] is False
    assert f.log["loaded_weights"] == [f.weights.read_bytes()]
    assert np.all(f.log["images"][0][0] == 100)


@pytest.mark.parametrize(
    "change", ["weights", "missing_weights", "binding", "unknown_option"]
)
def test_unsafe_or_changed_configuration_never_calls_model_eval(
    tmp_path, monkeypatch, change
):
    f = fixture(tmp_path, monkeypatch)
    if change == "weights":
        f.weights.write_bytes(b"changed weights")
    elif change == "missing_weights":
        f.weights.unlink()
    elif change == "binding":
        f.bound["steps"]["segment_cell"]["channel_to_segment"] = 2
    else:
        p = deepcopy(f.pipeline)
        p["steps"]["segment_cell"]["segmenter_kwargs"]["ignored_option"] = 1
        with pytest.raises(ValueError, match="Unsupported"):
            bind_assay_pipeline(
                p, f.decision, model_identities={f.model.model_id: f.model.identity}
            )
        return
    with pytest.raises((ValueError, PipelineStepError)):
        run_pipeline_return_state(f.bound, None, f.init, backend="async")
    assert not f.log["constructed"] and not f.log["images"]


@pytest.mark.parametrize("problem", ["zero", "nonfinite", "float_baby"])
def test_actual_pixels_are_checked_before_model_submission(
    tmp_path, monkeypatch, problem
):
    kind = "nahual_baby" if problem == "float_baby" else "cellpose"
    f = fixture(tmp_path, monkeypatch, kind=kind, dtype=np.float64)
    if problem == "zero":
        f.pixels[:, :, 2, 0] = 0
    elif problem == "nonfinite":
        f.pixels[0, 0, 2, 0, 0, 0] = np.nan
    with pytest.raises(PipelineStepError):
        run_pipeline_return_state(f.bound, None, f.init, backend="async")
    assert not f.log["images"]


@pytest.mark.parametrize("bound", [True, False])
def test_unguarded_baby_client_is_rejected_before_loading_session(
    tmp_path, monkeypatch, bound
):
    f = fixture(tmp_path, monkeypatch, kind="nahual_baby")
    del f.stub._UNSAFE_SESSIONS
    with pytest.raises(RuntimeError, match="guarded"):
        run_pipeline_return_state(
            f.bound if bound else f.pipeline, None, f.init, backend="async"
        )
    assert not f.log["constructed"] and not f.log["tilers"]


def test_baby_label_mismatch_aborts_without_a_second_frame(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch, kind="nahual_baby")
    original = f.stub.process_data

    def bad(*a, **kw):
        result = original(*a, **kw)
        result["metadata"][0]["cell_label"] = [999]
        return result

    f.stub.process_data = bad
    with pytest.raises(PipelineStepError, match="tracking metadata"):
        run_pipeline_return_state(f.bound, None, f.init, backend="async")
    assert len(f.log["images"]) == 1


def test_binder_rejects_wrong_identity_unsupported_feature_and_ambiguous_measurement(
    tmp_path, monkeypatch
):
    f = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="identity"):
        bind_assay_pipeline(f.pipeline, f.decision, model_identities={})
    for measurements, message in [
        (
            (replace(f.decision.request.measurements[0], features=("nonexistent",)),),
            "Unsupported extractor",
        ),
        (
            (
                f.decision.request.measurements[0],
                MeasurementRequest("red-other-z", "cell", (ChannelInput(0, (0,)),)),
            ),
            "One selection",
        ),
    ]:
        decision = decide_assay(
            replace(f.decision.request, measurements=measurements), (f.model,)
        )
        with pytest.raises(ValueError, match=message):
            bind_assay_pipeline(
                f.pipeline,
                decision,
                model_identities={f.model.model_id: f.model.identity},
            )
    assert not f.log["constructed"]


def test_native_baby_api_writes_matching_tracking_and_profiles(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch, kind="nahual_baby")
    from aliby.pipe_baby import run_pipeline_and_post_async
    import pyarrow.parquet as pq

    profiles, _ = asyncio.run(
        run_pipeline_and_post_async(f.bound, "fixture", tmp_path / "run", max_workers=3)
    )
    tracking = pq.read_table(tmp_path / "run/tracking/fixture_segment_cell.parquet")
    assert len(profiles) == len(tracking) == 4
    assert (
        set(profiles["metadata_label"].to_pylist())
        == set(tracking["cell_label"].to_pylist())
        == {17, 18}
    )


def test_multichannel_cell_painting_binding_and_different_object_features(
    tmp_path, monkeypatch
):
    from aliby.assay import ModelCapability, SegmentationRequest
    from aliby.pipe_builder import build_pipeline_steps
    from types import SimpleNamespace
    import sys

    f = fixture(tmp_path, monkeypatch)
    seen = []

    class Model:
        def __init__(self, **kwargs):
            pass

        def eval(self, images, **kwargs):
            depth = images[0].shape[-1]
            seen.append(depth)
            assert depth in (1, 2)
            for image in images:
                if depth == 2:
                    assert np.all(image[..., 1] - image[..., 0] == 5)
            return [np.ones((9, 9), dtype=np.uint16) for _ in images], None, None

    monkeypatch.setitem(
        sys.modules, "cellpose.models", SimpleNamespace(CellposeModel=Model)
    )
    common = dict(
        identity=f.model.identity,
        z_planes=1,
        assays=("cell_painting",),
        reductions=("stack", "max"),
    )
    models = (
        ModelCapability(
            "nuclei", modalities=("fluorescence",), targets=("nucleus",), **common
        ),
        ModelCapability(
            "cells",
            modalities=("fluorescence", "fluorescence"),
            targets=("cell",),
            **common,
        ),
    )
    red, green = ChannelInput(0, tuple(range(5)), "max"), ChannelInput(1, (0,))
    request = replace(
        f.decision.request,
        assay_type="cell_painting",
        segmentations=(
            SegmentationRequest("nuclei", (green,), target="nucleus"),
            SegmentationRequest("cell", (red, green)),
        ),
        measurements=(
            MeasurementRequest("nuclear-green", "nuclei", (green,)),
            MeasurementRequest("cell-red", "cell", (red,)),
        ),
    )
    decision = decide_assay(request, models)
    p = build_pipeline_steps(channels_to_segment={"nuclei": 1, "cell": 0})
    p.update(ntps=2, save=[])
    p["steps"]["tile"]["image_kwargs"] = {"source": "fixture-only"}
    for name in ("segment_nuclei", "segment_cell"):
        p["steps"][name]["segmenter_kwargs"] = {
            "kind": "cellpose",
            "setup_params": {"pretrained_model": str(f.weights), "gpu": False},
        }
    bound = bind_assay_pipeline(
        p, decision, model_identities={m.model_id: m.identity for m in models}
    )
    state = run_pipeline_return_state(
        bound, None, f.init, backend="async", max_workers=4
    )
    table = get_profiles_from_state(state, bound)
    assert len(table) == 8 and sorted(seen) == [1, 1, 2, 2]
    assert set(table["metadata_object"].to_pylist()) == {"nuclei", "cell"}
    red_column = next(
        n for n in table.column_names if n.startswith("c0[") and "MeanIntensity" in n
    )
    assert table[red_column].null_count == 4


def test_saved_baby_artifacts_are_equivalent_across_backends(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch, kind="nahual_baby", save=True)
    outputs = []
    for backend in ("sequential", "async"):
        path = tmp_path / backend
        state = run_pipeline_return_state(f.bound, path, f.init, backend=backend)
        outputs.append(get_profiles_from_state(state, f.bound))
    assert outputs[0].equals(outputs[1])
    for path in (tmp_path / "sequential").rglob("*"):
        if path.is_file():
            assert (
                path.read_bytes()
                == (
                    tmp_path / "async" / path.relative_to(tmp_path / "sequential")
                ).read_bytes()
            )


def test_colocalization_binding_uses_real_two_channel_extractor(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    # Spatial variation avoids a zero-variance synthetic correlation.
    pattern = np.arange(81, dtype=np.uint16).reshape(9, 9)
    f.pixels[:, :, 0] += pattern
    f.pixels[:, :, 1, 0] += pattern * 2
    pair = MeasurementRequest(
        "pair",
        "cell",
        (ChannelInput(0, tuple(range(5)), "max"), ChannelInput(1, (0,))),
        ("pearson",),
    )
    decision = decide_assay(
        replace(
            f.decision.request, measurements=(*f.decision.request.measurements, pair)
        ),
        (f.model,),
    )
    bound = bind_assay_pipeline(
        f.pipeline, decision, model_identities={f.model.model_id: f.model.identity}
    )
    state = run_pipeline_return_state(bound, None, f.init, backend="async")
    profiles = get_profiles_from_state(state, bound)
    assert len(profiles) == 4
    assert any("|c1[" in name and "pearson" in name for name in profiles.column_names)

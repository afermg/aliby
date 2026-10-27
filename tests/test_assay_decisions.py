from dataclasses import FrozenInstanceError, replace
import json

import numpy as np
import pytest

from aliby.assay import (
    AssayRequest,
    ChannelEvidence,
    ChannelInput,
    MeasurementRequest,
    ModelCapability,
    PlaneSample,
    SegmentationRequest,
    decide_assay,
    select_inputs,
)


def channel(index, name, modality, states, acquired=None, **kwargs):
    return ChannelEvidence(
        index,
        name,
        modality,
        len(states),
        (PlaneSample(0, tuple(states), f"raw-plane-digest:{index}"),),
        acquired_z=acquired,
        metadata_evidence=("sha256:exact-site-acquisition",)
        if acquired is not None
        else (),
        **kwargs,
    )


def yeast_request():
    return AssayRequest(
        "14189201:phluorin_dot6_001",
        "md5:raw-archive",
        12,
        (
            channel(0, "mCherry", "fluorescence", ["nonzero"] * 5, tuple(range(5))),
            channel(1, "GFP", "fluorescence", ["nonzero"] + ["zero"] * 4, (0,)),
            channel(2, "Brightfield", "brightfield", ["nonzero"] + ["zero"] * 4, (0,)),
        ),
        (SegmentationRequest("cell", (ChannelInput(2, (0,)),)),),
        (
            MeasurementRequest(
                "mCherry", "cell", (ChannelInput(0, tuple(range(5)), "max"),)
            ),
            MeasurementRequest("GFP", "cell", (ChannelInput(1, (0,)),)),
        ),
        assay_type="yeast_microfluidics",
    )


def baby(z=1, **kwargs):
    return ModelCapability(
        f"baby-{z}z",
        f"sha256:model-{z}",
        ("brightfield",),
        z,
        "overlapping",
        True,
        assays=("yeast_microfluidics",),
        targets=("cell",),
        **kwargs,
    )


def codes(decision, severity="review"):
    return {i.code for i in decision.issues if i.severity == severity}


def test_channel_specific_single_plane_bf_keeps_fluorescence_stack():
    request = yeast_request()
    decision = decide_assay(request, (baby(5), baby(1)))
    assert decision.require_ready() is decision
    assert decision.selected_models == (("cell", "baby-1z"),)
    assert decision.request.measurements[0].inputs[0].z == (0, 1, 2, 3, 4)
    assert decision.request.channels[2].stored_z == 5
    assert "sampled_compatibility_only" in codes(decision, "warning")


@pytest.mark.parametrize("nz", [1, 3, 5])
def test_no_acquisition_sidecar_is_not_a_blocker(nz):
    request = AssayRequest(
        "new-site",
        "raw-digest",
        1,
        (channel(0, "configured BF", "brightfield", ["nonzero"] * nz),),
        (SegmentationRequest("cell", (ChannelInput(0, tuple(range(nz))),)),),
        assay_type="yeast_microfluidics",
    )
    decision = decide_assay(request, (baby(nz),))
    assert decision.ready
    assert "optional_metadata_pixel_fallback" in codes(decision, "warning")


@pytest.mark.parametrize(
    "reason",
    [
        "missing sidecar",
        "unsupported format",
        "unreadable sidecar",
        "no exact-site match",
    ],
)
def test_unusable_optional_metadata_is_a_warning(reason):
    request = yeast_request()
    bf = replace(
        request.channels[2],
        acquired_z=None,
        metadata_evidence=(),
        metadata_warnings=(reason,),
    )
    decision = decide_assay(
        replace(request, channels=(*request.channels[:2], bf)), (baby(),)
    )
    assert decision.ready
    assert reason in [i.detail for i in decision.issues]


def test_explicit_incompatible_model_never_falls_back_to_available_one_plane_model():
    request = yeast_request()
    request = replace(
        request, segmentations=(replace(request.segmentations[0], model_id="baby-5z"),)
    )
    decision = decide_assay(request, (baby(1), baby(5)))
    assert not decision.ready
    assert decision.selected_models == ()
    assert codes(decision) == {"model_input_incompatible"}


@pytest.mark.parametrize(
    "model_id, expected",
    [(None, "model_choice_ambiguous"), ("absent", "requested_model_unavailable")],
)
def test_ambiguous_or_unavailable_models_require_review(model_id, expected):
    request = yeast_request()
    request = replace(
        request, segmentations=(replace(request.segmentations[0], model_id=model_id),)
    )
    decision = decide_assay(request, (baby(), replace(baby(), model_id="other-camera")))
    assert not decision.ready
    assert expected in codes(decision)
    assert not decision.selected_models


def test_model_name_does_not_determine_capability_or_calibration():
    model = replace(baby(), model_id="name-says-5z-but-declaration-is-1z")
    assert decide_assay(yeast_request(), (model,)).ready
    assert not decide_assay(yeast_request(), (baby(5),)).ready


def test_metadata_conflicts_are_retained_not_overridden_by_visible_pixels():
    request = yeast_request()
    bf = replace(
        request.channels[2],
        metadata_conflicts=("BF index conflicts with exact-site acquisition",),
    )
    decision = decide_assay(
        replace(request, channels=(*request.channels[:2], bf)), (baby(),)
    )
    assert "acquisition_metadata_conflict" in codes(decision)
    assert decision.request.channels[2].metadata_conflicts == bf.metadata_conflicts


@pytest.mark.parametrize("acquired", [(), (1,)])
def test_not_acquired_plane_is_not_the_same_as_missing_metadata(acquired):
    request = yeast_request()
    bf = replace(request.channels[2], acquired_z=acquired)
    decision = decide_assay(
        replace(request, channels=(*request.channels[:2], bf)), (baby(),)
    )
    assert "selected_plane_not_acquired" in codes(decision)


@pytest.mark.parametrize("state", ["missing", "unreadable", "unknown"])
@pytest.mark.parametrize("index", [0, 2])
def test_unobservable_selected_segmentation_or_measurement_plane_requires_review(
    state, index
):
    request = yeast_request()
    channels = list(request.channels)
    original = channels[index]
    states = list(original.samples[0].states)
    states[0] = state
    channels[index] = replace(
        original, samples=(PlaneSample(0, tuple(states), "missing-evidence"),)
    )
    decision = decide_assay(replace(request, channels=channels), (baby(),))
    assert "selected_plane_unobservable" in codes(decision)
    with pytest.raises(ValueError, match="not ready"):
        decision.require_ready()


@pytest.mark.parametrize("index, ready", [(0, True), (2, False)])
def test_stored_zero_fluorescence_is_not_missing_or_no_cells(index, ready):
    request = yeast_request()
    channels = list(request.channels)
    channels[index] = replace(
        channels[index], samples=(PlaneSample(0, ("zero",) * 5, "zero-pixels"),)
    )
    decision = decide_assay(replace(request, channels=channels), (baby(),))
    assert decision.ready is ready
    assert ("zero_segmentation_plane" in codes(decision)) is (index == 2)
    assert (
        "zero_measurement_not_biological_absence" in codes(decision, "warning")
    ) is (index == 0)


def test_no_samples_and_changed_late_availability_require_review():
    request = yeast_request()
    bf = request.channels[2]
    for samples, expected in [
        ((), "raw_evidence_missing"),
        (
            bf.samples + (PlaneSample(11, ("missing",) + ("zero",) * 4, "late-raw"),),
            "selected_plane_unobservable",
        ),
    ]:
        decision = decide_assay(
            replace(
                request, channels=(*request.channels[:2], replace(bf, samples=samples))
            ),
            (baby(),),
        )
        assert expected in codes(decision)


def test_unused_zero_padding_is_not_submitted_and_missing_unused_slots_are_retained():
    request = yeast_request()
    bf = replace(
        request.channels[2],
        samples=(PlaneSample(0, ("nonzero",) + ("missing",) * 4, "sparse-archive"),),
    )
    decision = decide_assay(
        replace(request, channels=(*request.channels[:2], bf)), (baby(),)
    )
    assert (
        decision.ready
    )  # Selected Z0 only; not a declaration of full raw completeness.
    assert "missing" in json.dumps(decision.to_dict())


def test_unselected_nonzero_planes_cannot_hide_acquisition_conflicts():
    request = yeast_request()
    bf = replace(
        request.channels[2],
        samples=(PlaneSample(0, ("nonzero",) * 5, "conflicting-pixels"),),
    )
    request = replace(request, channels=(*request.channels[:2], bf))
    decision = decide_assay(request, (baby(),))
    assert "acquisition_pixel_conflict" in codes(decision)
    assert not decision.ready


def test_all_stored_planes_is_not_a_valid_default_for_single_plane_bf():
    request = yeast_request()
    request = replace(
        request,
        segmentations=(
            SegmentationRequest("cell", (ChannelInput(2, (0, 1, 2, 3, 4)),)),
        ),
    )
    decision = decide_assay(request, (baby(5),))
    assert {"zero_segmentation_plane", "selected_plane_not_acquired"}.issubset(
        codes(decision)
    )


def test_out_of_storage_selection_is_a_review_decision():
    request = yeast_request()
    request = replace(
        request, segmentations=(SegmentationRequest("cell", (ChannelInput(2, (5,)),)),)
    )
    assert "selected_z_out_of_bounds" in codes(decide_assay(request, (baby(),)))


def test_cell_painting_nuclei_and_cell_models_and_colocalization():
    inputs = (ChannelInput(0, (0,)), ChannelInput(1, (0,)))
    request = AssayRequest(
        "plate:A01:site1",
        "raw-manifest-digest",
        1,
        tuple(channel(i, f"dye-{i}", "fluorescence", ["nonzero"]) for i in range(5)),
        (
            SegmentationRequest("nuclei", inputs[:1], target="nucleus"),
            SegmentationRequest("cell", inputs),
        ),
        (MeasurementRequest("colocalization", "cell", inputs, ("pearson", "manders")),),
        assay_type="cell_painting",
    )
    models = (
        ModelCapability(
            "nuclei",
            "weights:nuclei",
            ("fluorescence",),
            1,
            assays=("cell_painting",),
            targets=("nucleus",),
        ),
        ModelCapability(
            "cells",
            "weights:cells",
            ("fluorescence", "fluorescence"),
            1,
            assays=("cell_painting",),
            targets=("cell",),
        ),
    )
    decision = decide_assay(request, models)
    assert decision.ready
    assert dict(decision.selected_models) == {"nuclei": "nuclei", "cell": "cells"}


@pytest.mark.parametrize(
    "change", [{"assays": ("mammalian_brightfield",)}, {"targets": ("nucleus",)}]
)
def test_shape_match_cannot_substitute_for_assay_or_target_compatibility(change):
    model = replace(baby(), **change)
    assert "no_compatible_model" in codes(decide_assay(yeast_request(), (model,)))


def test_overlapping_masks_do_not_silently_enable_colocalization():
    request = yeast_request()
    pair = MeasurementRequest(
        "coloc", "cell", (ChannelInput(0, (0,)), ChannelInput(1, (0,))), ("pearson",)
    )
    decision = decide_assay(replace(request, measurements=(pair,)), (baby(),))
    assert "overlapping_colocalization_unsupported" in codes(decision)


def test_one_plane_shape_does_not_imply_a_projection_is_approved_model_input():
    request = yeast_request()
    bf = channel(2, "BF", "brightfield", ["nonzero"] * 5, tuple(range(5)))
    request = replace(
        request,
        channels=(*request.channels[:2], bf),
        segmentations=(
            SegmentationRequest("cell", (ChannelInput(2, tuple(range(5)), "max"),)),
        ),
    )
    assert "no_compatible_model" in codes(decide_assay(request, (baby(),)))
    declared_projection_model = replace(baby(), reductions=("stack", "max"))
    assert decide_assay(request, (declared_projection_model,)).ready


def test_measurement_projection_must_be_explicit():
    request = yeast_request()
    measurement = replace(
        request.measurements[0], inputs=(ChannelInput(0, (0, 1, 2, 3, 4)),)
    )
    assert "measurement_requires_2d_input" in codes(
        decide_assay(replace(request, measurements=(measurement,)), (baby(),))
    )


def test_bf_selection_preserves_axes_and_other_channel_measurements_without_mutation():
    raw = np.zeros((2, 3, 5, 4, 4), dtype=np.int16)
    raw[:, 0] = np.arange(1, 6)[None, :, None, None]
    raw[:, 1, 0] = -7  # Wrong all-Z max would manufacture zero here.
    raw[:, 2, 0] = 100
    before = raw.copy()
    request = yeast_request()
    bf = select_inputs(raw, request.segmentations[0].inputs)
    red = select_inputs(raw, request.measurements[0].inputs)
    green = select_inputs(raw, request.measurements[1].inputs)
    assert bf.shape == red.shape == green.shape == (2, 1, 1, 4, 4)
    assert np.all(bf == 100) and np.all(red == 5) and np.all(green == -7)
    assert not bf.flags.writeable and not np.shares_memory(bf, raw)
    np.testing.assert_array_equal(raw, before)
    assert raw.flags.writeable


def test_explicit_projection_and_multichannel_selection():
    raw = np.arange(2 * 2 * 5 * 3 * 3).reshape(2, 2, 5, 3, 3)
    inputs = (ChannelInput(1, (1, 2, 3), "mean"), ChannelInput(0, (0,)))
    actual = select_inputs(raw, inputs)
    assert actual.shape == (2, 2, 1, 3, 3)
    np.testing.assert_equal(actual[:, 0, 0], raw[:, 1, 1:4].mean(axis=1))
    np.testing.assert_equal(actual[:, 1, 0], raw[:, 0, 0])


@pytest.mark.parametrize(
    "inputs",
    [
        (ChannelInput(0, (5,)),),
        (ChannelInput(3, (0,)),),
        (ChannelInput(0, (0,)), ChannelInput(1, (0, 1))),
        (ChannelInput(0, (0,)), ChannelInput(0, (1,))),
        (),
    ],
)
def test_actual_pixel_selection_rejects_invalid_requests(inputs):
    with pytest.raises(ValueError):
        select_inputs(np.ones((2, 3, 5, 3, 3)), inputs)


def test_no_nonfinite_filling_or_axis_guessing():
    with pytest.raises(ValueError, match="FCZYX"):
        select_inputs(np.ones((3, 5, 3, 3)), (ChannelInput(0, (0,)),))
    raw = np.ones((2, 1, 5, 3, 3))
    raw[0, 0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        select_inputs(raw, (ChannelInput(0, (0,)),))


@pytest.mark.parametrize("z", [(), (0, 0), (1, 0), (-1,), (True,), (0.0,)])
def test_z_is_explicit_ordered_unique_integer_selection(z):
    with pytest.raises(ValueError):
        ChannelInput(0, z)


@pytest.mark.parametrize(
    "change",
    [
        {"ntps": True},
        {"ntps": 0},
        {"source_identity": ""},
        {"segmentations": ()},
        {"channels": ()},
        {
            "measurements": (
                MeasurementRequest("bad", "absent", (ChannelInput(0, (0,)),)),
            )
        },
        {"segmentations": (SegmentationRequest("cell", (ChannelInput(9, (0,)),)),)},
    ],
)
def test_malformed_requests_fail_before_any_work(change):
    with pytest.raises(ValueError):
        replace(yeast_request(), **change)


def test_duplicate_channels_objects_models_and_sample_times_are_rejected():
    request = yeast_request()
    with pytest.raises(ValueError, match="unique"):
        replace(request, channels=(*request.channels, request.channels[0]))
    with pytest.raises(ValueError, match="unique"):
        replace(request, segmentations=request.segmentations * 2)
    with pytest.raises(ValueError, match="duplicate"):
        replace(request.channels[0], samples=request.channels[0].samples * 2)
    with pytest.raises(ValueError, match="unique"):
        decide_assay(request, (baby(), baby()))


def test_evidence_scope_and_identity_are_explicit():
    with pytest.raises(ValueError, match="evidence"):
        PlaneSample(0, ("nonzero",), "")
    with pytest.raises(ValueError, match="metadata evidence"):
        ChannelEvidence(0, "BF", "brightfield", 1, (), acquired_z=(0,))
    request = yeast_request()
    bad = replace(
        request.channels[2], samples=(PlaneSample(12, ("nonzero",) * 5, "outside"),)
    )
    with pytest.raises(ValueError, match="outside requested"):
        replace(request, channels=(*request.channels[:2], bad))


def test_decisions_are_stable_serializable_immutable_and_evidence_bound():
    request = yeast_request()
    models = [baby(5), baby()]
    decision = decide_assay(request, models)
    assert decision == decide_assay(request, list(reversed(models)))
    assert len(decision.fingerprint) == 64
    assert (
        json.loads(json.dumps(decision.to_dict()))["status"]
        == "compatible_sampled_input"
    )
    assert (
        decide_assay(
            replace(request, source_identity="different-raw"), models
        ).fingerprint
        != decision.fingerprint
    )
    assert (
        decide_assay(
            request, (replace(baby(), identity="different-weights"), baby(5))
        ).fingerprint
        != decision.fingerprint
    )
    assert (
        decide_assay(request, (replace(baby(), stateful=False), baby(5))).fingerprint
        != decision.fingerprint
    )
    models.clear()
    assert len(decision.models) == 2
    with pytest.raises(FrozenInstanceError):
        request.ntps = 2
    supplied = [0, 1]
    inp = ChannelInput(0, supplied)
    supplied.clear()
    assert inp.z == (0, 1)


def test_multichannel_inputs_need_joint_sample_evidence():
    request = yeast_request()
    red = replace(
        request.channels[0],
        samples=(PlaneSample(1, ("nonzero",) * 5, "other-timepoint"),),
    )
    measurement = MeasurementRequest(
        "pair", "cell", (ChannelInput(0, (0,)), ChannelInput(1, (0,)))
    )
    request = replace(
        request, channels=(red, *request.channels[1:]), measurements=(measurement,)
    )
    model = replace(baby(), masks="labels")
    assert "no_joint_channel_sample" in codes(decide_assay(request, (model,)))


def test_documented_example_is_executable_without_models_or_io():
    import re
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "docs/ASSAY_DECISIONS.md").read_text()
    examples = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
    assert len(examples) == 1
    namespace = {}
    exec(compile(examples[0], "ASSAY_DECISIONS.md", "exec"), namespace)
    assert namespace["report"]["status"] == "compatible_sampled_input"
    assert namespace["decision"].require_ready()


def test_hand_edited_decisions_cannot_be_admitted_by_require_ready():
    decision = decide_assay(yeast_request(), (baby(5),))
    with pytest.raises(ValueError, match="not ready or has changed"):
        replace(decision, issues=()).require_ready()
    good = decide_assay(yeast_request(), (baby(),))
    with pytest.raises(ValueError, match="changed"):
        replace(good, selected_models=(("cell", "baby-5z"),)).require_ready()

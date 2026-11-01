"""Computational expansion admission/identity gates; no model or real-data runs."""

import json
from pathlib import Path
import sys

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))
import corpus_assay_trial as trial
import corpus_assay_expansion as driver

sys.path.pop(0)


@pytest.fixture
def scoped(tmp_path, monkeypatch):
    ordinary = [{"deposition": 1, "key": f"site{i:03d}.zip"} for i in range(386)]
    manifest = ordinary + [
        {"deposition": 14191444, "key": "gcd2_002.zip"},
        {"deposition": 14190058, "key": "gcd7_020.zip"},
    ]
    prior = {"jobs": [{"record_key": f"1:site{i:03d}"} for i in range(104)]}
    jobs = [{"key": f"1:site{i:03d}", "family": f"f{i % 22}"} for i in range(104, 386)]
    families = []
    for name in dict.fromkeys(j["family"] for j in jobs):
        keys = [j["key"] for j in jobs if j["family"] == name]
        families.append({"id": name, "jobs": keys, "representative": keys[0]})
    plan = {
        "schema": "aliby.corpus-expansion.v1",
        "authorized_sites": 282,
        "metadata_sha256": "metadata",
        "previous_plan_sha256": "prior",
        "previous_plan": "/work/users/amunoz/aliby_baby/analyses/corpus-rebuild-20261021/batch-plan-v1.json",
        "source_manifest_sha256": "manifest",
        "jobs": jobs,
        "families": families,
        "limits": {
            "workers": 4,
            "baby_workers": 4,
            "site_hours": 4,
            "batch_hours": 48,
            "minimum_free_gib": 300,
        },
    }
    hashes = {
        "metadata.json": "metadata",
        "batch-plan-v1.json": "prior",
        "zenodo_manifest.json": "manifest",
    }
    objects = {
        "plan.json": plan,
        "batch-plan-v1.json": prior,
        "zenodo_manifest.json": manifest,
    }
    monkeypatch.setattr(trial, "load", lambda path: objects[Path(path).name])
    monkeypatch.setattr(trial, "sha", lambda path: hashes[Path(path).name])
    return tmp_path, plan, hashes


def test_exact_complement(scoped):
    root, plan, _ = scoped
    assert len(trial.check_scope(root, require_seal=False)["jobs"]) == 282
    groups = driver.batches(plan)
    assert len(groups[0]) == 1
    assert max(map(len, groups)) <= 4
    assert len([x for g in groups for x in g]) == 282
    assert len(set(x for g in groups for x in g)) == 282


@pytest.mark.parametrize(
    "bad_key", ["14191444:gcd2_002", "14190058:gcd7_020", "1:site000", "1:site104"]
)
def test_protected_and_duplicate_jobs_rejected(scoped, bad_key):
    root, plan, _ = scoped
    plan["jobs"][-1]["key"] = bad_key
    with pytest.raises(ValueError, match="authorized complement"):
        trial.check_scope(root, require_seal=False)


@pytest.mark.parametrize(
    "name", ["metadata.json", "batch-plan-v1.json", "zenodo_manifest.json"]
)
def test_changed_source_inputs_rejected(scoped, name):
    root, _, hashes = scoped
    hashes[name] = "changed"
    with pytest.raises(ValueError, match="changed"):
        trial.check_scope(root, require_seal=False)


@pytest.mark.parametrize(
    "name,value",
    [
        ("workers", 5),
        ("workers", 0),
        ("baby_workers", 5),
        ("site_hours", 5),
        ("batch_hours", 49),
        ("minimum_free_gib", 299),
    ],
)
def test_resource_ceilings(scoped, name, value):
    root, plan, _ = scoped
    plan["limits"][name] = value
    with pytest.raises(ValueError, match="resource"):
        trial.check_scope(root, require_seal=False)


def test_family_cannot_borrow_a_completed_or_other_representative(scoped):
    _, plan, _ = scoped
    plan["families"][0]["representative"] = "14191444:gcd2_002"
    with pytest.raises(ValueError, match="representative"):
        driver.batches(plan)


def test_family_partition_cannot_omit_or_duplicate(scoped):
    _, plan, _ = scoped
    plan["families"][0]["jobs"].append(plan["families"][1]["jobs"][0])
    with pytest.raises(ValueError, match="partition"):
        driver.batches(plan)


def test_family_id_must_match(scoped):
    _, plan, _ = scoped
    plan["jobs"][0]["family"] = "other"
    with pytest.raises(ValueError, match="mismatch"):
        driver.batches(plan)


def test_snapshot_preserves_and_checks_files(tmp_path):
    (tmp_path / "a").write_text("original")
    inventory = trial.snapshot_files(tmp_path)
    trial.verify_files(tmp_path, inventory)
    (tmp_path / "a").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        trial.verify_files(tmp_path, inventory)


@pytest.mark.parametrize("name", ["../escape", "/absolute"])
def test_inventory_path_escape_rejected(tmp_path, name):
    with pytest.raises(ValueError, match="Unsafe"):
        trial.verify_files(tmp_path, {name: "irrelevant"})


def test_symlink_snapshot_rejected(tmp_path):
    (tmp_path / "a").write_text("data")
    (tmp_path / "link").symlink_to(tmp_path / "a")
    with pytest.raises(ValueError, match="Symlink"):
        trial.snapshot_files(tmp_path)


@pytest.fixture
def attempt(tmp_path):
    release = tmp_path / "release"
    release.mkdir()
    (release / "payload").write_text("output")
    trial.save(
        tmp_path / "result.json",
        {
            "status": "computationally_validated",
            "backend": "async",
            "output": str(release),
            "decision_sha256": "decision-test",
            "biological_approval": False,
        },
    )
    trial.save(
        tmp_path / "validated.json",
        {
            "accepted": True,
            "checks": {
                k: True
                for k in (
                    "raw_identity",
                    "model_identity",
                    "input_layout",
                    "semantic_keys",
                    "transport",
                    "roi_geometry",
                    "repeated_masks",
                )
            },
        },
    )
    trial.save(
        tmp_path / "shutdown.json",
        {"owned_service_stopped": True, "owned_pipeline_stopped": True},
    )
    trial.save(
        release / "completion.json",
        {
            "status": "computationally_validated",
            "biological_approval": False,
            "decision_sha256": "decision-test",
            "artifacts_sha256": {"payload": trial.sha(release / "payload")},
        },
    )
    return tmp_path


def rewrite(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def test_fresh_attempt_passes(attempt):
    assert trial.verify_attempt(attempt)[0]["status"] == "computationally_validated"


@pytest.mark.parametrize("key", ["owned_service_stopped", "owned_pipeline_stopped"])
def test_live_ownership_blocks_follow_on(attempt, key):
    rewrite(attempt / "shutdown.json", lambda v: v.update({key: False}))
    with pytest.raises(ValueError, match="not stopped"):
        trial.verify_attempt(attempt)


def test_changed_release_blocks_follow_on(attempt):
    (attempt / "release/payload").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        trial.verify_attempt(attempt)


def test_missing_checks_rejected(attempt):
    rewrite(attempt / "validated.json", lambda v: v.update(checks={}))
    with pytest.raises(ValueError, match="Unvalidated"):
        trial.verify_attempt(attempt)


def test_empty_release_inventory_rejected(attempt):
    rewrite(
        attempt / "release/completion.json", lambda v: v.update(artifacts_sha256={})
    )
    with pytest.raises(ValueError, match="Empty"):
        trial.verify_attempt(attempt)


def test_inventory_rejects_symlinked_parent(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "payload").write_text("unchanged")
    root = tmp_path / "root"
    root.mkdir()
    (root / "nested").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Unsafe"):
        trial.verify_files(root, {"nested/payload": trial.sha(outside / "payload")})


def test_unknown_backend_cannot_skip_completion(attempt):
    rewrite(attempt / "result.json", lambda v: v.update(backend="typo"))
    with pytest.raises(ValueError, match="backend"):
        trial.verify_attempt(attempt)


def test_attempt_cannot_borrow_another_release(attempt, tmp_path):
    import shutil

    other = tmp_path / "borrowed"
    shutil.copytree(attempt / "release", other)
    rewrite(attempt / "result.json", lambda v: v.update(output=str(other)))
    with pytest.raises(ValueError, match="owned"):
        trial.verify_attempt(attempt)


def test_uninventoried_release_file_rejected(attempt):
    (attempt / "release" / "unexpected.npz").write_bytes(b"extra")
    with pytest.raises(ValueError, match="inventory"):
        trial.verify_attempt(attempt)


def test_missing_seal_blocks_execution(scoped):
    root, _, _ = scoped
    with pytest.raises(ValueError, match="seal"):
        trial.check_scope(root)


def test_sequential_diagnostic_has_checked_inventory(attempt):
    output = attempt / "sequential-diagnostic"
    (attempt / "release").rename(output)
    (output / "completion.json").rename(output / "diagnostic-completion.json")
    rewrite(
        attempt / "result.json",
        lambda v: v.update(backend="sequential", output=str(output)),
    )
    trial.verify_attempt(attempt)
    (output / "payload").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        trial.verify_attempt(attempt)


def test_decision_identity_must_match_completion(attempt):
    rewrite(
        attempt / "release/completion.json",
        lambda v: v.update(decision_sha256="different"),
    )
    with pytest.raises(ValueError, match="identity"):
        trial.verify_attempt(attempt)


def test_computation_cannot_claim_biological_approval(attempt):
    rewrite(attempt / "result.json", lambda v: v.update(biological_approval=True))
    with pytest.raises(ValueError, match="Unvalidated"):
        trial.verify_attempt(attempt)

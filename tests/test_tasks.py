import json
from pathlib import Path
import shutil

import pytest

from tasks.catalog import generate
from tasks.pipeline import build, grade
from tasks.sandbox import execute, probe, verify


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    try:
        probe()
    except RuntimeError as error:
        pytest.skip(f"[SKIP] genuine bubblewrap sandbox unavailable: {error}")
    root = tmp_path_factory.mktemp("tasks") / "bundle"
    build(root, variants=1, smoke=True)
    return root


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def attempts_file(tmp_path, task, completions):
    path = tmp_path / "attempts.jsonl"
    path.write_text("".join(json.dumps({"attempt_id": str(i), "task_id": task["task_id"], "completion": c}) + "\n"
                            for i, c in enumerate(completions)))
    return path


def task_from(bundle, split="train"):
    return next(t for t in read_rows(bundle / "private/tasks.jsonl") if t["split"] == split)


def test_build_is_reproducible_and_keeps_sealed_material_private(bundle, tmp_path):
    second = tmp_path / "second"
    build(second, variants=1, smoke=True)
    assert (second / "manifest.json").read_bytes() == (bundle / "manifest.json").read_bytes()
    tasks = read_rows(bundle / "private/tasks.jsonl")
    public = read_rows(bundle / "prompts/train.jsonl") + read_rows(bundle / "prompts/development.jsonl")
    assert {t["task_id"] for t in public}.isdisjoint(t["task_id"] for t in tasks if t["split"] == "evaluation")
    assert all("reference" not in t and "cases" not in t and "mutants" not in t for t in public)
    assert {t["split"] for t in read_rows(bundle / "references-train.jsonl")} == {"train"}
    assert (bundle / "private").stat().st_mode & 0o077 == 0


def test_generation_and_all_mutants_are_real_execution_checks(bundle):
    tasks = list(generate(2, 42))
    ids = {task["task_id"] for task in tasks}
    assert len(ids) == len(tasks)
    owners = {}
    for task in tasks:
        assert verify(task["reference"], task)["status"] == "passed"
        assert all(verify(code, task)["status"] == "wrong_answer" for code in task["mutants"])
        owners.setdefault(task["structure_id"], set()).add(task["split"])
    assert all(len(splits) == 1 for splits in owners.values())


def test_worker_cannot_read_host_files_and_parent_checks_typed_results(bundle, tmp_path):
    secret = tmp_path / "expected-answers.txt"
    secret.write_text("not mounted")
    source = f"def solve(x):\n    try:\n        return open({str(secret)!r}).read()\n    except FileNotFoundError:\n        return 'isolated'\n"
    assert execute(source, [[0]]) == {"status": "completed", "values": ["isolated"]}
    task = {"cases": [{"args": [0], "expected": 1}]}
    assert verify("def solve(x):\n    return True\n", task)["status"] == "wrong_answer"
    forged = "import os\nos.write(1,b'{\"passed\":true}')\nos._exit(0)\n"
    assert execute(forged, [[0]])["status"] == "invalid_output"


@pytest.mark.parametrize("source, statuses", [
    ("def solve(x):\n    while True: pass\n", {"timeout", "execution_error"}),
    ("def solve(x):\n    return 'x' * (512*1024*1024)\n", {"execution_error"}),
    ("def solve(x):\n    print('x' * 100000)\n    return 0\n", {"output_limit"}),
    ("import os\nos.write(1,b'['*2000+b'0'+b']'*2000)\nos._exit(0)\n", {"invalid_output"}),
])
def test_resource_limits_and_adversarial_output(bundle, source, statuses):
    assert execute(source, [[0]])["status"] in statuses


def test_grade_filters_wrong_answers_and_round_duplicates(bundle, tmp_path):
    task = task_from(bundle)
    attempts = attempts_file(tmp_path, task, [task["mutants"][0], task["reference"], task["reference"] + "\n"])
    out = tmp_path / "graded"
    report = grade(bundle, attempts, out, "test-fixture-not-a-model", 0, 0)
    assert report["passed"] == 2 and report["accepted"] == 1
    assert report["verified_unique_examples_per_gpu_hour"] is None
    assert not report["yield_met"] and not report["eligible_for_training"]
    assert [r["status"] for r in read_rows(out / "feedback.jsonl")] == ["wrong_answer", "passed", "passed"]
    second = grade(bundle, attempts, tmp_path / "second", "same-frozen-fixture", 1, 1,
                   previous_accepted=out / "accepted.jsonl")
    assert second["accepted"] == 0


def test_deep_candidate_ast_does_not_abort_feedback_batch(bundle, tmp_path):
    task = task_from(bundle)
    deep = "x = " + "+".join(["1"] * 1000) + "\n" + task["reference"]
    attempts = attempts_file(tmp_path, task, [deep, task["reference"]])
    out = tmp_path / "graded"
    report = grade(bundle, attempts, out, "fixture", 0, 0)
    assert report["accepted"] == 1
    assert [r["status"] for r in read_rows(out / "feedback.jsonl")] == ["invalid_code", "passed"]


def test_escaped_code_payload_is_bounded_without_crashing_grader(bundle):
    task = next(t for t in generate(1) if t["family"] == "word_counts")
    source = "\n" * (32768 - len(task["reference"].encode())) + task["reference"]
    assert verify(source, task)["status"] == "input_limit"


def test_development_successes_never_become_training_data(bundle, tmp_path):
    task = task_from(bundle, "development")
    attempts = attempts_file(tmp_path, task, [task["reference"]])
    out = tmp_path / "graded"
    report = grade(bundle, attempts, out, "fixture", 1, 1, split="development")
    assert report["passed"] == 1 and report["accepted"] == 0
    assert (out / "accepted.jsonl").read_text() == ""
    with pytest.raises(ValueError, match="wrong split"):
        grade(bundle, attempts, tmp_path / "bad", "fixture", 1, 1)


def test_tamper_and_invalid_metadata_never_publish_output(bundle, tmp_path):
    copied = tmp_path / "copy"
    shutil.copytree(bundle, copied)
    with (copied / "private/tasks.jsonl").open("a") as stream:
        stream.write("{}\n")
    task = task_from(bundle)
    attempts = attempts_file(tmp_path, task, [task["reference"]])
    out = tmp_path / "bad"
    with pytest.raises(ValueError, match="hash/path"):
        grade(copied, attempts, out, "fixture", 1, 1)
    assert not out.exists()
    for invalid in (float("nan"), -1, True):
        with pytest.raises(ValueError, match="finite"):
            grade(bundle, attempts, out, "fixture", invalid, 1)


def test_missing_isolation_and_production_evidence_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr("tasks.sandbox.shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="no unsandboxed fallback"):
        build(tmp_path / "missing", smoke=True)
    assert not (tmp_path / "missing").exists()
    with pytest.raises(ValueError, match="production needs"):
        build(tmp_path / "production")

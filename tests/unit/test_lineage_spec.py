"""Tests for the DataHub lineage spec: validation rules and agreement with the DAG sources."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
# `platform` shadows the stdlib module, so the folder is put on the path instead of imported.
sys.path.insert(0, str(ROOT / "platform" / "datahub"))

from lineage_spec import SpecError, load_spec, parse_spec  # noqa: E402

SPEC = load_spec()
PIPELINES = {p.dag_id: p for p in SPEC.pipelines}


def _raw(**task_overrides) -> dict:
    task = {"inputs": ["src"], "outputs": ["dst"]}
    task.update(task_overrides)
    return {
        "datasets": {
            "src": {"platform": "s3", "name": "bucket/src"},
            "dst": {"platform": "postgres", "name": "db.schema.dst"},
        },
        "pipelines": {"dag": {"tasks": {"load": task}}},
    }


def test_minimal_spec_parses_with_defaults():
    spec = parse_spec(_raw())
    assert (spec.env, spec.orchestrator) == ("PROD", "airflow")
    task = spec.pipelines[0].tasks[0]
    assert (task.task_id, task.inputs, task.outputs, task.upstream) == ("load", ("src",), ("dst",), ())


# Equivalence partitions: one invalid spec per validation rule.
@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r["pipelines"]["dag"]["tasks"]["load"].update(inputs=["nope"]), "unknown dataset 'nope'"),
        (lambda r: r["pipelines"]["dag"]["tasks"]["load"].update(outputs=["src", "dst"]), "both input and output"),
        (lambda r: r["pipelines"]["dag"]["tasks"]["load"].update(upstream=["ghost"]), "unknown upstream task 'ghost'"),
        (lambda r: r["pipelines"]["dag"]["tasks"]["load"].update(upstream=["load"]), "unknown upstream task 'load'"),
        (lambda r: r["datasets"].update(extra={"platform": "s3", "name": "bucket/extra"}), "'extra' is not used"),
        (lambda r: r["datasets"].update(dst={"platform": "s3", "name": "bucket/src"}), "s3:bucket/src is declared twice"),
        (lambda r: r["pipelines"].update(empty={}), "empty: has no tasks"),
    ],
    ids=[
        "unknown-dataset",
        "input-is-output",
        "unknown-upstream",
        "self-upstream",
        "unused-dataset",
        "duplicate-dataset",
        "pipeline-without-tasks",
    ],
)
def test_invalid_spec_is_rejected(mutate, message):
    raw = _raw()
    mutate(raw)
    with pytest.raises(SpecError, match=re.escape(message)):
        parse_spec(raw)


def test_all_errors_are_reported_together():
    raw = _raw(inputs=["a"], outputs=["b"])
    with pytest.raises(SpecError) as err:
        parse_spec(raw)
    assert "unknown dataset 'a'" in str(err.value)
    assert "unknown dataset 'b'" in str(err.value)


def test_spec_covers_the_three_data_pipelines():
    assert {"dp1_ingest_bronze", "dp2_silver_gold", "dp3_features"} <= set(PIPELINES)


@pytest.mark.parametrize("dag_id", sorted(PIPELINES))
def test_spec_tasks_match_dag_source(dag_id):
    pipeline = PIPELINES[dag_id]
    source = (ROOT / pipeline.source_file).read_text()
    assert f'dag_id="{dag_id}"' in source
    assert {t.task_id for t in pipeline.tasks} == set(re.findall(r'task_id="([^"]+)"', source))


def test_every_dataset_name_appears_in_pipeline_code():
    code = "\n".join(
        p.read_text()
        for folder in ("platform/spark/jobs", "platform/airflow/dags", "platform/feature_repo")
        for p in (ROOT / folder).glob("*.py")
    )
    missing = []
    for dataset in SPEC.datasets.values():
        needle = {
            "s3": dataset.name,
            "delta-lake": dataset.name,
            "postgres": dataset.name.removeprefix("pairlab."),
            "feast": dataset.name.removeprefix("pairlab."),
        }[dataset.platform]
        if needle not in code:
            missing.append(dataset.name)
    assert missing == []


def test_pipelines_chain_bronze_to_backtest_input():
    """Following outputs → inputs from the vendor bars must reach the backtest table."""
    tasks = [t for p in SPEC.pipelines for t in p.tasks]
    reached = {"vendor_daily_bars"}
    while True:
        new = {o for t in tasks if reached & set(t.inputs) for o in t.outputs} - reached
        if not new:
            break
        reached |= new
    assert {"gold_obt_pair_backtest_input", "online_pair_daily_fv"} <= reached

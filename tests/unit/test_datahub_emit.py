"""Tests for the DataHub proposals built from the lineage spec.

Skipped in the project environment (the DataHub client is not installed there). Run with:
uv run --no-project --with "acryl-datahub[datahub-rest]" --with pyyaml --with pytest \
    pytest tests/unit/test_datahub_emit.py -o addopts=
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("datahub")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "platform" / "datahub"))

from emit_lineage import build_proposals  # noqa: E402
from lineage_spec import load_spec, parse_spec  # noqa: E402

RAW = {
    "datasets": {
        "src": {"platform": "s3", "name": "bucket/src", "description": "source"},
        "dst": {"platform": "postgres", "name": "db.schema.dst"},
    },
    "pipelines": {
        "dag": {
            "tasks": {
                "load": {"inputs": ["src"], "outputs": ["dst"]},
                "check": {"inputs": ["dst"], "upstream": ["load"]},
            }
        }
    },
}
SRC = "urn:li:dataset:(urn:li:dataPlatform:s3,bucket/src,PROD)"
DST = "urn:li:dataset:(urn:li:dataPlatform:postgres,db.schema.dst,PROD)"
LOAD = "urn:li:dataJob:(urn:li:dataFlow:(airflow,dag,prod),load)"
CHECK = "urn:li:dataJob:(urn:li:dataFlow:(airflow,dag,prod),check)"


def _aspects(proposals, name):
    return {p.entityUrn: p.aspect for p in proposals if p.aspect.get_aspect_name() == name}


def test_datasets_flow_and_jobs_are_all_described():
    proposals = build_proposals(parse_spec(RAW))
    assert set(_aspects(proposals, "datasetProperties")) == {SRC, DST}
    assert set(_aspects(proposals, "dataFlowInfo")) == {"urn:li:dataFlow:(airflow,dag,prod)"}
    assert set(_aspects(proposals, "dataJobInfo")) == {LOAD, CHECK}
    assert _aspects(proposals, "datasetProperties")[SRC].description == "source"


def test_task_edges_point_at_dataset_and_upstream_task_urns():
    edges = _aspects(build_proposals(parse_spec(RAW)), "dataJobInputOutput")
    assert (edges[LOAD].inputDatasets, edges[LOAD].outputDatasets) == ([SRC], [DST])
    assert edges[LOAD].inputDatajobs == []
    assert (edges[CHECK].inputDatasets, edges[CHECK].outputDatasets) == ([DST], [])
    assert edges[CHECK].inputDatajobs == [LOAD]


def test_real_spec_builds_valid_proposals():
    spec = load_spec()
    proposals = build_proposals(spec)
    tasks = sum(len(p.tasks) for p in spec.pipelines)
    assert len(proposals) == len(spec.datasets) + len(spec.pipelines) + 2 * tasks
    assert all(p.validate() for p in proposals)

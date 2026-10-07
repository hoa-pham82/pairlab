# /// script
# requires-python = ">=3.11"
# dependencies = ["acryl-datahub[datahub-rest]>=1.3", "pyyaml>=6.0"]
# ///
"""Publish the lineage spec to DataHub: datasets, Airflow DAGs, tasks and their edges.

Run with `uv run platform/datahub/emit_lineage.py --dry-run`; uv installs the
DataHub client into a throwaway environment, not the project one.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datahub.emitter.mce_builder import (
    make_data_flow_urn,
    make_data_job_urn,
    make_dataset_urn,
)
from datahub.emitter.mcp import MetadataChangeProposalWrapper
from datahub.emitter.rest_emitter import DatahubRestEmitter
from datahub.metadata.schema_classes import (
    DataFlowInfoClass,
    DataJobInfoClass,
    DataJobInputOutputClass,
    DatasetPropertiesClass,
)
from lineage_spec import DEFAULT_SPEC, Dataset, LineageSpec, load_spec


def dataset_urn(spec: LineageSpec, dataset: Dataset) -> str:
    """DataHub identifier of a dataset."""
    return make_dataset_urn(dataset.platform, dataset.name, spec.env)


def build_proposals(spec: LineageSpec) -> list[MetadataChangeProposalWrapper]:
    """Turn the spec into DataHub metadata change proposals."""
    cluster = spec.env.lower()
    urns = {key: dataset_urn(spec, d) for key, d in spec.datasets.items()}

    proposals = [
        MetadataChangeProposalWrapper(
            entityUrn=urns[key],
            aspect=DatasetPropertiesClass(name=d.name, description=d.description),
        )
        for key, d in spec.datasets.items()
    ]

    for pipeline in spec.pipelines:
        proposals.append(
            MetadataChangeProposalWrapper(
                entityUrn=make_data_flow_urn(spec.orchestrator, pipeline.dag_id, cluster),
                aspect=DataFlowInfoClass(
                    name=pipeline.dag_id,
                    description=pipeline.description,
                    customProperties={"source_file": pipeline.source_file},
                ),
            )
        )
        for task in pipeline.tasks:
            job_urn = make_data_job_urn(spec.orchestrator, pipeline.dag_id, task.task_id, cluster)
            proposals.append(
                MetadataChangeProposalWrapper(
                    entityUrn=job_urn,
                    aspect=DataJobInfoClass(
                        name=task.task_id, type="COMMAND", description=task.description
                    ),
                )
            )
            proposals.append(
                MetadataChangeProposalWrapper(
                    entityUrn=job_urn,
                    aspect=DataJobInputOutputClass(
                        inputDatasets=[urns[k] for k in task.inputs],
                        outputDatasets=[urns[k] for k in task.outputs],
                        inputDatajobs=[
                            make_data_job_urn(spec.orchestrator, pipeline.dag_id, up, cluster)
                            for up in task.upstream
                        ],
                    ),
                )
            )
    return proposals


def main() -> None:
    """Emit the spec to DataHub, or print it with --dry-run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument(
        "--gms-url", default=os.environ.get("DATAHUB_GMS_URL", "http://localhost:8080")
    )
    parser.add_argument("--dry-run", action="store_true", help="print proposals, send nothing")
    args = parser.parse_args()

    proposals = build_proposals(load_spec(args.spec))
    if args.dry_run:
        print(json.dumps([p.to_obj() for p in proposals], indent=2))
        return

    emitter = DatahubRestEmitter(args.gms_url, token=os.environ.get("DATAHUB_GMS_TOKEN"))
    emitter.test_connection()
    for proposal in proposals:
        emitter.emit(proposal)
    print(f"Emitted {len(proposals)} proposals to {args.gms_url}")


if __name__ == "__main__":
    main()

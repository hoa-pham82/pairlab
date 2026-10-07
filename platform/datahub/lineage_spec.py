"""Load and validate the lineage spec (datasets, pipelines, tasks) from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_SPEC = Path(__file__).with_name("lineage.yaml")


class SpecError(ValueError):
    """The lineage spec is internally inconsistent."""


@dataclass(frozen=True)
class Dataset:
    """A table or file set on one storage platform."""

    key: str
    platform: str
    name: str
    description: str = ""


@dataclass(frozen=True)
class Task:
    """One Airflow task with the datasets it reads and writes."""

    task_id: str
    description: str = ""
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    upstream: tuple[str, ...] = ()


@dataclass(frozen=True)
class Pipeline:
    """One Airflow DAG."""

    dag_id: str
    description: str = ""
    source_file: str = ""
    tasks: tuple[Task, ...] = ()


@dataclass(frozen=True)
class LineageSpec:
    """Everything the emitter publishes."""

    env: str
    orchestrator: str
    datasets: dict[str, Dataset] = field(default_factory=dict)
    pipelines: tuple[Pipeline, ...] = ()


def parse_spec(raw: dict) -> LineageSpec:
    """Build a validated LineageSpec from the parsed YAML mapping."""
    datasets = {
        key: Dataset(key, d["platform"], d["name"], d.get("description", ""))
        for key, d in (raw.get("datasets") or {}).items()
    }
    pipelines = tuple(
        Pipeline(
            dag_id,
            p.get("description", ""),
            p.get("source_file", ""),
            tuple(
                Task(
                    task_id,
                    t.get("description", ""),
                    tuple(t.get("inputs", ())),
                    tuple(t.get("outputs", ())),
                    tuple(t.get("upstream", ())),
                )
                for task_id, t in (p.get("tasks") or {}).items()
            ),
        )
        for dag_id, p in (raw.get("pipelines") or {}).items()
    )
    spec = LineageSpec(raw.get("env", "PROD"), raw.get("orchestrator", "airflow"), datasets, pipelines)
    validate(spec)
    return spec


def load_spec(path: Path = DEFAULT_SPEC) -> LineageSpec:
    """Read and validate a lineage spec file."""
    return parse_spec(yaml.safe_load(Path(path).read_text()))


def validate(spec: LineageSpec) -> None:
    """Raise SpecError listing every inconsistency in the spec."""
    errors: list[str] = []

    names = [(d.platform, d.name) for d in spec.datasets.values()]
    for dup in sorted({n for n in names if names.count(n) > 1}):
        errors.append(f"dataset {dup[0]}:{dup[1]} is declared twice")

    used: set[str] = set()
    for pipeline in spec.pipelines:
        if not pipeline.tasks:
            errors.append(f"{pipeline.dag_id}: has no tasks")
        task_ids = {t.task_id for t in pipeline.tasks}
        for task in pipeline.tasks:
            where = f"{pipeline.dag_id}.{task.task_id}"
            for key in (*task.inputs, *task.outputs):
                if key not in spec.datasets:
                    errors.append(f"{where}: unknown dataset '{key}'")
            for key in sorted(set(task.inputs) & set(task.outputs)):
                errors.append(f"{where}: '{key}' is both input and output")
            for up in task.upstream:
                if up not in task_ids or up == task.task_id:
                    errors.append(f"{where}: unknown upstream task '{up}'")
            used.update(task.inputs, task.outputs)

    for key in sorted(set(spec.datasets) - used):
        errors.append(f"dataset '{key}' is not used by any task")

    if errors:
        raise SpecError("; ".join(errors))

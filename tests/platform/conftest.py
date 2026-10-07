"""Fixtures for platform job tests: a local Spark session and the job modules on sys.path."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

JOBS_DIR = Path(__file__).resolve().parents[2] / "platform" / "spark" / "jobs"


def _java_runs() -> bool:
    """True if a Java runtime answers; macOS ships a /usr/bin/java stub without one."""
    if shutil.which("java") is None:
        return False
    return subprocess.run(["java", "-version"], capture_output=True).returncode == 0


@pytest.fixture(scope="session")
def spark():
    """A single-threaded local SparkSession in UTC; skips when Java or pyspark is missing."""
    if not _java_runs():
        pytest.skip("Java is not installed")
    pyspark_sql = pytest.importorskip("pyspark.sql")
    if str(JOBS_DIR) not in sys.path:
        sys.path.insert(0, str(JOBS_DIR))
    session = (
        pyspark_sql.SparkSession.builder.master("local[1]")
        .appName("pairlab-tests")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()

"""Export the registry model at an alias to a joblib file, for baking into the API image.

Usage:
    uv run python -m services.signal_api.package_model \
        --tracking-uri http://localhost:5001 --out build/model
Prints ``MODEL_VERSION=<name>-v<version>`` on the last line.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from services.signal_api.sources import MlflowModel

MODEL_FILE = "meta_label.joblib"


def export_model(tracking_uri: str, out_dir: str | Path, name: str = "meta_label",
                 alias: str = "production") -> tuple[Path, str]:
    """Write the model at ``name@alias`` to ``out_dir`` and return (path, version)."""
    model = MlflowModel(tracking_uri, name, alias)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / MODEL_FILE
    joblib.dump(model.estimator, path)
    return path, model.version


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--out", default="build/model")
    parser.add_argument("--name", default="meta_label")
    parser.add_argument("--alias", default="production")
    args = parser.parse_args(argv)
    path, version = export_model(args.tracking_uri, args.out, args.name, args.alias)
    print(f"wrote {path}")
    print(f"MODEL_VERSION={version}")


if __name__ == "__main__":
    main()

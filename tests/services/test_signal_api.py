"""Tests for the signal API: validation, decisions, health, metrics, idempotency."""

from __future__ import annotations

import math

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from ml.dataset import PAIR_FEATURES
from services.ab import assign_variant
from services.signal_api.app import build_default_app, create_app
from services.signal_api.sources import FeastFeatureSource, JoblibModel, MlflowModel

from tests.services.conftest import FEATURES, FakeFeatureSource, FakeModel


def _client(prob: float | None = None, threshold: float = 0.5, features=None) -> TestClient:
    source = FakeFeatureSource({"KO__PEP": features or dict(FEATURES)})
    return TestClient(create_app(source, FakeModel(prob), threshold))


class TestSignal:
    def test_returns_probability_decision_and_features(self, signal_client, feature_source):
        body = signal_client.post("/signal", json={"pair_id": "KO__PEP"}).json()
        assert body["pair_id"] == "KO__PEP"
        assert body["prob"] == pytest.approx(1 / (1 + math.exp(-2.3)))
        assert body["take_trade"] is True
        assert body["model_version"] == "test-1"
        assert body["features"] == FEATURES
        assert feature_source.calls == ["KO__PEP"]

    # Boundary values for the decision: probability just below, at, and just
    # above the threshold. A trade is taken when prob >= threshold.
    @pytest.mark.parametrize(
        ("prob", "threshold", "take_trade"),
        [
            (0.49, 0.5, False),
            (0.5, 0.5, True),
            (0.51, 0.5, True),
            (0.0, 0.0, True),
            (1.0, 1.0, True),
            (0.99, 1.0, False),
        ],
    )
    def test_threshold_boundary(self, prob, threshold, take_trade):
        body = _client(prob, threshold).post("/signal", json={"pair_id": "KO__PEP"}).json()
        assert body["take_trade"] is take_trade
        assert body["threshold"] == threshold

    # Equivalence partitions for pair_id:
    #   valid   — two tickers of 1–10 chars (A–Z, 0–9, "."), joined by "__"
    #   invalid — wrong separator, lowercase, empty leg, too long, leading digit,
    #             extra leg, wrong type, missing field
    @pytest.mark.parametrize(
        "pair_id", ["KO__PEP", "A__B", "BRK.B__JPM", "ABCDEFGHIJ__ABCDEFGHIJ", "T1__T2"]
    )
    def test_valid_pair_ids_reach_the_feature_lookup(self, pair_id):
        source = FakeFeatureSource({pair_id: dict(FEATURES)})
        client = TestClient(create_app(source, FakeModel(0.7)))
        assert client.post("/signal", json={"pair_id": pair_id}).status_code == 200

    @pytest.mark.parametrize(
        "payload",
        [
            {"pair_id": "KO_PEP"},
            {"pair_id": "KO-PEP"},
            {"pair_id": "ko__pep"},
            {"pair_id": "__PEP"},
            {"pair_id": "KO__"},
            {"pair_id": ""},
            {"pair_id": "ABCDEFGHIJK__PEP"},
            {"pair_id": "1KO__PEP"},
            {"pair_id": "KO__PEP__XOM"},
            {"pair_id": "KO__PEP; DROP TABLE"},
            {"pair_id": 42},
            {},
        ],
    )
    def test_invalid_requests_are_rejected_before_any_lookup(self, payload):
        source = FakeFeatureSource({})
        client = TestClient(create_app(source, FakeModel(0.7)))
        assert client.post("/signal", json=payload).status_code == 422
        assert source.calls == []

    def test_unknown_pair_is_404(self, signal_client):
        response = signal_client.post("/signal", json={"pair_id": "NOPE__X"})
        assert response.status_code == 404
        assert "NOPE__X" in response.json()["detail"]

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_non_finite_features_are_422_not_a_prediction(self, bad):
        client = _client(0.7, features={**FEATURES, "zscore": bad})
        assert client.post("/signal", json={"pair_id": "KO__PEP"}).status_code == 422

    @pytest.mark.parametrize("threshold", [-0.01, 1.01])
    def test_threshold_outside_unit_interval_rejected(self, threshold):
        with pytest.raises(ValueError):
            create_app(FakeFeatureSource({}), FakeModel(0.5), threshold)


class TestHealth:
    def test_healthz_is_always_ok(self):
        client = TestClient(create_app(None, None))
        assert client.get("/healthz").json() == {"status": "ok"}

    def test_readyz_ready_with_model_and_features(self, signal_client):
        response = signal_client.get("/readyz")
        assert (response.status_code, response.json()) == (200, {"status": "ready"})

    @pytest.mark.parametrize(
        ("features", "model"),
        [(None, FakeModel(0.5)), (FakeFeatureSource({}), None), (None, None)],
    )
    def test_not_ready_without_model_or_features(self, features, model):
        client = TestClient(create_app(features, model))
        assert client.get("/readyz").status_code == 503
        assert client.post("/signal", json={"pair_id": "KO__PEP"}).status_code == 503


class TestMetrics:
    def test_requests_and_decisions_are_counted(self, signal_client):
        signal_client.post("/signal", json={"pair_id": "KO__PEP"})
        signal_client.post("/signal", json={"pair_id": "NOPE__X"})
        text = signal_client.get("/metrics").text
        assert 'signal_api_requests_total{route="/signal",status="200"} 1.0' in text
        assert 'signal_api_requests_total{route="/signal",status="404"} 1.0' in text
        assert (
            'signal_api_decisions_total{model_version="test-1",take_trade="true",'
            'variant="champion"} 1.0'
        ) in text
        assert "signal_api_request_seconds_bucket" in text

    def test_unknown_route_is_counted_as_unmatched(self, signal_client):
        signal_client.get("/nope")
        assert 'route="unmatched",status="404"' in signal_client.get("/metrics").text

    def test_each_app_has_its_own_registry(self, signal_client):
        signal_client.post("/signal", json={"pair_id": "KO__PEP"})
        assert "signal_api_decisions_total{" not in _client(0.5).get("/metrics").text


class TestIdempotency:
    """Property-based: the same request always gives the same answer."""

    @settings(
        max_examples=100, deadline=None, suppress_health_check=[HealthCheck.differing_executors]
    )
    @given(
        values=st.fixed_dictionaries(
            {name: st.floats(-1e6, 1e6, allow_nan=False) for name in PAIR_FEATURES}
        ),
        repeats=st.integers(2, 5),
    )
    def test_repeated_requests_give_identical_responses(self, values, repeats):
        client = _client(features=values)
        responses = [
            client.post("/signal", json={"pair_id": "KO__PEP"}).json() for _ in range(repeats)
        ]
        assert all(r == responses[0] for r in responses)
        assert 0.0 <= responses[0]["prob"] <= 1.0
        assert responses[0]["take_trade"] is (responses[0]["prob"] >= 0.5)

    @settings(
        max_examples=25, deadline=None, suppress_health_check=[HealthCheck.differing_executors]
    )
    @given(zscores=st.lists(st.floats(-5, 5, allow_nan=False), min_size=1, max_size=8))
    def test_real_model_predictions_are_repeatable(self, zscores, trained_model_path):
        model = JoblibModel(trained_model_path)
        for z in zscores:
            row = {**FEATURES, "zscore": z}
            assert model.predict_proba(row) == model.predict_proba(row)


@pytest.fixture(scope="module")
def trained_model_path(tmp_path_factory):
    """A small real LightGBM model saved with joblib."""
    import pandas as pd
    from ml.train import save_model, train_model

    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(120, len(PAIR_FEATURES))), columns=PAIR_FEATURES)
    model = train_model(x, (x["zscore"] > 0).astype(int), params={"n_estimators": 10})
    return save_model(model, tmp_path_factory.mktemp("model") / "m.joblib")


class TestSources:
    def test_joblib_model_scores_and_versions_by_content(self, trained_model_path, tmp_path):
        model = JoblibModel(trained_model_path)
        assert 0.0 <= model.predict_proba(FEATURES) <= 1.0
        assert model.predict_proba({**FEATURES, "zscore": 3.0}) > model.predict_proba(
            {**FEATURES, "zscore": -3.0}
        )
        copy = tmp_path / "copy.joblib"
        copy.write_bytes(trained_model_path.read_bytes())
        assert JoblibModel(copy).version == model.version
        joblib.dump({"other": 1}, tmp_path / "other.joblib")
        assert JoblibModel(tmp_path / "other.joblib").version != model.version

    def test_feast_source_maps_online_response(self):
        class Store:
            def __init__(self, row):
                self.row, self.kwargs = row, None

            def get_online_features(self, **kwargs):
                self.kwargs = kwargs
                return type("R", (), {"to_dict": lambda _s: self.row})()

        store = Store({"symbol_pair": ["KO__PEP"], **{k: [v] for k, v in FEATURES.items()}})
        assert FeastFeatureSource(store).get_features("KO__PEP") == FEATURES
        assert store.kwargs["entity_rows"] == [{"symbol_pair": "KO__PEP"}]
        assert store.kwargs["features"] == [f"pair_daily_fv:{n}" for n in PAIR_FEATURES]

        missing = Store({"symbol_pair": ["X__Y"], **{k: [None] for k in FEATURES}})
        assert FeastFeatureSource(missing).get_features("X__Y") is None


@pytest.fixture(scope="module")
def registry_uri(tmp_path_factory):
    """A local MLflow registry holding one production model trained by the pipeline."""
    import os

    import pandas as pd
    from ml.pipeline import PipelineConfig, run_pipeline

    root = tmp_path_factory.mktemp("mlflow")
    rng = np.random.default_rng(0)
    n = 200
    dates = pd.bdate_range("2022-01-03", periods=n)
    frame = pd.DataFrame(rng.normal(size=(n, len(PAIR_FEATURES))), columns=PAIR_FEATURES).assign(
        pair_date_id=[f"KO__PEP|{d.date()}" for d in dates],
        symbol_pair="KO__PEP",
        event_timestamp=dates,
    )
    frame["label"] = (frame["zscore"] > 0).astype(int)
    cwd = os.getcwd()
    os.chdir(root)
    try:
        cfg = PipelineConfig(
            tracking_uri=f"sqlite:///{root}/mlflow.db",
            data_table_uri=str(root / "frame"),
            horizon_bars=5,
            model_params={"n_estimators": 10},
        )
        run_pipeline(frame, cfg)
    finally:
        os.chdir(cwd)
    return cfg.tracking_uri


class TestMlflowModel:
    def test_loads_the_production_version(self, registry_uri):
        model = MlflowModel(registry_uri, "meta_label")
        assert model.version == "meta_label-v1"
        assert model.predict_proba({**FEATURES, "zscore": 3.0}) > 0.5
        assert model.predict_proba({**FEATURES, "zscore": -3.0}) < 0.5

    def test_missing_alias_is_an_error(self, registry_uri):
        with pytest.raises(Exception, match="challenger"):
            MlflowModel(registry_uri, "meta_label", alias="challenger")

    def test_default_app_reads_the_registry_when_configured(
        self, monkeypatch, tmp_path, registry_uri
    ):
        monkeypatch.setenv("FEAST_REPO", str(tmp_path / "no_repo"))
        monkeypatch.setenv("MLFLOW_TRACKING_URI", registry_uri)
        client = TestClient(build_default_app())
        assert client.get("/readyz").status_code == 503  # model loaded, features missing

    def test_unreachable_registry_starts_not_ready(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FEAST_REPO", str(tmp_path / "no_repo"))
        monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/empty.db")
        monkeypatch.chdir(tmp_path)
        assert TestClient(build_default_app()).get("/readyz").status_code == 503


class TestPackagedModel:
    """CI exports the production model to a file and bakes it into the image."""

    @pytest.fixture
    def packaged(self, registry_uri, tmp_path):
        from services.signal_api.package_model import export_model

        return export_model(registry_uri, tmp_path / "model")

    def test_export_keeps_registry_version_and_predictions(self, packaged, registry_uri):
        path, version = packaged
        assert version == "meta_label-v1"
        exported = JoblibModel(path, version)
        served = MlflowModel(registry_uri, "meta_label")
        for z in (-3.0, 0.0, 3.0):
            row = {**FEATURES, "zscore": z}
            assert exported.predict_proba(row) == served.predict_proba(row)

    def test_cli_prints_the_version_last(self, registry_uri, tmp_path, capsys):
        from services.signal_api.package_model import main

        main(["--tracking-uri", registry_uri, "--out", str(tmp_path / "m")])
        assert capsys.readouterr().out.strip().splitlines()[-1] == "MODEL_VERSION=meta_label-v1"

    # Partitions of MODEL_SOURCE: registry, file (with / without MODEL_VERSION), invalid.
    @pytest.mark.parametrize(
        ("version_env", "expected_prefix"), [("meta_label-v1", "meta_label-v1"), (None, "file-")]
    )
    def test_file_source_wins_over_a_configured_registry(
        self, monkeypatch, packaged, registry_uri, version_env, expected_prefix
    ):
        from services.signal_api.app import _load_model

        monkeypatch.setenv("MLFLOW_TRACKING_URI", registry_uri)
        monkeypatch.setenv("MODEL_SOURCE", "file")
        monkeypatch.setenv("MODEL_PATH", str(packaged[0]))
        if version_env:
            monkeypatch.setenv("MODEL_VERSION", version_env)
        else:
            monkeypatch.delenv("MODEL_VERSION", raising=False)
        assert _load_model().version.startswith(expected_prefix)

    def test_registry_source_is_explicit_too(self, monkeypatch, registry_uri):
        from services.signal_api.app import _load_model

        monkeypatch.setenv("MLFLOW_TRACKING_URI", registry_uri)
        monkeypatch.setenv("MODEL_SOURCE", "registry")
        assert _load_model().version == "meta_label-v1"

    def test_unknown_source_is_rejected(self, monkeypatch):
        from services.signal_api.app import _load_model

        monkeypatch.setenv("MODEL_SOURCE", "s3")
        with pytest.raises(ValueError, match="MODEL_SOURCE"):
            _load_model()


class TestDefaultApp:
    @pytest.fixture(autouse=True)
    def _no_registry(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    def test_missing_model_and_repo_start_not_ready(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FEAST_REPO", str(tmp_path / "no_repo"))
        monkeypatch.setenv("MODEL_PATH", str(tmp_path / "no_model.joblib"))
        client = TestClient(build_default_app())
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 503

    def test_threshold_comes_from_the_environment(self, monkeypatch, tmp_path, trained_model_path):
        monkeypatch.setenv("FEAST_REPO", str(tmp_path / "no_repo"))
        monkeypatch.setenv("MODEL_PATH", str(trained_model_path))
        monkeypatch.setenv("TAKE_THRESHOLD", "1.5")
        with pytest.raises(ValueError):
            build_default_app()


class TestAbRouting:
    PAIRS = [f"P{i}__Q{i}" for i in range(400)]

    def _challenger(self) -> FakeModel:
        model = FakeModel(0.9)
        model.version = "test-2"
        return model

    def _ab_client(self, share: float, pairs=None) -> TestClient:
        rows = {pair: dict(FEATURES) for pair in (pairs or ["KO__PEP"])}
        return TestClient(
            create_app(
                FakeFeatureSource(rows),
                FakeModel(0.1),
                challenger=self._challenger(),
                challenger_share=share,
            )
        )

    # Boundary values for the share: none, all, and the two invalid sides.
    @pytest.mark.parametrize(("share", "variant"), [(0.0, "champion"), (1.0, "challenger")])
    def test_share_boundaries(self, share, variant):
        assert {assign_variant(pair, share) for pair in self.PAIRS} == {variant}

    @pytest.mark.parametrize("share", [-0.01, 1.01])
    def test_invalid_share_rejected(self, share):
        with pytest.raises(ValueError):
            assign_variant("KO__PEP", share)
        with pytest.raises(ValueError):
            create_app(FakeFeatureSource({}), FakeModel(0.5), challenger_share=share)

    def test_split_is_close_to_the_requested_share(self):
        share = sum(assign_variant(pair, 0.3) == "challenger" for pair in self.PAIRS) / 400
        assert 0.22 <= share <= 0.38

    def test_raising_the_share_only_moves_pairs_towards_the_challenger(self):
        low = {pair for pair in self.PAIRS if assign_variant(pair, 0.2) == "challenger"}
        high = {pair for pair in self.PAIRS if assign_variant(pair, 0.6) == "challenger"}
        assert low <= high

    @settings(
        max_examples=100, deadline=None, suppress_health_check=[HealthCheck.differing_executors]
    )
    @given(
        pair=st.from_regex(r"[A-Z]{1,5}__[A-Z]{1,5}", fullmatch=True),
        share=st.floats(0, 1, allow_nan=False),
    )
    def test_assignment_is_stable(self, pair, share):
        assert assign_variant(pair, share) == assign_variant(pair, share)

    def test_response_and_metrics_name_the_variant_that_scored(self):
        client = self._ab_client(1.0)
        body = client.post("/signal", json={"pair_id": "KO__PEP"}).json()
        assert (body["variant"], body["model_version"], body["prob"]) == (
            "challenger",
            "test-2",
            0.9,
        )
        assert (
            'signal_api_decisions_total{model_version="test-2",take_trade="true",'
            'variant="challenger"} 1.0'
        ) in client.get("/metrics").text

    def test_same_pair_always_gets_the_same_variant_over_http(self):
        client = self._ab_client(0.5, pairs=self.PAIRS[:40])

        def variants() -> dict[str, str]:
            return {
                pair: client.post("/signal", json={"pair_id": pair}).json()["variant"]
                for pair in self.PAIRS[:40]
            }

        first = variants()
        assert variants() == first
        assert set(first.values()) == {"champion", "challenger"}

    def test_without_a_challenger_everything_is_champion(self, signal_client):
        body = signal_client.post("/signal", json={"pair_id": "KO__PEP"}).json()
        assert body["variant"] == "champion"

    def test_default_app_loads_a_registry_challenger_when_present(
        self, monkeypatch, tmp_path, registry_uri
    ):
        from mlflow.tracking import MlflowClient

        MlflowClient(registry_uri).set_registered_model_alias("meta_label", "challenger", "1")
        monkeypatch.setenv("FEAST_REPO", str(tmp_path / "no_repo"))
        monkeypatch.setenv("MLFLOW_TRACKING_URI", registry_uri)
        monkeypatch.setenv("CHALLENGER_SHARE", "2")
        with pytest.raises(ValueError):
            build_default_app()

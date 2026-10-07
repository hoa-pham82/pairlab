# Low-level ML design

Five components carry the ML side of the project. Signatures below are the
real ones in the code.

## 1. Label builder — `ml/labels.py`

Turns z-score history into the two-column label table.

```python
@dataclass(frozen=True)
class LabelConfig:
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    horizon_bars: int = 20

def label_entries(zscore: pd.Series, cfg: LabelConfig | None = None) -> pd.Series: ...
def build_label_table(features: pd.DataFrame, cfg: LabelConfig | None = None) -> pd.DataFrame: ...
def pair_date_id(symbol_a: str, symbol_b: str, ts: pd.Timestamp) -> str: ...
```

- Invalid thresholds are rejected when the config is created.
- Entries whose outcome is not known yet are dropped, not labelled 0.

## 2. Training data — `ml/dataset.py`

Joins labels to features and splits without leakage.

```python
def entity_frame(labels: pd.DataFrame) -> pd.DataFrame: ...
def load_training_frame(store, labels: pd.DataFrame) -> pd.DataFrame: ...
def purged_time_split(frame, valid_fraction=0.25, horizon_bars=20) -> tuple[pd.DataFrame, pd.DataFrame]: ...
def features_and_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]: ...
```

- `store` is anything with `get_historical_features`, so tests pass a fake.
- The split removes training rows whose label window reaches validation.

## 3. Data versioning — `ml/versioning.py`

Stores each training table as a Delta version, writing only what changed.

```python
@dataclass(frozen=True)
class DataVersion:
    version: int
    rows_inserted: int
    rows_updated: int
    rows_deleted: int

def snapshot_training_frame(frame, table_uri, storage_options=None) -> DataVersion: ...
def load_training_frame_version(table_uri, version, storage_options=None) -> pd.DataFrame: ...
```

## 4. Training pipeline — `ml/pipeline.py`

Runs the steps in order and records them in MLflow.

```python
@dataclass(frozen=True)
class PipelineConfig:
    tracking_uri: str
    data_table_uri: str
    experiment: str = "meta_label"
    model_name: str = "meta_label"
    valid_fraction: float = 0.25
    horizon_bars: int = 20

@dataclass(frozen=True)
class PipelineResult:
    run_id: str
    model_version: str
    alias: str            # "production" or "challenger"
    data_version: DataVersion
    metrics: dict[str, float]

def run_pipeline(frame: pd.DataFrame, cfg: PipelineConfig) -> PipelineResult: ...
```

- Model training and scoring live in `ml/train.py` (`train_model`, `evaluate`).
- A new model takes `production` only if its AUC is at least the current one's.

## 5. Serving ports — `services/signal_api/sources.py`, `services/regime_api/sources.py`

The APIs depend on small interfaces, not on Feast, MLflow or Postgres directly.

```python
class FeatureSource(Protocol):
    def get_features(self, pair_id: str) -> dict[str, float] | None: ...

class Model(Protocol):
    version: str
    def predict_proba(self, features: dict[str, float]) -> float: ...

class PriceSource(Protocol):
    def get_closes(self, pair_id: str, n_bars: int) -> tuple[np.ndarray, np.ndarray] | None: ...
    def ping(self) -> bool: ...

def create_app(features: FeatureSource | None, model: Model | None, threshold: float = 0.5) -> FastAPI: ...
```

| Interface | Real implementation | Test implementation |
|---|---|---|
| `FeatureSource` | `FeastFeatureSource` (Redis online store) | `FakeFeatureSource` (dict) |
| `Model` | `MlflowModel` (registry), `JoblibModel` (file) | `FakeModel` |
| `PriceSource` | `PostgresPriceSource` | `FakePriceSource` |

## How they connect

```
gold.feat_pair_daily ──► label builder ──► gold.label_pair_reversion
                                                   │
Feast offline store ──► training data ◄────────────┘
                              │
                              ├──► data versioning ──► Delta table (version N)
                              ▼
                      training pipeline ──► MLflow registry (meta_label@production)
                                                   │
Feast online store ──► signal_api ◄────────────────┘
warehouse prices   ──► regime_api
```

## Design patterns used

| Pattern | Where | Why |
|---|---|---|
| Ports and adapters (dependency inversion) | `FeatureSource`, `Model`, `PriceSource` protocols with Feast, MLflow, joblib and Postgres adapters | The APIs can be tested with no running infrastructure, and a store can be swapped without touching endpoint code |
| Factory | `create_app(...)` and `build_default_app()` in both services | One function builds a fully wired app; tests build as many isolated apps as they need |
| Strategy | `_load_model()` picks `MlflowModel` or `JoblibModel` from configuration | The serving code is the same whichever source the model comes from |
| Template method | `_ClassifierModel.predict_proba` shared by `JoblibModel` and `MlflowModel` | Subclasses only say how to load; scoring is written once |
| Value object | Frozen dataclasses: `LabelConfig`, `DataVersion`, `PipelineConfig`, `PipelineResult`, `RegimeThresholds`, `RegimeAssessment` | Immutable, comparable results; invalid settings fail at construction |

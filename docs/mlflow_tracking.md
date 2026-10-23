# MLflow experiment tracking and model registry

Day 11 tracks the frozen Day 10 research model in a local MLflow experiment.
The local backend is `mlflow.db`, artifacts are stored under `mlartifacts/`, and
both are ignored by Git. The selected pipeline is registered as
`CreditRiskEarlyWarning` and the service-facing version receives the
`champion` alias.

The registered model contains the fitted preprocessing and logistic-regression
classifier as one scikit-learn pipeline. A second MLflow artifact records the
fitted preprocessor on its own. The run also records:

- hyperparameters, cross-validation configuration, and split metadata;
- validation discrimination, classification, and calibration metrics;
- the feature contract and training configuration as structured JSON;
- validation predictions, data definitions, evaluation reports, explanations,
  and evaluation plots;
- the frozen classification configuration and original artifact checksum.

The held-out test split is not read by the tracking command. It remains unused
for model selection and warning-threshold tuning. The target remains the South
German Credit `bad_credit` prototype proxy and must not be described as a
validated 90-day probability of default.

## Reproduce the registered run

From the repository root in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.data_preparation
python -m src.train
python -m src.optimize
python -m src.predict freeze
python -m src.mlflow_tracking
```

The first four commands reproduce the prerequisite split, training,
optimization, and frozen Day 10 artifacts. The last command creates the MLflow
experiment, logs the selected model and evidence, registers a new model version,
and moves the `champion` alias to that version. Re-running it creates another
immutable experiment run and model version.

To inspect the experiment and model registry locally:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000`. Stop the UI with `Ctrl+C`.

## Prediction-service loading

The service loader resolves the alias instead of embedding a numeric model
version:

```python
from src.model_registry import load_registered_model

bundle = load_registered_model()
classifier = bundle.build_classifier()
```

Before returning a pipeline, the loader verifies the registered model version's
configuration version, original frozen-model checksum, and `test_split_used`
tag against `config/risk_classification.json`. Set `MLFLOW_TRACKING_URI` only
when using another compatible tracking server.

# FaultScope

Calibrated industrial fault classification with a review policy.

## Executed result

Held-out macro-F1 **0.781** and accuracy **78.15%** on 389 test records. Accepted-case accuracy **89.36%** at **60.41%** coverage.

## Requirements and installation

- Python **3.12**, Git, and a terminal (Windows PowerShell, macOS, or Linux).
- Internet access for the initial package/data download. No paid API key is required.
- Create and activate a virtual environment in this project folder:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate
python -m pip install --upgrade pip
```

## Problem and implementation

Route measured steel-plate faults to one of seven defect categories while flagging low-confidence cases for review. This is a tabular classification benchmark over 27 sensor/image-derived features, not an image detector or a deployed factory inspection system.

The pipeline compares a majority baseline, standardized logistic regression, random forest and histogram gradient boosting. Selection uses validation macro-F1. The winner is calibrated with sigmoid calibration and three training-only folds. A confidence threshold is chosen on validation data to maximize coverage among thresholds meeting 90% observed validation accuracy and at least 25% coverage; if none qualify, threshold 0 disables abstention. The target is not a guarantee on unseen data.

## Run

```bash
pip install -r requirements.txt
python app.py train
python app.py predict --input results/example_request.json
python app.py serve --port 8012
```

Open http://localhost:8012. Training downloads a small UCI archive and runs on CPU; allow a few minutes and roughly 2 GB RAM. GET /health and POST /predict are available. The request is `{"features":[...]}` with exactly 27 finite numbers in the FEATURES order in app.py; results/example_request.json is an actual held-out example. A full probability map and review_required flag accompany each prediction.

## Evaluation protocol and limits

- 1,941 rows; fixed seed 42; stratified train/validation/test sizes 1,164/388/389. Feature-vector duplicates are removed before splitting. Original data have no removed duplicates.
- Preprocessing is fitted only within training. Model selection and review-threshold selection never use test labels. Selected calibrated model: random forest.
- Test macro-F1 is complemented by per-class precision/recall, confusion matrix, log loss, accepted accuracy and coverage. The 1,000-resample bootstrap interval measures test-row variability, not seed variability.
- Feature importance is permutation-based on validation data, not causal attribution. Machine/batch identifiers are unavailable; random splits may overstate generalization to new equipment.
- The 90% validation target became about 89.36% accepted accuracy on held-out data at 60.41% coverage. Both numbers must be reported together.
- Latency uses 100 warmed batch-1 predictions and reports P50/P95; loading and network are excluded.

## Source

[UCI Steel Plates Faults](https://archive.ics.uci.edu/dataset/198/steel+plates+faults), 27 features and seven classes. UCI lists CC BY 4.0. Cite the dataset and keep its terms if redistributing. Source ZIP SHA-256 and split indices are stored with results. Raw measurements and model binaries are excluded from the source archive.

## Tests and reproducibility

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

`results/metrics.json` contains the executed protocol, environment and measured values; `results/dashboard.html` is the self-contained searchable report. Source code, tests, input schema, sample request, and GitHub Actions are included. CI runs utility tests; it does not train/download the entire benchmark on every push. Hosted CI starts after publication. Python 3.12 and pinned top-level packages were used for the recorded run. CPU thread limits affect timing; exact speed is not portable.

Dockerfile is supplied for packaging, but a Docker build was not executed in this environment. Local Python and the HTTP/browser demo are the validated paths. Bindings are localhost-only; the server has no authentication, TLS, rate limiting or production deployment claim. Load only locally generated trusted joblib/checkpoint artifacts. Training/downloading must finish before serving.

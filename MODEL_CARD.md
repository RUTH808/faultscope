# FaultScope model and evaluation card

Held-out macro-F1 **0.781** and accuracy **78.15%** on 389 test records. Accepted-case accuracy **89.36%** at **60.41%** coverage.

## Evaluation protocol and limits

- 1,941 rows; fixed seed 42; stratified train/validation/test sizes 1,164/388/389. Feature-vector duplicates are removed before splitting. Original data have no removed duplicates.
- Preprocessing is fitted only within training. Model selection and review-threshold selection never use test labels. Selected calibrated model: random forest.
- Test macro-F1 is complemented by per-class precision/recall, confusion matrix, log loss, accepted accuracy and coverage. The 1,000-resample bootstrap interval measures test-row variability, not seed variability.
- Feature importance is permutation-based on validation data, not causal attribution. Machine/batch identifiers are unavailable; random splits may overstate generalization to new equipment.
- The 90% validation target became about 89.36% accepted accuracy on held-out data at 60.41% coverage. Both numbers must be reported together.
- Latency uses 100 warmed batch-1 predictions and reports P50/P95; loading and network are excluded.

## Source

[UCI Steel Plates Faults](https://archive.ics.uci.edu/dataset/198/steel+plates+faults), 27 features and seven classes. UCI lists CC BY 4.0. Cite the dataset and keep its terms if redistributing. Source ZIP SHA-256 and split indices are stored with results. Raw measurements and model binaries are excluded from the source archive.

## Intended use

Educational portfolio, reproducible offline evaluation, and local interactive demonstration. No autonomous high-impact decisions. Results apply to the named benchmark under the recorded split and hardware. See README for dependencies and run commands.

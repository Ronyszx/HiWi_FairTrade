# Setup Notes

Environment used:
- Python 3.11.15
- macOS ARM
- torch 2.0.1
- torchvision 0.15.2
- scikit-learn 1.2.2

Changes/issues:
- Original requirements.txt used scikit-learn==0.24.2, which failed to build on Python 3.11/macOS ARM.
- Updated scikit-learn to 1.2.2.
- crypten required deprecated sklearn package installation, so SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True was used during installation.
- pip check reports: "torch 2.0.1 is not supported on this platform", but torch and torchvision import successfully and MPS is available.

PsmPy compatibility:
-The original code used psm.knn_matched(), but the installed psmpy version exposes kdtree_matched() instead. I added a compatibility check so the code supports both method names.

Runtime compatibility/code issue:
- FairTrade.py called evaluate() expecting three return values, but evaluate() returned only the objective tensor.
- I fixed the call site by assigning only `objectives`, since the following code already extracts fairness and balanced accuracy from that tensor.
- Full reasoning is documented in BUG_HUNT.md.

Task 2 evaluation support:
- Added and pinned `pytest==9.1.1`, using the exact version installed in the project virtual environment.
- Added and pinned the installed headless plotting dependency, `matplotlib==3.11.0`.
- Task 2 is opt-in through `--task2_evaluation`; omitting the flag preserves Task 1 training and output behavior.
- Sensitive encodings are inferred from the exact Adult CSV before group names are assigned.
- The full Task 2 run regenerated Task 1 arrays with the same previously recorded SHA-256 hashes.

Baseline reproducibility and portability:
- Added `--seed` with a default of 42 and seeded Python, NumPy, PyTorch, and CUDA when available before loading data or creating the model.
- Preserved all upstream dataset splits that explicitly use `random_state=42`.
- Added `--device` with validated `auto`, `cpu`, `mps`, and `cuda` options. Auto selection prefers CUDA, then MPS, then CPU.
- Added automatic creation of `results/<dataset_name>/` before saving result arrays.
- Added the installed `psmpy==0.3.16` version to `requirements.txt` and synchronized the README dependencies.
- These changes do not alter the training mathematics, fairness objective, model architecture, evaluation split, or MOBO candidate loop.

Task 3 multi-attribute support:
- Added opt-in `--task3_multi_attribute`, restricted to the Adult/statistical-parity/random configuration and mutually exclusive with `--task2_evaluation`.
- Prepared aligned gender and binary race vectors once after dataset loading. Adult race code `4` is White; codes `0` through `3` are grouped as Non-White.
- Validated that every client and the test split contain both groups for each sensitive attribute before training.
- Combined the two local differentiable demographic-parity losses as `0.5 * gender_loss + 0.5 * race_loss`.
- Used rounded predictions only for the MOBO objective `[-max(abs(gender_spd), abs(race_spd)), balanced_accuracy]`, with qEHVI reference point `[-1.01, -0.01]`.
- Kept candidate evaluations out of the four start-of-round history arrays and isolated every Task 3 artifact under `results/task3/`.
- Reused the Task 2 intersectional evaluator exactly once for final Task 3 predictions.
- A Task 3-only `ModelFittingError` fallback retains the previous valid alpha and learning rate, skips the remaining MOBO iterations in the affected communication round, and continues with the next round. Baseline behavior is unchanged.
- The full seed-42 CPU `15/50/10` Task 3 experiment completed with one recovered GP fit failure.
- Preserved the existing Sigmoid, `BCEWithLogitsLoss`, second constraint sigmoid, MOBO test-split use, repeated candidate loop, and weighted candidate selection for a direct baseline comparison.

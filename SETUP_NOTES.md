# Setup Notes

This file records the environment and compatibility decisions used for the HiWi challenge. New users should follow the step-by-step **HiWi Challenge Reproduction** section in `README.md`.

## Verified environment

- Branch: `rony-challenge-work`
- Python: 3.11.15
- Platform: macOS ARM
- Execution device for reported runs: CPU
- NumPy: 1.26.4
- SciPy: 1.17.1
- PyTorch: 2.0.1
- TorchVision: 0.15.2
- pandas: 1.5.3
- scikit-learn: 1.2.2
- BoTorch: 0.8.5
- GPyTorch: 1.10
- CrypTen: 0.4.1
- PsmPy: 0.3.16
- matplotlib: 3.11.0
- pytest: 9.1.1

Python 3.11 is part of the reproducibility contract. The pinned research stack has not been validated on newer Python versions. The README uses CPU for the portable path and scopes its shell commands to macOS/Linux or WSL2.

## Fresh-install safeguards

- Clone `rony-challenge-work` explicitly instead of relying on the repository's default branch.
- Run all commands from the repository root because dataset paths are relative to that directory.
- Create the environment with `python3.11 -m venv .venv`.
- Install with `SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True` because CrypTen declares the deprecated `sklearn` package name.
- Pin NumPy 1.26.4 and SciPy 1.17.1 as used by the successful environment. This avoids an unconstrained numerical stack being selected around pandas 1.5.3, scikit-learn 1.2.2, and BoTorch 0.8.5.
- Use `--device cpu` for the documented reproduction. Explicit CUDA and MPS choices validate device availability before training.
- Run imports, CLI help, tests, and the short Adult smoke test before starting a full experiment.

## Verification performed

- Core dependency imports succeed in the verified virtual environment.
- A new temporary Python 3.11 environment installed every package from `requirements.txt` from scratch and passed the same import checks, CLI check, tests, and Adult smoke test.
- `python FairTrade.py --help` displays all baseline, seed, device, Task 2, and Task 3 options.
- `python -m pytest -q` passes all 22 tests.
- The tests currently emit nine pandas warnings about deprecated NumPy APIs; they do not fail the suite.
- The Adult CPU smoke test completes with seed 42.
- The full Task 1, Task 2, and Task 3 seed-42 CPU experiments completed and their final artifacts are tracked under `results/`.
- `pip check` on the verified macOS ARM environment reports `torch 2.0.1 is not supported on this platform`. Torch and TorchVision still import, and CPU execution completed. Other dependency errors are not expected.

## Compatibility changes

### scikit-learn

The original `scikit-learn==0.24.2` failed to build on Python 3.11/macOS ARM. It was updated to `scikit-learn==1.2.2`.

### PsmPy

The original code used `psm.knn_matched()`, while PsmPy 0.3.16 exposes `kdtree_matched()`. The loader contains a compatibility check for both method names.

### Baseline runtime bug

`FairTrade.py` called `evaluate()` expecting three return values even though it returned only the objective tensor. The call site now assigns only `objectives`; following code already extracts fairness and balanced accuracy from its two columns. The full reasoning is documented in `BUG_HUNT.md`.

## Reproducibility and portability changes

- `--seed` defaults to 42 and seeds Python, NumPy, PyTorch, and CUDA when available before data loading and model creation.
- Existing dataset splits with explicit `random_state=42` remain unchanged.
- `--device` accepts validated `auto`, `cpu`, `mps`, and `cuda` values. Auto prefers CUDA, then MPS, then CPU.
- Result directories are created automatically before arrays, CSV files, or charts are saved.
- These changes do not alter training mathematics, fairness objectives, model architecture, evaluation splits, or the baseline MOBO candidate loop.

## Task 2 evaluation support

- Task 2 is opt-in through `--task2_evaluation`; omitting the flag preserves Task 1 behavior.
- Sensitive encodings are verified from the included Adult CSV before group names are assigned.
- `pytest==9.1.1` and the headless plotting dependency `matplotlib==3.11.0` are pinned.
- The full Task 2 run reproduced the previously recorded Task 1 array hashes.

## Task 3 multi-attribute support

- Task 3 is opt-in through `--task3_multi_attribute` and requires Adult, statistical parity, and random client distribution.
- Adult race code 4 is White; codes 0 through 3 are grouped as Non-White.
- Every client and the test split are validated to contain both gender and binary-race groups.
- Local fairness uses `0.5 * gender_loss + 0.5 * race_loss`.
- MOBO uses rounded predictions for `[-max(abs(gender_spd), abs(race_spd)), balanced_accuracy]` with reference point `[-1.01, -0.01]`.
- Candidate evaluations are excluded from the four start-of-round history arrays, and Task 3 outputs stay under `results/task3/`.
- Final intersectional evaluation reuses the Task 2 evaluator once after training.
- A Task 3-only `ModelFittingError` fallback keeps the previous valid alpha and learning rate, skips the remaining search in that communication round, and continues training.
- The full seed-42 CPU `15/50/10` experiment completed with one recovered GP fitting failure.

## Preserved methodological limitations

For direct Task 1 versus Task 3 comparison, the challenge work preserves the existing final Sigmoid, `BCEWithLogitsLoss`, second sigmoid in `ConstraintLoss`, MOBO use of the test split, repeated candidate loop, and weighted candidate selection. These are documented limitations rather than silent fixes.

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

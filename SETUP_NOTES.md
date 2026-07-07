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

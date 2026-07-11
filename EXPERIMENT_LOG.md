## Adult CPU smoke test

Purpose: Verify baseline execution and reproducibility before the full experiment.

Configuration:

- Dataset: Adult
- Sensitive attribute: gender
- Fairness notion: statistical parity
- Clients: 3
- Epochs: 1
- Communication rounds: 2
- MOBO rounds: 1
- Seed: 42
- Device: CPU

Two identical runs produced the same final metrics:

- Test accuracy: 0.23598955934285276
- Balanced accuracy: 0.5
- Statistical parity: 0.0
- ATE: 0.001150837108025049

Saved balanced-accuracy and statistical-parity arrays were byte-identical.
Complete logs differed only in a temporary Matplotlib cache-directory name.
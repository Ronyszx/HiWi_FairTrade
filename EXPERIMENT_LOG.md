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

## Task 1 - Adult Baseline Reproduction

Purpose: Reproduce the FairTrade Adult baseline with the default training configuration and a fixed seed.

### Configuration

- Dataset: Adult
- Sensitive attribute: gender (`sex`)
- Fairness notion: statistical parity
- Clients: 3
- Client epochs: 15
- Communication rounds: 50
- MOBO rounds per communication round: 10
- Distribution: random
- Seed: 42
- Device: CPU
- Run date: 2026-07-11
- Git commit: `808fdc7`

Command:

```bash
python FairTrade.py \
  --dataset_name adult \
  --fairness_notion stat_parity \
  --num_clients 3 \
  --epochs 15 \
  --communication_rounds 50 \
  --mobo_optimization_rounds 10 \
  --distribution_type random \
  --seed 42 \
  --device cpu
```

### Results

| Metric | Value | Interpretation |
| --- | ---: | --- |
| Final balanced accuracy | 0.7690318212 | Final model after the last MOBO evaluation |
| Final statistical parity difference | 0.0284339752 | Final printed signed SPD |
| Last-10 balanced accuracy | 0.7706942707 +/- 0.0016498609 | Mean and population standard deviation of start-of-round evaluations |
| Last-10 SPD | 0.0271854400 +/- 0.0036096978 | Mean and population standard deviation of start-of-round evaluations |

The balanced accuracy closely matches the value near 0.77 in the Adult R3C demographic-parity comparison (Table 1 of the [FairTrade paper](https://doi.org/10.1609/aaai.v38i10.28971)). The obtained SPD of 0.0284 is materially higher than the reported value near 0.001. The challenge names Table 2, but that table evaluates the causal FACE objective; Table 1 is the like-for-like comparison for this `stat_parity` run. Exact numerical reproduction is not required, so the difference is retained and reported rather than hidden or tuned away.

The saved arrays contain one evaluation recorded at the start of each communication round, before that round's inner MOBO evaluations. Consequently, their final value and last-10 summary do not exactly equal the final metrics printed after the last MOBO evaluation. The final model values above come from the final lines of the log.

### MOBO and the Pareto front

FairTrade treats the fairness regularization strength (`alpha`) and learning rate as inputs to multi-objective Bayesian optimization. Evaluating a candidate runs federated training and returns two objectives: negative signed SPD and balanced accuracy. Separate Gaussian-process surrogate models approximate these objectives, and qExpectedHypervolumeImprovement proposes candidates expected to expand the objective-space hypervolume relative to a reference point. The Pareto front contains non-dominated candidates: improving one objective from such a point requires worsening the other. This exposes the accuracy-fairness trade-off instead of assuming that one candidate is universally best. The implementation subsequently uses a fixed weighted sum to choose one observed candidate for the next communication round.

### Warnings and provenance

The run completed all 50 communication rounds and produced finite 50-element arrays. BoTorch emitted nonfatal float-precision and scaling warnings, 301 random-initial-candidate fallbacks, 21 SciPy GP-fit optimization failure warnings, six candidate-optimization failures, and Cholesky jitter warnings. These indicate imperfect numerical conditioning but did not stop training or prevent convergence.

- Full log: `results/task1/adult_seed42_full_run1.log` (kept locally, not tracked)
- Balanced-accuracy array: `results/adult/3_bal_acc_stat_parity.npy` (kept locally, not tracked)
- Statistical-parity objective array: `results/adult/3_stat_parity.npy` (kept locally, not tracked)
- Log SHA-256: `fc6f6d0733c365f963b03d9216b0e49d0a22ebe4ee4422632ac0d4beace15492`
- Balanced-accuracy array SHA-256: `2d51b7374e5dce81cac53cf577e246a559f16be5de4d71d3680e4f67048cd376`
- Statistical-parity objective array SHA-256: `3ad207641e555121103e96fc6c7a6667feab623872a0eac8bd556f7d329b74ef`

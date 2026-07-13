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

## Task 2 - Adult Intersectional Fairness Evaluation

Purpose: Evaluate the final Task 1 model across gender, race, and their four requested intersections without changing training or optimization.

### Configuration

- Dataset: Adult
- Training sensitive attribute: gender (`sex`)
- Clients: 3
- Client epochs: 15
- Communication rounds: 50
- MOBO rounds per communication round: 10
- Distribution: random
- Seed: 42
- Device: CPU
- Evaluation split: existing held-out Adult test split (`20%`, `random_state=42`)
- Run date: 2026-07-12
- Base commit: `6458cbe` with uncommitted Task 2 working-tree changes

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
  --device cpu \
  --task2_evaluation
```

### Verified encodings and definitions

- Gender: `Female=0`, `Male=1`
- Race: `Amer-Indian-Eskimo=0`, `Asian-Pac-Islander=1`, `Black=2`, `Other=3`, `White=4`
- Gender signed SPD: `PPR(Male) - PPR(Female)`
- Race signed SPD: `PPR(White) - PPR(Non-White)`
- Intersectional signed SPD: `PPR(White Male) - PPR(group)`
- Intersectional max-min gap: maximum four-group PPR minus minimum four-group PPR

The mappings were inferred from the exact Adult CSV with `LabelEncoder`, not assumed from numeric values. Non-White combines the four non-White dataset categories as required for the four-group challenge comparison.

### Results

| Metric / group | Positive prediction rate | Signed SPD | Absolute SPD / gap | Count |
| --- | ---: | ---: | ---: | ---: |
| Male | 0.4733944954 | - | - | 4,360 |
| Female | 0.4449605202 | - | - | 2,153 |
| Gender: Male vs Female | - | 0.0284339752 | 0.0284339752 | 6,513 |
| White | 0.4908405172 | - | - | 5,568 |
| Non-White | 0.3058201058 | - | - | 945 |
| Race: White vs Non-White | - | 0.1850204114 | 0.1850204114 | 6,513 |
| White Male | 0.4960896767 | 0.0000000000 | 0.0000000000 | 3,836 |
| White Female | 0.4792147806 | 0.0168748961 | 0.0168748961 | 1,732 |
| Non-White Male | 0.3072519084 | 0.1888377683 | 0.1888377683 | 524 |
| Non-White Female | 0.3040380048 | 0.1920516720 | 0.1920516720 | 421 |
| Intersectional max-min gap | - | - | 0.1920516720 | 6,513 |

The gender SPD is only 2.84 percentage points, but White and Non-White positive prediction rates differ by 18.50 points. The largest intersectional difference is 19.21 points between White Male and Non-White Female. Therefore, the aggregate gender result alone would conceal a much larger race-associated and intersectional disparity. The Non-White aggregate can itself hide differences among its four constituent race categories, which remains a limitation of the challenge's requested binary race grouping.

### Verification and provenance

- Tests: `8 passed` with `.venv/bin/python -m pytest -q`
- Task 2 hook executions in the full log: `1`
- Existing Task 1 balanced-accuracy array SHA-256: `2d51b7374e5dce81cac53cf577e246a559f16be5de4d71d3680e4f67048cd376` (unchanged)
- Existing Task 1 SPD-objective array SHA-256: `3ad207641e555121103e96fc6c7a6667feab623872a0eac8bd556f7d329b74ef` (unchanged)
- Full Task 2 log: `results/task2/adult_seed42_task2_simplified.log` (kept locally, not tracked)
- Full log SHA-256: `d3440b5e64c12e64b2ed86849eb6cc078ce47f98a31878f158a04a3944bf3566`
- Metrics CSV SHA-256: `cdb3e65671fbfaaa05c876d05008790fb4e5a8228c7d7debd08f4babd386b1d7`
- Chart SHA-256: `75f2d6b32b833805c84ffa01265f75e3e85c495dd7826f564d984d45489d7f9d`

The full run retained the known nonfatal BoTorch warnings: 301 random initial-candidate fallbacks, 21 GP-fit optimization failures, six candidate-generation optimization failures, and 12 Cholesky jitter warnings. Task 2 introduced no additional runtime warning or training failure. `pip check` continues to report the pre-existing `torch 2.0.1 is not supported on this platform` warning.

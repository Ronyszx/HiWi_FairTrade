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

## Task 3 - Multi-Attribute Implementation and Smoke Test

Purpose: Extend the opt-in Adult experiment so gender and race both influence local fairness training, while MOBO conservatively optimizes the worse of their two absolute evaluation disparities.

Status: Implementation and short smoke verification completed on 2026-07-13. The first full run failed on 2026-07-14; after adding the documented Task 3-only GP-fit recovery, the full `15 epochs / 50 communication rounds / 10 MOBO rounds` run completed successfully later that day.

### Design

- Adult race is binarized as `White=1` and `Non-White=0`; gender remains `Male=1` and `Female=0`.
- Every client is checked for both gender groups, both binary race groups, and row alignment with its feature matrix before training.
- Local differentiable surrogate: `0.5 * gender_dp_loss + 0.5 * race_dp_loss` on continuous model outputs.
- Evaluation SPD uses rounded predictions: `PPR(Male) - PPR(Female)` and `PPR(White) - PPR(Non-White)`.
- MOBO objectives: `[-max(abs(gender_spd), abs(race_spd)), balanced_accuracy]`.
- qEHVI reference point: `[-1.01, -0.01]`, strictly dominated by feasible Task 3 outcomes.
- Each history array receives only the initial evaluation for a communication round. Candidate evaluations are used internally by MOBO and are not appended.
- Final intersectional metrics are computed once after training using the Task 2 evaluator.

The local surrogate and reported SPD are related fairness measures but are not numerically identical. The former must remain differentiable for gradient training; the latter is a black-box metric calculated from binary decisions.

### Smoke verification

Configuration: Adult, three clients, one client epoch, two communication rounds, one MOBO round, random distribution, seed 42, CPU, and `--task3_multi_attribute`.

| Metric | Final smoke value |
| --- | ---: |
| Balanced accuracy | 0.5000000000 |
| Signed gender SPD | 0.0000000000 |
| Signed race SPD | 0.0000000000 |
| Worst absolute SPD | 0.0000000000 |
| Intersectional max-min gap | 0.0000000000 |

The smoke model predicted the positive class for every test row. Its zero disparity therefore does not demonstrate a useful fair model: balanced accuracy remained `0.5`. This run verifies execution and artifacts only.

Start-of-round histories, each with exactly two finite values:

- Balanced accuracy: `[0.4954027832, 0.5000000000]`
- Signed gender SPD: `[-0.0054320406, 0.0000000000]`
- Signed race SPD: `[-0.0661871693, 0.0000000000]`
- Worst absolute SPD: `[0.0661871693, 0.0000000000]`

Verification: `21 passed` with `.venv/bin/python -m pytest -q`; all Python files compiled; the final summary and intersectional CSVs parsed; and the comparison chart was a valid non-empty PNG. The existing Task 1 balanced-accuracy and SPD arrays retained SHA-256 hashes `2d51b7374e5dce81cac53cf577e246a559f16be5de4d71d3680e4f67048cd376` and `3ad207641e555121103e96fc6c7a6667feab623872a0eac8bd556f7d329b74ef`.

The smoke run retained the baseline BoTorch float32, unscaled-input, unstandardized-output, and nonfatal SciPy optimization warnings. The sandbox also used a temporary writable Matplotlib cache because the default home cache was unavailable. The existing Sigmoid/loss mismatch, second constraint sigmoid, MOBO test-split use, and repeated candidate evaluation remain unchanged for a direct Task 1 comparison.

### Failed full-run diagnosis and GP-fit recovery

The seed-42 CPU full run completed communication rounds 1 through 12. In communication round 13, MOBO indices 0 through 8 completed; `fit_gpytorch_mll` then raised `ModelFittingError` at index 9, before candidate generation or observation append. Final evaluation and artifact generation were not reached.

All 271 printed objective evaluations before the crash were finite. The failed log contained 13 float32 GP warnings, 13 unscaled-input warnings, 13 unstandardized-output warnings, two candidate optimization retries, seven GP-fit optimization warnings, 41 Cholesky jitter warnings, and one terminal `ModelFittingError`.

The nested candidate/model loop evaluates each proposed candidate once per GP output model. Therefore, each MOBO candidate is evaluated twice and its identical input is appended twice. The failed run produced exactly 271 evaluations: 12 completed communication rounds with `1 + 2 * 10` evaluations each, followed by 19 evaluations in round 13. Without the duplicate candidate evaluations, 142 evaluations would have been expected at the same stopping point. Because `evaluate()` also updates the global model, duplicate inputs can receive different outputs. This likely contributed to the non-positive-definite covariance matrices, but the loop remains unchanged here to preserve baseline behavior and comparability.

Evidence was preserved without deletion:

- Failed log: `results/task3/adult_seed42_task3_failed_round13.log`
- Original stale smoke artifacts: `results/task3/smoke/pre_recovery_seed42_1_2_1/`
- Post-recovery normal smoke artifacts: `results/task3/smoke/post_recovery_normal_seed42_1_2_1/`
- Forced-recovery artifacts and logs: `results/task3/smoke/post_recovery_forced_failure_seed42_1_2_1/`

The narrow recovery catches only `ModelFittingError` in Task 3 mode. It records the communication round, one-based MOBO iteration, printed zero-based index, exception type, and retained alpha/learning rate. It then skips the remaining MOBO iterations in that communication round and continues with the next communication round using the previous valid hyperparameters. Baseline mode continues to call `fit_gpytorch_mll` directly without this catch.

Recovery verification:

- Focused test suite: `22 passed`.
- Normal `1/2/1` Task 3 smoke test: exit code 0, zero GP-fit failures, and four finite two-value history arrays.
- Forced first-fit Task 3 failure: the message reported round 1, iteration 1, `ModelFittingError`, retained `alpha=100.0`, and retained learning rate `0.001`; communication round 2 and final artifact generation completed.
- Forced baseline failure: exited nonzero with the original `ModelFittingError` and no Task 3 recovery message.
- A new `15/50/10` run was subsequently completed; its final results follow.

### Full Task 3 result

Configuration:

- Dataset: Adult
- Training mode: joint gender-and-race statistical parity
- Clients: 3
- Client epochs: 15
- Communication rounds: 50
- MOBO rounds per communication round: 10
- Distribution: random
- Seed: 42
- Device: CPU
- Evaluation split: existing Adult test split (`20%`, `random_state=42`)
- Run date: 2026-07-14

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
  --task3_multi_attribute
```

Final-model metrics:

| Metric | Value |
| --- | ---: |
| Balanced accuracy | 0.7683912709437309 |
| Signed gender SPD | 0.010463530725209502 |
| Absolute gender SPD | 0.010463530725209502 |
| Signed race SPD | 0.0003118728334245424 |
| Absolute race SPD | 0.0003118728334245424 |
| Worst absolute gender/race SPD | 0.010463530725209502 |
| Intersectional max-min gap | 0.09037460789468915 |

The four history arrays contain exactly 50 finite start-of-round evaluations. Candidate evaluations are excluded. Last-10 statistics use the population standard deviation (`ddof=0`):

| Start-of-round history | Last-10 mean | Last-10 standard deviation |
| --- | ---: | ---: |
| Balanced accuracy | 0.7683008193969727 | 0.0034044703096556303 |
| Signed gender SPD | 0.025938502707977353 | 0.01078095439098539 |
| Signed race SPD | 0.008913690476190483 | 0.006297371037753976 |
| Worst absolute SPD | 0.025938502707977353 | 0.01078095439098539 |

Task 1 versus Task 3:

| Metric | Task 1 | Task 3 | Absolute change or reduction | Relative reduction |
| --- | ---: | ---: | ---: | ---: |
| Balanced accuracy | 0.7690318211867191 | 0.7683912709437309 | 0.0006405502429881471 decrease | - |
| Absolute gender SPD | 0.02843397520847807 | 0.010463530725209502 | 0.017970444483268566 reduction | 63.20060544299263% |
| Absolute race SPD | 0.18502041142127348 | 0.0003118728334245424 | 0.18470853858784894 reduction | 99.83143868774866% |
| Worst absolute SPD | 0.18502041142127348 | 0.010463530725209502 | 0.17455688069606398 reduction | 94.34466141068887% |
| Intersectional max-min gap | 0.19205167199601722 | 0.09037460789468915 | 0.10167706410132807 reduction | 52.94255605514159% |

Absolute changes are differences in the metric's native 0-to-1 units. Relative reductions divide that difference by the Task 1 value and are percentages, not percentage-point changes. The full-precision Task 1 race and intersectional values were reconstructed from the saved integer group and positive-prediction counts rather than from their 10-decimal CSV display values.

Local client training used an equal average of gender and race demographic-parity surrogate losses, while MOBO optimized the worst absolute gender/race SPD. Balanced accuracy decreased only slightly, and both aggregate disparities improved strongly. The intersectional max-min gap also improved but remained non-zero at approximately `0.0904`, showing that small marginal gender and race gaps do not guarantee intersectional parity. One GP fitting failure used the documented fallback, retaining the previous valid hyperparameters instead of terminating training.

The recovery occurred at communication round 13, MOBO iteration 10 (zero-based index 9), and retained `alpha=1276.331787109375` and learning rate `0.008032766170799732`. The completed log contains 50 float32 GP warnings, 50 unscaled-input warnings, 50 unstandardized-output warnings, 18 GP-fit SciPy failures, seven candidate-generation SciPy retries, 76 Cholesky jitter warnings, and one recovered `ModelFittingError`. No terminal traceback occurred.

Artifacts and SHA-256 hashes:

- `results/task3/adult_seed42_multi_attribute_summary.csv`: `691123e3f2e263268c42513c0377418aca505710aab6b8b30fe9fe158079a8a4`
- `results/task3/adult_seed42_intersectional_metrics.csv`: `75dfae398e565391e942aa7e4002b3411025642ecc9c5258ee823da4b70645e9`
- `results/task3/adult_seed42_task1_vs_task3.png`: `d5c03861390e05dedd180a42eaa121b4bbeb8e6718d8c03580e854efd524f4df`
- `results/task3/adult_seed42_balanced_accuracy.npy`: `1246ead3b132740fb4c7f6c9dabde55f79c5e7f7da3bd6c9225e57ef7b347874`
- `results/task3/adult_seed42_gender_signed_spd.npy`: `c8b95ccabcb830e168304c7a367cc0b8800d7e931df5df563c346d02342ed64e`
- `results/task3/adult_seed42_race_signed_spd.npy`: `fc7e28dd40280fc61d4ec67001e082b08992a10bc95ffaf491d3900fc82aff4b`
- `results/task3/adult_seed42_worst_absolute_spd.npy`: `90ab022bf59752f30a214a986fa6038f52b51d4bf358b76aa5c88a47edaa9b1c`
- `results/task3/adult_seed42_task3_full.log`: `c5c64ce7b189a5afad38d4cb9f8046b8607e02f62797d3b650a096d53a3b5699`

The summary and intersectional CSVs serialize floating-point values to 10 decimal places, and they match the corresponding final-log values at that stored precision. The full-precision values above come from the final log. The comparison chart was visually checked against the Task 1 and Task 3 summary values.

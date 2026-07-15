# FairTrade: Achieving Pareto-Optimal Trade-offs Between Balanced Accuracy and Fairness in Federated Learning
As Federated Learning (FL) gains prominence in distributed machine learning applications, achieving fairness without compromising predictive performance becomes paramount. The data being gathered from distributed clients in an FL environment often leads to class imbalance. In such scenarios, balanced accuracy rather than accuracy is the true representation of model performance. However, most state-of-the-art fair FL methods report accuracy as the measure of performance,  which can lead to misguided interpretations of the model's effectiveness to mitigate discrimination. To the best of our knowledge, this work presents the first attempt towards achieving Pareto-optimal trade-offs between balanced accuracy and fairness in a federated environment (FairTrade). By utilizing multi-objective optimization, the framework negotiates the intricate balance between model's balanced accuracy and fairness. The framework's agnostic design adeptly accommodates both statistical and causal fairness notions, ensuring its adaptability across diverse FL contexts. We provide empirical evidence of our novel framework's efficacy through extensive experiments on five real-world datasets and comparisons with six competing baselines. The empirical results underscore the significant potential of our framework in improving the trade-off between fairness and balanced accuracy in FL applications.
## The datsets used in this project
* [Adult Census](https://archive.ics.uci.edu/dataset/2/adult)
* [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
* [Default](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
* [Law School](https://github.com/iosifidisvasileios/FABBOO/blob/master/Data/law_dataset.arff)
## Code
### Dataset Processing Scripts

The `datasets` directory contains all the datasets used in this project. Below is a description of python scripts written to process datasets:

- `load_data_utilities.py`: Utility script for loading and preprocessing all the datasets (Adult, Bank, Default, Law).

### Utility Scripts
- `utilities.py`: Utility script for computing evaluation metrics including 'statistical parity', average treatment effect (ATE), balanced accuracy, and accuracy.

### FairTrade main scripts
The following scripts constitute the complete methodology of FairTrade
- `Fairtrade-crypten.py`: Main script for the 'FairTrade' framework that orchestrates the fairness aware federated learning process on different datasets with secure multiparty protocol.
- `Fairtrade.py`: Main script for the 'FairTrade' framework that orchestrates the fairness aware federated learning process on different datasets without secure multiparty protocol.

- `constraint.py`: The script contains the implementation of fairness constraints for discrimination mitigation.

## HiWi Challenge Reproduction

The HiWi challenge baseline uses the Adult dataset with gender (`sex`) as the sensitive attribute. The commands below use CPU explicitly so that the reported run does not depend on CUDA or Apple MPS availability.

### Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True python -m pip install -r requirements.txt
```

The environment variable is required because CrypTen declares the deprecated `sklearn` package name. The project itself uses `scikit-learn`.

### Adult smoke test

```bash
python FairTrade.py \
  --dataset_name adult \
  --fairness_notion stat_parity \
  --num_clients 3 \
  --epochs 1 \
  --communication_rounds 2 \
  --mobo_optimization_rounds 1 \
  --distribution_type random \
  --seed 42 \
  --device cpu
```

### Adult default reproduction

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

### Task 1 result

| Metric | This reproduction | Reported Adult FairTrade result |
| --- | ---: | ---: |
| Balanced accuracy | 0.7690 | approximately 0.77 |
| Statistical parity difference | 0.0284 | approximately 0.001 |

Predictive performance closely matches the Adult R3C demographic-parity result in Table 1 of the [FairTrade paper](https://doi.org/10.1609/aaai.v38i10.28971), while the statistical parity difference is higher and is reported without adjustment. Detailed configuration, optimizer warnings, and the distinction between final-model and start-of-round metrics are recorded in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md). The tracked numeric summary is available in [results/task1/task1_summary.csv](results/task1/task1_summary.csv).

`--seed` controls Python, NumPy, and PyTorch randomness. The upstream dataset splits that explicitly use `random_state=42` are preserved unchanged. `--device auto` prefers CUDA, then MPS, then CPU; use `--device cpu` for the documented reproduction run. Exact bit-for-bit agreement across different hardware or library versions is not guaranteed.

### Task 2 intersectional fairness evaluation

Task 2 is an optional post-training evaluation and does not change local training, aggregation, fairness loss, MOBO, or the Task 1 result arrays.

Enable it on the same Adult baseline with:

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

The evaluator derives the sensitive-attribute mappings from the exact Adult CSV using the same `LabelEncoder` behavior as preprocessing. For the included dataset, `Female=0`, `Male=1`, and the race codes are `Amer-Indian-Eskimo=0`, `Asian-Pac-Islander=1`, `Black=2`, `Other=3`, and `White=4`.

Positive prediction rate is `P(prediction=1 | group)`. Signed gender SPD is `PPR(Male) - PPR(Female)`, and signed race SPD is `PPR(White) - PPR(Non-White)`. Absolute SPD reports the magnitude regardless of direction. For intersectional rows, signed SPD is `PPR(White Male) - PPR(group)`, with White Male documented as the reference. The max-min gap is the highest of the four intersectional PPRs minus the lowest.

The four reported groups are White Male, White Female, Non-White Male, and Non-White Female. Combining four race categories as Non-White follows the challenge specification but can hide differences within that aggregate. Intersectional evaluation is useful because small aggregate gender or race gaps do not guarantee that every combined subgroup has a similar positive prediction rate.

Outputs are written to:

- `results/task2/adult_seed42_intersectional_metrics.csv`
- `results/task2/adult_seed42_intersectional_spd.png`

The full seed-42 run produced:

| Metric / group | Value | Test samples |
| --- | ---: | ---: |
| Male positive prediction rate | 0.4734 | 4,360 |
| Female positive prediction rate | 0.4450 | 2,153 |
| Absolute gender SPD | 0.0284 | 6,513 |
| White positive prediction rate | 0.4908 | 5,568 |
| Non-White positive prediction rate | 0.3058 | 945 |
| Absolute race SPD | 0.1850 | 6,513 |
| Intersectional max-min gap | 0.1921 | 6,513 |

The aggregate gender gap is relatively small, but the race and intersectional gaps are much larger. White Male has the highest intersectional positive prediction rate (`0.4961`), while Non-White Female has the lowest (`0.3040`). This is the central Task 2 finding: a single aggregate gender metric can conceal substantially different outcomes across combined gender-race groups.

### Task 3 multi-attribute fairness optimization

Task 3 responds to the Task 2 finding by making gender and race part of training and optimization together. It is opt-in, Adult-specific, and leaves Task 1 and Task 2 unchanged when `--task3_multi_attribute` is omitted. Run the full seed-42 configuration with:

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

Each client uses the differentiable local surrogate `0.5 * gender_dp_loss + 0.5 * race_dp_loss`, so both sensitive attributes contribute gradients. MOBO evaluates rounded test predictions with `gender_spd = PPR(Male) - PPR(Female)` and `race_spd = PPR(White) - PPR(Non-White)`, then maximizes `-max(abs(gender_spd), abs(race_spd))` together with balanced accuracy. These are different quantities: the local surrogate works on continuous model outputs during training, while reported SPD works on rounded predictions. Task 3 uses qEHVI reference point `[-1.01, -0.01]`, which is below every feasible Task 3 objective in both dimensions.

The run writes only to `results/task3/`. The four `.npy` arrays store one initial evaluation per communication round, before that round's candidate loop; candidate evaluations are intentionally excluded. Final-model metrics are separate:

- `adult_seed42_balanced_accuracy.npy`
- `adult_seed42_gender_signed_spd.npy`
- `adult_seed42_race_signed_spd.npy`
- `adult_seed42_worst_absolute_spd.npy`
- `adult_seed42_multi_attribute_summary.csv`
- `adult_seed42_intersectional_metrics.csv`
- `adult_seed42_task1_vs_task3.png`

The completed seed-42 CPU run produced:

| Metric | Task 3 result |
| --- | ---: |
| Balanced accuracy | 0.7683912709437309 |
| Absolute gender SPD | 0.010463530725209502 |
| Absolute race SPD | 0.0003118728334245424 |
| Worst absolute SPD | 0.010463530725209502 |
| Intersectional max-min gap | 0.09037460789468915 |
| Recovered GP fitting failures | 1 |

For a direct Task 1 comparison, the implementation deliberately preserves the existing model Sigmoid, `BCEWithLogitsLoss`, the additional sigmoid inside `ConstraintLoss`, MOBO use of `X_test`, repeated candidate-loop behavior, and the final `0.6/0.4` weighted selection. Detailed results and limitations are recorded in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

The original repository's Bank commands are preserved below for reference.

## Running the FairTrade-crypten.py Script

To run the `FairTrade-crypten.py` script with the default settings, you can use the following command:

```bash
python FairTrade-crypten.py --fairness_notion 'stat_parity' --num_clients 3 --dataset_name 'bank' --epochs 15 --communication_rounds 50 --mobo_optimization_rounds 10 --distribution_type 'random'
```
## Running the FairTrade.py Script
To run the `FairTrade.py` script with the default settings, you can use the following command:

```bash
python FairTrade.py --fairness_notion 'stat_parity' --num_clients 3 --dataset_name 'bank' --epochs 15 --communication_rounds 50 --mobo_optimization_rounds 10 --distribution_type 'random'
```
## Prerequisites

Before running the script, ensure you have the following Python libraries installed:

- torch==2.0.1
- torchvision==0.15.2
- scikit-learn==1.2.2
- psmpy==0.3.16
- pandas==1.5.3
- gpytorch==1.10
- botorch==0.8.5
- crypten==0.4.1
- cvxopt==1.3.1
- cvxpy==1.3.2
- pytest==9.1.1
- matplotlib==3.11.0

## Citation Request
If you find this work useful in your research, please consider citing:
```bash
@inproceedings{badar2024fairtrade,
  title={FairTrade: Achieving Pareto-Optimal Trade-offs Between Balanced Accuracy and Fairness in Federated Learning},
  author={Badar, Maryam and Sikdar, Sandipan and Nejdl, Wolfgang and Fisichella, Marco},
  booktitle={Proceedings of the 38th Annual AAAI Conference on Artificial Intelligence},
  year={2024}
}
```

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

`--seed` controls Python, NumPy, and PyTorch randomness. The upstream dataset splits that explicitly use `random_state=42` are preserved unchanged. `--device auto` prefers CUDA, then MPS, then CPU; use `--device cpu` for the documented reproduction run. Exact bit-for-bit agreement across different hardware or library versions is not guaranteed.

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

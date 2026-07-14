"""Small helpers for Adult gender-and-race fairness optimization."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TASK3_REFERENCE_POINT = (-1.01, -0.01)
DATASET_SPLIT = "Adult test split (20%, random_state=42)"


def _integer_vector(values, name):
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite numeric values.")
    rounded = np.rint(array).astype(np.int64)
    if not np.allclose(array, rounded):
        raise ValueError(f"{name} must contain integer-valued entries.")
    return rounded


def binarize_adult_race(race_values, white_code=4):
    race = _integer_vector(race_values, "race")
    if not set(np.unique(race)).issubset({0, 1, 2, 3, 4}):
        raise ValueError("race contains a value outside the Adult encoding.")
    if white_code != 4:
        raise ValueError("The verified Adult White race code is 4.")
    return (race == white_code).astype(np.int64)


def validate_client_sensitive_groups(gender, binary_race, client_name):
    gender = _integer_vector(gender, f"{client_name}: gender")
    binary_race = _integer_vector(binary_race, f"{client_name}: binary race")
    if len(gender) != len(binary_race):
        raise ValueError(f"{client_name}: sensitive vector lengths do not match.")
    for name, values in (("gender", gender), ("binary race", binary_race)):
        groups = set(np.unique(values))
        if not groups.issubset({0, 1}):
            raise ValueError(f"{client_name}: {name} must contain only 0 and 1.")
        if groups != {0, 1}:
            raise ValueError(f"{client_name}: {name} must contain both groups.")
    return gender, binary_race


def combine_fairness_losses(gender_loss, race_loss):
    return 0.5 * gender_loss + 0.5 * race_loss


def compute_attribute_metrics(predictions, gender, binary_race):
    predictions = _integer_vector(predictions, "predictions")
    if not set(np.unique(predictions)).issubset({0, 1}):
        raise ValueError("predictions must contain only 0 and 1.")
    gender, binary_race = validate_client_sensitive_groups(
        gender, binary_race, "evaluation"
    )
    if len(predictions) != len(gender):
        raise ValueError("predictions, gender, and binary race lengths do not match.")

    masks = {
        "male": gender == 1,
        "female": gender == 0,
        "white": binary_race == 1,
        "non_white": binary_race == 0,
    }
    rates = {name: float(predictions[mask].mean()) for name, mask in masks.items()}
    gender_spd = rates["male"] - rates["female"]
    race_spd = rates["white"] - rates["non_white"]
    return {
        **{f"{name}_positive_rate": rate for name, rate in rates.items()},
        "gender_signed_spd": gender_spd,
        "gender_absolute_spd": abs(gender_spd),
        "race_signed_spd": race_spd,
        "race_absolute_spd": abs(race_spd),
        "worst_absolute_spd": max(abs(gender_spd), abs(race_spd)),
        "counts": {name: int(mask.sum()) for name, mask in masks.items()},
    }


def save_task3_summary(
    metrics, intersectional_metrics, output_path,
    task1_summary_path, task2_metrics_path, metadata,
):
    for path in (task1_summary_path, task2_metrics_path):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Required baseline artifact not found: {path}")
    task1 = pd.read_csv(task1_summary_path).set_index("metric")["value"]
    task2 = pd.read_csv(task2_metrics_path)

    def task2_value(metric, column):
        match = task2[task2["metric_name"] == metric]
        if len(match) != 1:
            raise ValueError(f"Expected one Task 2 baseline row for {metric}.")
        return float(match.iloc[0][column])

    gender_base = float(task1["final_spd"])
    race_base = task2_value("race_spd", "signed_spd")
    gap_base = task2_value("intersectional_max_min_gap", "metric_value")
    comparisons = [
        ("balanced_accuracy", float(task1["final_balanced_accuracy"]), "higher"),
        ("gender_signed_spd", gender_base, "context"),
        ("gender_absolute_spd", abs(gender_base), "lower"),
        ("race_signed_spd", race_base, "context"),
        ("race_absolute_spd", abs(race_base), "lower"),
        ("worst_absolute_spd", max(abs(gender_base), abs(race_base)), "lower"),
        ("intersectional_max_min_gap", gap_base, "lower"),
    ]
    common = {
        "dataset": "adult", "dataset_split": DATASET_SPLIT,
        "seed": metadata["seed"], "device": metadata["device"],
        "clients": metadata["clients"], "epochs": metadata["epochs"],
        "communication_rounds": metadata["communication_rounds"],
        "mobo_rounds": metadata["mobo_rounds"],
        "local_fairness_loss": "0.5 * gender_dp_loss + 0.5 * race_dp_loss",
        "mobo_fairness_objective": "-max(abs(gender_spd), abs(race_spd))",
    }
    total = metrics["counts"]["male"] + metrics["counts"]["female"]
    rows = [{
        "record_type": "final_metric", "metric": name,
        "task1_value": baseline,
        "task3_value": intersectional_metrics["intersectional_max_min_gap"]
        if name == "intersectional_max_min_gap" else metrics[name],
        "sample_count": total, "preferred_direction": direction,
        "notes": "Final model after all communication and MOBO evaluations.",
        **common,
    } for name, baseline, direction in comparisons]

    counts = {
        "Male": metrics["counts"]["male"], "Female": metrics["counts"]["female"],
        "White": metrics["counts"]["white"], "Non-White": metrics["counts"]["non_white"],
        **{name: intersectional_metrics["counts"][name] for name in (
            "White Male", "White Female", "Non-White Male", "Non-White Female"
        )},
    }
    rows.extend({
        "record_type": "group_count", "metric": name,
        "task1_value": None, "task3_value": count, "sample_count": count,
        "preferred_direction": "not_applicable",
        "notes": "Adult test-split group count.", **common,
    } for name, count in counts.items())
    rows.append({
        "record_type": "history_semantics", "metric": "round_history_arrays",
        "task1_value": None, "task3_value": None, "sample_count": None,
        "preferred_direction": "not_applicable",
        "notes": "One start-of-round evaluation per communication round; candidate evaluations are excluded.",
        **common,
    })
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.10f")
    return path


def save_task3_comparison_chart(summary_path, output_path):
    summary = pd.read_csv(summary_path)
    order = ["balanced_accuracy", "gender_absolute_spd", "race_absolute_spd",
             "worst_absolute_spd", "intersectional_max_min_gap"]
    final = summary[summary["record_type"] == "final_metric"].set_index("metric")
    if not set(order).issubset(final.index):
        raise ValueError("Task 3 summary is missing comparison metrics.")
    task1 = final.loc[order, "task1_value"].astype(float).to_numpy()
    task3 = final.loc[order, "task3_value"].astype(float).to_numpy()
    labels = ("Balanced\naccuracy", "|Gender SPD|", "|Race SPD|",
              "Worst\nattribute SPD", "Intersectional\nmax-min gap")
    x, width = np.arange(len(order)), 0.36

    figure, axis = plt.subplots(figsize=(11, 6))
    bars1 = axis.bar(x - width / 2, task1, width, label="Task 1", color="#6C8798")
    bars3 = axis.bar(x + width / 2, task3, width, label="Task 3", color="#D97745")
    axis.set(ylim=(0, 1), ylabel="Metric value",
             title="Adult: Task 1 baseline vs Task 3 multi-attribute optimization")
    axis.set_xticks(x, labels)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend()
    for bar in (*bars1, *bars3):
        value = bar.get_height()
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.012,
                  f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    figure.tight_layout()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path

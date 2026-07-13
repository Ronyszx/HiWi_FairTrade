"""Adult gender-race intersectional fairness evaluation."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REFERENCE_GROUP = "White Male"
GROUP_ORDER = (
    "White Male",
    "White Female",
    "Non-White Male",
    "Non-White Female",
)
DEFAULT_DATASET_SPLIT = "Adult test split (20%, random_state=42)"
EXPECTED_ENCODINGS = {
    "sex": {"Female": 0, "Male": 1},
    "race": {
        "Amer-Indian-Eskimo": 0,
        "Asian-Pac-Islander": 1,
        "Black": 2,
        "Other": 3,
        "White": 4,
    },
}


def validate_inputs(predictions, gender, race, adult_csv_path):
    """Validate aligned arrays and recover the exact Adult encodings."""

    arrays = []
    for name, values in (
        ("predictions", predictions),
        ("gender", gender),
        ("race", race),
    ):
        array = np.asarray(values)
        if array.ndim != 1 or array.size == 0:
            raise ValueError(f"{name} must be a non-empty one-dimensional array.")
        if not np.issubdtype(array.dtype, np.number) or not np.all(
            np.isfinite(array)
        ):
            raise ValueError(f"{name} must contain finite numeric values.")
        rounded = np.rint(array).astype(np.int64)
        if not np.allclose(array, rounded):
            raise ValueError(f"{name} must contain integer-valued entries.")
        arrays.append(rounded)

    predictions, gender, race = arrays
    if len({len(predictions), len(gender), len(race)}) != 1:
        raise ValueError("predictions, gender, and race must have equal lengths.")
    if not set(np.unique(predictions)).issubset({0, 1}):
        raise ValueError("predictions must contain only 0 and 1.")
    if not set(np.unique(gender)).issubset({0, 1}):
        raise ValueError("gender must contain only 0 and 1.")

    raw = pd.read_csv(adult_csv_path, usecols=["sex", "race"])
    encodings = {}
    for column in ("sex", "race"):
        encoder = LabelEncoder().fit(raw[column])
        encodings[column] = dict(
            zip(
                encoder.classes_.tolist(),
                encoder.transform(encoder.classes_).astype(int).tolist(),
            )
        )
    if encodings != EXPECTED_ENCODINGS:
        raise ValueError(f"Unexpected Adult sensitive encodings: {encodings}")
    if not set(np.unique(race)).issubset(encodings["race"].values()):
        raise ValueError("race contains a value outside the Adult encoding.")

    return predictions, gender, race, encodings


def positive_prediction_rate(predictions, mask, group_name):
    """Return P(prediction=1 | group) and the group count."""

    count = int(mask.sum())
    if count == 0:
        raise ValueError(f"Subgroup {group_name!r} is empty.")
    return float(predictions[mask].mean()), count


def compute_intersectional_metrics(predictions, gender, race, encodings):
    """Compute aggregate and intersectional positive-rate differences."""

    female = gender == encodings["sex"]["Female"]
    male = gender == encodings["sex"]["Male"]
    white = race == encodings["race"]["White"]
    non_white = ~white

    masks = {
        "Male": male,
        "Female": female,
        "White": white,
        "Non-White": non_white,
        "White Male": white & male,
        "White Female": white & female,
        "Non-White Male": non_white & male,
        "Non-White Female": non_white & female,
    }
    rates = {}
    counts = {}
    for group_name, mask in masks.items():
        rates[group_name], counts[group_name] = positive_prediction_rate(
            predictions, mask, group_name
        )

    gender_signed = rates["Male"] - rates["Female"]
    race_signed = rates["White"] - rates["Non-White"]
    reference_rate = rates[REFERENCE_GROUP]
    subgroup_signed = {
        group: reference_rate - rates[group] for group in GROUP_ORDER
    }

    subgroup_rates = {group: rates[group] for group in GROUP_ORDER}
    maximum_rate = max(subgroup_rates.values())
    minimum_rate = min(subgroup_rates.values())
    maximum_group = next(
        group for group in GROUP_ORDER if subgroup_rates[group] == maximum_rate
    )
    minimum_group = next(
        group
        for group in reversed(GROUP_ORDER)
        if subgroup_rates[group] == minimum_rate
    )

    return {
        "positive_rates": rates,
        "counts": counts,
        "gender_signed_spd": float(gender_signed),
        "gender_absolute_spd": abs(float(gender_signed)),
        "race_signed_spd": float(race_signed),
        "race_absolute_spd": abs(float(race_signed)),
        "subgroup_signed_spd": subgroup_signed,
        "subgroup_absolute_spd": {
            group: abs(value) for group, value in subgroup_signed.items()
        },
        "intersectional_max_min_gap": float(maximum_rate - minimum_rate),
        "maximum_rate_group": maximum_group,
        "minimum_rate_group": minimum_group,
    }


def save_metrics_csv(
    metrics,
    output_path,
    seed,
    dataset_split=DEFAULT_DATASET_SPLIT,
):
    """Write the requested Task 2 rows using the established CSV schema."""

    rates = metrics["positive_rates"]
    counts = metrics["counts"]
    rows = []
    for group in ("Male", "Female", "White", "Non-White"):
        rows.append([
            "positive_prediction_rate", group, rates[group], rates[group],
            None, None, counts[group], "", seed, dataset_split, "",
        ])
    rows.extend(
        [
            [
                "gender_spd", "Male vs Female", metrics["gender_signed_spd"], None,
                metrics["gender_signed_spd"],
                metrics["gender_absolute_spd"],
                counts["Male"] + counts["Female"], "Male", seed, dataset_split,
                "Signed SPD = PPR(Male) - PPR(Female).",
            ],
            [
                "race_spd", "White vs Non-White", metrics["race_signed_spd"], None,
                metrics["race_signed_spd"],
                metrics["race_absolute_spd"],
                counts["White"] + counts["Non-White"], "White", seed, dataset_split,
                "Signed SPD = PPR(White) - PPR(Non-White).",
            ],
        ]
    )
    for group in GROUP_ORDER:
        rows.append([
            "intersectional_positive_prediction_rate", group,
            rates[group], rates[group],
            metrics["subgroup_signed_spd"][group],
            metrics["subgroup_absolute_spd"][group],
            counts[group], REFERENCE_GROUP, seed, dataset_split,
            "Subgroup signed SPD = PPR(White Male) - PPR(group).",
        ])
    rows.append([
            "intersectional_max_min_gap",
            f"{metrics['maximum_rate_group']} vs {metrics['minimum_rate_group']}",
            metrics["intersectional_max_min_gap"], None, None, None,
            sum(counts[group] for group in GROUP_ORDER),
            "", seed, dataset_split,
            "Maximum intersectional subgroup PPR minus minimum subgroup PPR.",
    ])

    columns = [
        "metric_name", "group", "metric_value", "positive_prediction_rate",
        "signed_spd", "absolute_spd", "sample_count", "reference_group",
        "seed", "dataset_split", "notes",
    ]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(
        path, index=False, float_format="%.10f"
    )
    return path


def save_comparison_chart(metrics, output_path):
    """Plot absolute gender SPD, race SPD, and the max-min gap."""

    counts = metrics["counts"]
    labels = ("Gender SPD", "Race SPD", "Intersectional\nmax-min gap")
    values = (
        metrics["gender_absolute_spd"],
        metrics["race_absolute_spd"],
        metrics["intersectional_max_min_gap"],
    )
    count_labels = (
        f"Male n={counts['Male']}\nFemale n={counts['Female']}",
        f"White n={counts['White']}\nNon-White n={counts['Non-White']}",
        f"{metrics['maximum_rate_group']} n={counts[metrics['maximum_rate_group']]}\n"
        f"{metrics['minimum_rate_group']} n={counts[metrics['minimum_rate_group']]}",
    )

    figure, axis = plt.subplots(figsize=(9, 6))
    bars = axis.bar(
        labels,
        values,
        color=("#157A78", "#D9A441", "#E46D4C"),
        width=0.58,
    )
    axis.set_ylim(0, 1)
    axis.set_ylabel("Absolute positive-rate difference")
    axis.set_title("Adult Task 2: aggregate and intersectional fairness gaps")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    for bar, value, count_label in zip(bars, values, count_labels):
        high_bar = value > 0.85
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value - 0.025 if high_bar else value + 0.025,
            f"{value:.4f}\n{count_label}",
            ha="center",
            va="top" if high_bar else "bottom",
            fontsize=9,
        )
    figure.text(
        0.5,
        0.015,
        "Absolute SPD is shown for gender/race; intersectional gap is max PPR - min PPR.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path

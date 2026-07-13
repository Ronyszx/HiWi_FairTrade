from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from intersectional_fairness import (
    compute_intersectional_metrics,
    save_comparison_chart,
    save_metrics_csv,
    validate_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
ADULT_CSV = ROOT / "datasets" / "adult.csv"
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


def four_group_metrics(predictions=(1, 1, 1, 0)):
    gender = np.array([1, 1, 0, 0])
    race = np.array([4, 2, 4, 2])
    values = validate_inputs(predictions, gender, race, ADULT_CSV)
    return compute_intersectional_metrics(*values)


def test_known_gender_spd():
    metrics = four_group_metrics()

    assert metrics["positive_rates"]["Male"] == pytest.approx(1.0)
    assert metrics["positive_rates"]["Female"] == pytest.approx(0.5)
    assert metrics["gender_signed_spd"] == pytest.approx(0.5)
    assert metrics["gender_absolute_spd"] == pytest.approx(0.5)


def test_known_race_spd():
    metrics = four_group_metrics()

    assert metrics["positive_rates"]["White"] == pytest.approx(1.0)
    assert metrics["positive_rates"]["Non-White"] == pytest.approx(0.5)
    assert metrics["race_signed_spd"] == pytest.approx(0.5)
    assert metrics["race_absolute_spd"] == pytest.approx(0.5)


def test_all_four_intersectional_groups():
    metrics = four_group_metrics()

    assert metrics["counts"] == {
        "Male": 2,
        "Female": 2,
        "White": 2,
        "Non-White": 2,
        "White Male": 1,
        "White Female": 1,
        "Non-White Male": 1,
        "Non-White Female": 1,
    }
    assert [metrics["positive_rates"][group] for group in (
        "White Male", "White Female", "Non-White Male", "Non-White Female"
    )] == [1.0, 1.0, 1.0, 0.0]


def test_signed_and_absolute_disparity():
    metrics = four_group_metrics(predictions=(0, 0, 1, 1))

    assert metrics["gender_signed_spd"] == pytest.approx(-1.0)
    assert metrics["gender_absolute_spd"] == pytest.approx(1.0)
    assert metrics["subgroup_signed_spd"]["White Female"] == pytest.approx(-1.0)
    assert metrics["subgroup_absolute_spd"]["White Female"] == pytest.approx(1.0)


def test_intersectional_max_min_gap():
    metrics = four_group_metrics()

    assert metrics["intersectional_max_min_gap"] == pytest.approx(1.0)
    assert metrics["maximum_rate_group"] == "White Male"
    assert metrics["minimum_rate_group"] == "Non-White Female"


def test_empty_subgroup_fails_clearly():
    values = validate_inputs(
        predictions=[1, 1, 0],
        gender=[1, 0, 1],
        race=[4, 4, 2],
        adult_csv_path=ADULT_CSV,
    )
    with pytest.raises(ValueError, match="Non-White Female.*empty"):
        compute_intersectional_metrics(*values)


def test_mismatched_input_lengths():
    with pytest.raises(ValueError, match="equal lengths"):
        validate_inputs([0, 1], [0], [4, 2], ADULT_CSV)


def test_adult_counts_csv_schema_and_chart(tmp_path):
    raw = pd.read_csv(ADULT_CSV, usecols=["sex", "race"])
    encoded = pd.DataFrame({
        "gender": raw["sex"].map(EXPECTED_ENCODINGS["sex"]),
        "race": raw["race"].map(EXPECTED_ENCODINGS["race"]),
    })
    _, test = train_test_split(encoded, test_size=0.2, random_state=42)
    values = validate_inputs(
        np.zeros(len(test)), test["gender"], test["race"], ADULT_CSV
    )
    metrics = compute_intersectional_metrics(*values)

    assert values[3] == EXPECTED_ENCODINGS
    assert {group: metrics["counts"][group] for group in (
        "White Male", "White Female", "Non-White Male", "Non-White Female"
    )} == {
        "White Male": 3836,
        "White Female": 1732,
        "Non-White Male": 524,
        "Non-White Female": 421,
    }
    assert sum(metrics["counts"][group] for group in (
        "White Male", "White Female", "Non-White Male", "Non-White Female"
    )) == 6513

    csv_path = save_metrics_csv(metrics, tmp_path / "metrics.csv", seed=42)
    chart_path = save_comparison_chart(metrics, tmp_path / "chart.png")
    saved = pd.read_csv(csv_path)
    assert saved.columns.tolist() == [
        "metric_name", "group", "metric_value", "positive_prediction_rate",
        "signed_spd", "absolute_spd", "sample_count", "reference_group",
        "seed", "dataset_split", "notes",
    ]
    assert chart_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert chart_path.stat().st_size > 0

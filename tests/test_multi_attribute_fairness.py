from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from multi_attribute_fairness import (
    TASK3_REFERENCE_POINT,
    binarize_adult_race,
    combine_fairness_losses,
    compute_attribute_metrics,
    save_task3_comparison_chart,
    save_task3_summary,
    validate_client_sensitive_groups,
)


ROOT = Path(__file__).resolve().parents[1]
TASK1_SUMMARY = ROOT / "results" / "task1" / "task1_summary.csv"
TASK2_METRICS = ROOT / "results" / "task2" / "adult_seed42_intersectional_metrics.csv"


def known_metrics(predictions=(1, 1, 1, 0)):
    gender = np.array([1, 1, 0, 0])
    binary_race = np.array([1, 0, 1, 0])
    return compute_attribute_metrics(predictions, gender, binary_race)


def test_adult_race_binarization():
    assert binarize_adult_race([0, 1, 2, 3, 4]).tolist() == [0, 0, 0, 0, 1]


def test_known_gender_spd():
    metrics = known_metrics()
    assert metrics["male_positive_rate"] == pytest.approx(1.0)
    assert metrics["female_positive_rate"] == pytest.approx(0.5)
    assert metrics["gender_signed_spd"] == pytest.approx(0.5)


def test_known_race_spd():
    metrics = known_metrics()
    assert metrics["white_positive_rate"] == pytest.approx(1.0)
    assert metrics["non_white_positive_rate"] == pytest.approx(0.5)
    assert metrics["race_signed_spd"] == pytest.approx(0.5)


def test_worst_absolute_spd_selection():
    metrics = known_metrics(predictions=(1, 0, 1, 0))
    assert metrics["gender_absolute_spd"] == pytest.approx(0.0)
    assert metrics["race_absolute_spd"] == pytest.approx(1.0)
    assert metrics["worst_absolute_spd"] == pytest.approx(1.0)


def test_input_alignment_validation():
    with pytest.raises(ValueError, match="lengths do not match"):
        compute_attribute_metrics([1, 0], [1, 0, 1], [1, 0, 1])


def test_missing_gender_group_fails():
    with pytest.raises(ValueError, match="gender must contain both groups"):
        validate_client_sensitive_groups([1, 1], [0, 1], "client_1")


def test_missing_race_group_fails():
    with pytest.raises(ValueError, match="binary race must contain both groups"):
        validate_client_sensitive_groups([0, 1], [1, 1], "client_1")


def test_invalid_race_code_fails():
    with pytest.raises(ValueError, match="outside the Adult encoding"):
        binarize_adult_race([0, 5])


def test_combined_loss_uses_both_gradients():
    gender_loss = torch.tensor(2.0, requires_grad=True)
    race_loss = torch.tensor(4.0, requires_grad=True)
    combined = combine_fairness_losses(gender_loss, race_loss)
    combined.backward()

    assert combined.item() == pytest.approx(3.0)
    assert gender_loss.grad.item() == pytest.approx(0.5)
    assert race_loss.grad.item() == pytest.approx(0.5)


def test_task3_reference_is_strictly_worse():
    reference = np.array(TASK3_REFERENCE_POINT)
    feasible = np.array([[-1.0, 0.0], [-0.2, 0.77], [0.0, 1.0]])
    assert np.all(reference < feasible)


def task3_output(tmp_path):
    gender = np.array([1] * 3836 + [0] * 1732 + [1] * 524 + [0] * 421)
    binary_race = np.array([1] * 5568 + [0] * 945)
    metrics = compute_attribute_metrics(np.zeros(6513), gender, binary_race)
    metrics["balanced_accuracy"] = 0.75
    intersectional = {
        "intersectional_max_min_gap": 0.0,
        "counts": {
            "Male": 4360,
            "Female": 2153,
            "White": 5568,
            "Non-White": 945,
            "White Male": 3836,
            "White Female": 1732,
            "Non-White Male": 524,
            "Non-White Female": 421,
        },
    }
    task3_directory = tmp_path / "results" / "task3"
    summary = save_task3_summary(
        metrics,
        intersectional,
        task3_directory / "summary.csv",
        TASK1_SUMMARY,
        TASK2_METRICS,
        {
            "seed": 42,
            "device": "cpu",
            "clients": 3,
            "epochs": 1,
            "communication_rounds": 2,
            "mobo_rounds": 1,
        },
    )
    chart = save_task3_comparison_chart(summary, task3_directory / "comparison.png")
    return summary, chart


def test_task3_outputs_stay_in_task3_directory(tmp_path):
    summary, chart = task3_output(tmp_path)
    expected_parent = tmp_path / "results" / "task3"
    assert summary.parent == expected_parent
    assert chart.parent == expected_parent
    assert sorted(path.name for path in (tmp_path / "results").iterdir()) == ["task3"]


def test_summary_schema(tmp_path):
    summary, _ = task3_output(tmp_path)
    saved = pd.read_csv(summary)
    assert saved.columns.tolist() == [
        "record_type", "metric", "task1_value", "task3_value",
        "sample_count", "preferred_direction", "notes", "dataset",
        "dataset_split", "seed", "device", "clients", "epochs",
        "communication_rounds", "mobo_rounds", "local_fairness_loss",
        "mobo_fairness_objective",
    ]
    assert "history_semantics" in saved["record_type"].values
    counts = saved[saved["record_type"] == "group_count"]
    assert len(counts) == 8
    assert counts["sample_count"].sum() == 19539


def test_headless_chart_creation(tmp_path):
    _, chart = task3_output(tmp_path)
    assert chart.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert chart.stat().st_size > 0

from app.training.acceptance import check_acceptance


def _metrics(accuracy_lift=0.05, mae_improvement=0.0001):
    return {
        "aggregate": {
            "n_oos_rows": 500,
            "mean_accuracy": 0.6,
            "mean_accuracy_lift_vs_baseline": accuracy_lift,
            "mean_mae_improvement_vs_baseline": mae_improvement,
        }
    }


def test_acceptance_requires_improvement_over_naive_baselines():
    result = check_acceptance(_metrics())
    assert result["passed"]

    result = check_acceptance(_metrics(accuracy_lift=0.0))
    assert not result["passed"]
    assert any("persistence baseline" in reason for reason in result["reasons"])

    result = check_acceptance(_metrics(mae_improvement=0.0))
    assert not result["passed"]
    assert any("zero-return baseline" in reason for reason in result["reasons"])


def test_acceptance_rejects_missing_baseline_comparisons():
    result = check_acceptance(
        {"aggregate": {"n_oos_rows": 500, "mean_accuracy": 0.6}}
    )
    assert not result["passed"]
    assert "baseline direction comparison unavailable" in result["reasons"]
    assert "baseline price comparison unavailable" in result["reasons"]

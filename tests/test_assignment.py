"""
Automated tests for Assignment 1 -- Brain Tumor Classification.
Run locally with:  pytest tests/ -v
"""

import pytest
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Exercise 3 -- Metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    """Tests for compute_metrics(tp, fp, fn)."""

    def test_glioma_example_from_exercise(self):
        """Values taken directly from Exercise 3: TP=42, FP=8, FN=18."""
        from assignment import compute_metrics
        precision, recall, f1 = compute_metrics(42, 8, 18)
        assert abs(precision - 0.84) < 1e-3, f"Expected precision ~0.84, got {precision}"
        assert abs(recall - 0.70) < 1e-3, f"Expected recall ~0.70, got {recall}"
        assert abs(f1 - 0.7636) < 1e-3, f"Expected F1 ~0.7636, got {f1}"

    def test_perfect_classifier(self):
        """A perfect classifier: FP=0, FN=0 -> P=R=F1=1.0."""
        from assignment import compute_metrics
        precision, recall, f1 = compute_metrics(tp=100, fp=0, fn=0)
        assert precision == pytest.approx(1.0)
        assert recall == pytest.approx(1.0)
        assert f1 == pytest.approx(1.0)

    def test_returns_three_floats(self):
        from assignment import compute_metrics
        result = compute_metrics(10, 5, 5)
        assert len(result) == 3, "Expected a tuple of 3 values"
        assert all(isinstance(v, float) for v in result), "All values should be floats"

    def test_recall_reflects_missed_cases(self):
        """High FN should produce low recall."""
        from assignment import compute_metrics
        _, recall, _ = compute_metrics(tp=10, fp=0, fn=90)
        assert recall < 0.15, "Recall should be low when FN is large"


# ---------------------------------------------------------------------------
# Exercise 5 -- Class weights
# ---------------------------------------------------------------------------

class TestComputeClassWeights:
    """Tests for compute_class_weights(class_counts)."""

    COUNTS = {
        "glioma": 120,
        "meningioma": 240,
        "pituitary": 300,
        "no_tumor": 840,
    }

    def test_exercise_5_values(self):
        """Expected weights from Exercise 5 solution."""
        from assignment import compute_class_weights
        weights = compute_class_weights(self.COUNTS)
        assert abs(weights["glioma"] - 3.125) < 1e-2
        assert abs(weights["meningioma"] - 1.5625) < 1e-2
        assert abs(weights["pituitary"] - 1.25) < 1e-2
        assert abs(weights["no_tumor"] - 0.4464) < 1e-2

    def test_rarest_class_has_highest_weight(self):
        """Glioma is rarest, so it must have the highest weight."""
        from assignment import compute_class_weights
        weights = compute_class_weights(self.COUNTS)
        assert weights["glioma"] == max(weights.values()), \
            "The rarest class should have the highest weight"

    def test_balanced_classes_give_weight_one(self):
        """When all classes have equal counts, all weights should be 1.0."""
        from assignment import compute_class_weights
        balanced = {"a": 100, "b": 100, "c": 100, "d": 100}
        weights = compute_class_weights(balanced)
        for cls, w in weights.items():
            assert abs(w - 1.0) < 1e-6, f"Class {cls} expected weight 1.0, got {w}"

    def test_returns_dict_with_all_keys(self):
        from assignment import compute_class_weights
        weights = compute_class_weights(self.COUNTS)
        assert set(weights.keys()) == set(self.COUNTS.keys())


# ---------------------------------------------------------------------------
# Exercise 2 -- SimpleCNN architecture
# ---------------------------------------------------------------------------

class TestSimpleCNN:
    """Tests for the SimpleCNN model."""

    def test_output_shape_4_classes(self):
        from assignment import SimpleCNN
        model = SimpleCNN(num_classes=4)
        model.eval()
        x = torch.zeros(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 4), f"Expected shape (2, 4), got {out.shape}"

    def test_output_shape_custom_classes(self):
        from assignment import SimpleCNN
        model = SimpleCNN(num_classes=10)
        model.eval()
        x = torch.zeros(4, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 10), f"Expected shape (4, 10), got {out.shape}"

    def test_model_is_nn_module(self):
        import torch.nn as nn
        from assignment import SimpleCNN
        model = SimpleCNN()
        assert isinstance(model, nn.Module)

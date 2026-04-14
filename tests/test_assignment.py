import json
import pathlib
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Notebook — written answers
# ---------------------------------------------------------------------------

class TestNotebookAnswers:

    PLACEHOLDER = "> *Your answer.*"

    def _markdown_cells(self):
        nb = json.loads(pathlib.Path("notebook.ipynb").read_text())
        return [
            "".join(cell["source"])
            for cell in nb["cells"]
            if cell["cell_type"] == "markdown"
        ]

    def test_no_placeholders_remaining(self):
        unanswered = [
            cell[:80] for cell in self._markdown_cells()
            if self.PLACEHOLDER in cell
        ]
        assert not unanswered, (
            f"{len(unanswered)} written question(s) still contain the placeholder "
            f"'{self.PLACEHOLDER}'. Fill in all answers in notebook.ipynb."
        )

    def test_notebook_was_executed(self):
        nb = json.loads(pathlib.Path("notebook.ipynb").read_text())
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        executed = [c for c in code_cells if c.get("execution_count") is not None]
        assert len(executed) > 0, "notebook.ipynb has no executed cells — run it before submitting."


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:

    def test_known_values(self):
        from assignment import compute_metrics
        p, r, f1 = compute_metrics(42, 8, 18)
        assert abs(p - 0.84) < 1e-3
        assert abs(r - 0.70) < 1e-3
        assert abs(f1 - 0.7636) < 1e-3

    def test_perfect_predictions(self):
        from assignment import compute_metrics
        p, r, f1 = compute_metrics(tp=50, fp=0, fn=0)
        assert p == pytest.approx(1.0)
        assert r == pytest.approx(1.0)
        assert f1 == pytest.approx(1.0)

    def test_return_type(self):
        from assignment import compute_metrics
        result = compute_metrics(10, 5, 5)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)


# ---------------------------------------------------------------------------
# compute_class_weights
# ---------------------------------------------------------------------------

class TestComputeClassWeights:

    COUNTS = {"a": 100, "b": 200, "c": 300, "d": 400}

    def test_rarest_class_has_highest_weight(self):
        from assignment import compute_class_weights
        weights = compute_class_weights(self.COUNTS)
        assert max(weights, key=weights.get) == "a"

    def test_balanced_gives_unit_weights(self):
        from assignment import compute_class_weights
        balanced = {"a": 100, "b": 100, "c": 100}
        weights = compute_class_weights(balanced)
        assert all(abs(w - 1.0) < 1e-6 for w in weights.values())

    def test_returns_all_keys(self):
        from assignment import compute_class_weights
        weights = compute_class_weights(self.COUNTS)
        assert set(weights.keys()) == set(self.COUNTS.keys())


# ---------------------------------------------------------------------------
# SimpleCNN
# ---------------------------------------------------------------------------

class TestSimpleCNN:

    def test_output_shape(self):
        from assignment import SimpleCNN
        model = SimpleCNN(num_classes=4).eval()
        with torch.no_grad():
            out = model(torch.zeros(2, 3, 224, 224))
        assert out.shape == (2, 4)

    def test_num_classes_respected(self):
        from assignment import SimpleCNN
        model = SimpleCNN(num_classes=7).eval()
        with torch.no_grad():
            out = model(torch.zeros(1, 3, 64, 64))
        assert out.shape == (1, 7)

    def test_is_nn_module(self):
        from assignment import SimpleCNN
        assert isinstance(SimpleCNN(), nn.Module)

import unittest

from oepp.training.selection import is_better_direct_checkpoint


class CheckpointSelectionTests(unittest.TestCase):
    def test_first_validation_result_is_selected(self):
        self.assertTrue(
            is_better_direct_checkpoint({"sr": 0.0, "acc": 0.0, "mse": 1.0}, None, 1, None)
        )

    def test_success_rate_has_priority(self):
        best = {"sr": 0.1, "acc": 0.9, "mse": 0.01}
        candidate = {"sr": 0.2, "acc": 0.0, "mse": 10.0}
        self.assertTrue(is_better_direct_checkpoint(candidate, best, 2, 1))

    def test_accuracy_breaks_success_rate_tie(self):
        best = {"sr": 0.1, "acc": 0.4, "mse": 0.01}
        candidate = {"sr": 0.1, "acc": 0.5, "mse": 10.0}
        self.assertTrue(is_better_direct_checkpoint(candidate, best, 2, 1))

    def test_lower_mse_breaks_metric_tie(self):
        best = {"sr": 0.1, "acc": 0.5, "mse": 0.2}
        candidate = {"sr": 0.1, "acc": 0.5, "mse": 0.1}
        self.assertTrue(is_better_direct_checkpoint(candidate, best, 2, 1))

    def test_earlier_epoch_wins_exact_tie(self):
        metrics = {"sr": 0.1, "acc": 0.5, "mse": 0.1}
        self.assertFalse(is_better_direct_checkpoint(metrics, metrics, 2, 1))


if __name__ == "__main__":
    unittest.main()

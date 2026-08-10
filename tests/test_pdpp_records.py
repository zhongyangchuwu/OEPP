import json
import tempfile
import unittest
from pathlib import Path

from oepp.training.pdpp import (
    PDPP_SELECTION_FORMAT,
    PDPP_VALIDATION_HISTORY_FORMAT,
    load_pdpp_validation_history,
    write_pdpp_training_records,
)


class PDPPRecordTests(unittest.TestCase):
    def test_training_records_persist_selection_without_test_use(self) -> None:
        history = [
            {
                "epoch": 1,
                "train": {"objective": 1.2, "ce": 1.6, "mse": 2.4},
                "validation": {"sr": 0.2, "acc": 0.4, "miou1": 0.5},
            }
        ]
        best_metrics = {"sr": 0.2, "acc": 0.4, "miou1": 0.5}

        with tempfile.TemporaryDirectory() as directory:
            write_pdpp_training_records(directory, history, 1, best_metrics)

            history_payload = json.loads(
                (Path(directory) / "validation_history.json").read_text(encoding="utf-8")
            )
            selection_payload = json.loads(
                (Path(directory) / "selection.json").read_text(encoding="utf-8")
            )

            self.assertEqual(history_payload["format"], PDPP_VALIDATION_HISTORY_FORMAT)
            self.assertEqual(history_payload["epochs"], history)
            self.assertEqual(selection_payload["format"], PDPP_SELECTION_FORMAT)
            self.assertEqual(selection_payload["selected_epoch"], 1)
            self.assertEqual(selection_payload["validation_metrics"], best_metrics)
            self.assertFalse(selection_payload["test_sets_used_for_selection"])
            self.assertEqual(
                load_pdpp_validation_history(
                    Path(directory) / "validation_history.json", checkpoint_epoch=1
                ),
                history,
            )

    def test_history_rejects_records_ahead_of_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation_history.json"
            path.write_text(
                json.dumps(
                    {
                        "format": PDPP_VALIDATION_HISTORY_FORMAT,
                        "epochs": [{"epoch": 2, "train": {}, "validation": {}}],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "ahead of checkpoint epoch"):
                load_pdpp_validation_history(path, checkpoint_epoch=1)


if __name__ == "__main__":
    unittest.main()

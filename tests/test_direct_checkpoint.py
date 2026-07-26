import pickle
import tempfile
import unittest
from pathlib import Path

try:
    import torch

    from embedding_support import (
        DIRECT_CHECKPOINT_FORMAT,
        load_direct_checkpoint,
        save_direct_checkpoint,
    )

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "requires PyTorch")
class DirectCheckpointTests(unittest.TestCase):
    def _save_checkpoint(self, path: Path, torch_version: object) -> None:
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters())
        save_direct_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            epoch=1,
            config={"feature": "videoclip"},
            validation_metrics={"sr": 0.0, "acc": 0.0, "mse": 1.0},
            provenance={"torch_version": torch_version},
        )

    def test_loads_current_checkpoint_with_weights_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_path = Path(temporary) / "best.pt"
            self._save_checkpoint(checkpoint_path, str(torch.__version__))

            loaded = load_direct_checkpoint(checkpoint_path, torch.device("cpu"))

        self.assertEqual(loaded["format"], DIRECT_CHECKPOINT_FORMAT)
        self.assertEqual(loaded["epoch"], 1)

    def test_legacy_torch_version_requires_explicit_trust(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_path = Path(temporary) / "best.pt"
            self._save_checkpoint(checkpoint_path, torch.__version__)

            with self.assertRaises(pickle.UnpicklingError):
                load_direct_checkpoint(checkpoint_path, torch.device("cpu"))
            loaded = load_direct_checkpoint(
                checkpoint_path, torch.device("cpu"), trust_checkpoint=True
            )

        self.assertEqual(str(loaded["provenance"]["torch_version"]), str(torch.__version__))


if __name__ == "__main__":
    unittest.main()

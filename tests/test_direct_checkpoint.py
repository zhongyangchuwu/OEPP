import pickle
import tempfile
import unittest
from pathlib import Path

try:
    import torch

    from oepp.training.direct import _restore_direct_training_state
    from oepp.training.support import (
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

    def test_resume_restores_last_and_prior_best_selection(self) -> None:
        config = {"feature": "videoclip"}
        provenance = {"split_source_hashes": {"bundle": "hash"}}
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.Adam(model.parameters())
            loss = model(torch.ones(1, 2)).sum()
            loss.backward()
            optimizer.step()
            save_direct_checkpoint(
                run_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                epoch=4,
                config=config,
                validation_metrics={"sr": 0.4, "acc": 0.4, "mse": 0.4},
                provenance=provenance,
            )
            save_direct_checkpoint(
                run_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=3,
                config=config,
                validation_metrics={"sr": 0.6, "acc": 0.5, "mse": 0.3},
                provenance=provenance,
            )
            restored_model = torch.nn.Linear(2, 2)
            restored_optimizer = torch.optim.Adam(restored_model.parameters())
            start_epoch, best_metrics, best_epoch, restored_provenance = (
                _restore_direct_training_state(
                    run_dir=run_dir,
                    model=restored_model,
                    optimizer=restored_optimizer,
                    config=config,
                    device=torch.device("cpu"),
                    kepp_graph_provenance=None,
                )
            )

        self.assertEqual(start_epoch, 4)
        self.assertEqual(best_epoch, 3)
        self.assertEqual(best_metrics, {"sr": 0.6, "acc": 0.5, "mse": 0.3})
        self.assertEqual(restored_provenance, provenance)


if __name__ == "__main__":
    unittest.main()

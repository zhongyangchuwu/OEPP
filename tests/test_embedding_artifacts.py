import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import torch
    from embedding_artifacts import export_split
    EMBEDDING_DEPS_AVAILABLE = True
except ImportError:
    EMBEDDING_DEPS_AVAILABLE = False


class _Dataset:
    def __len__(self):
        return 1

    def metadata_at(self, index):
        self_index = index
        if self_index != 0:
            raise IndexError(index)
        return {
            'sample_id': 'base_video0_start0_T2',
            'split': 'base',
            'dataset': 'COIN',
            'task_name': 'Task',
            'task_id': 1,
            'task_id_old': 1,
            'vid': 'video0',
            'source_video_index': 0,
            'start_step': 0,
            'end_step': 1,
            'is_padded': False,
            'pad_count': 0,
            'actions': ['action a', 'action b'],
        }


@unittest.skipUnless(EMBEDDING_DEPS_AVAILABLE, 'requires the server NumPy and PyTorch runtime')
class EmbeddingArtifactTests(unittest.TestCase):
    def test_export_preserves_embeddings_metrics_and_metadata(self):
        ground_truth = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        labels = torch.tensor([[0, 1]])
        batch = [None, None, None, None, None, ground_truth, None, labels]
        candidates = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        with tempfile.TemporaryDirectory() as temporary:
            exported = export_split(
                dataset=_Dataset(),
                loader=[batch],
                split_name='base',
                action_pool=['action a', 'action b'],
                candidate_embeddings=candidates,
                predictor=lambda _: ground_truth.clone(),
                output_dir=Path(temporary),
            )
            arrays = np.load(Path(temporary) / 'raw' / 'base_embeddings.npz')
            self.assertEqual(arrays['pred_embeddings'].shape, (1, 2, 2))
            self.assertEqual(len(exported['rows']), 2)
            self.assertEqual(exported['summary']['overall']['correct_mean'], 1.0)
            self.assertEqual(exported['summary']['overall']['mse_mean'], 0.0)
            self.assertEqual(exported['summary']['overall']['cosine_mean'], 1.0)


if __name__ == '__main__':
    unittest.main()

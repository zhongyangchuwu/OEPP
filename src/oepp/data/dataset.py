"""PyTorch sequence dataset backed by an explicit validated SplitBundle."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .bundle import Partition, SplitBundle
from .features import (
    FeatureKind,
    FeatureRoots,
    default_videoclip_root,
    feature_dimension,
    resolve_feature_path,
)
from .windows import PaddingPolicy, SequenceWindow, build_windows


class SequenceDataset(torch.utils.data.Dataset[tuple[Any, ...]]):
    """Load OEPP endpoint features and action targets for one named partition."""

    observation_count = 3

    def __init__(
        self,
        bundle: SplitBundle,
        partition: Partition | str,
        *,
        feature: FeatureKind | str,
        horizon: int,
        padding: PaddingPolicy | str = PaddingPolicy.LEFT,
        total_pool: bool = False,
        feature_roots: FeatureRoots | None = None,
    ) -> None:
        self.bundle = bundle
        self.partition = Partition(partition)
        self.feature = FeatureKind(feature)
        self.horizon = int(horizon)
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        self.padding = PaddingPolicy(padding)
        self.dimension = feature_dimension(self.feature)
        self.feature_roots = feature_roots or FeatureRoots(videoclip=default_videoclip_root())
        self.windows = build_windows(
            self.bundle.partition_records(self.partition),
            split_id=self.bundle.split_id,
            partition=self.partition.value,
            horizon=self.horizon,
            padding=self.padding,
        )
        self.action_pool = self.bundle.action_pool_for(self.partition, total=total_pool)
        self.action_to_label = {action: index for index, action in enumerate(self.action_pool)}
        self.action_embeddings = self.bundle.embedding_dict(self.feature)

    def __len__(self) -> int:
        return len(self.windows)

    def metadata_at(self, index: int) -> dict[str, Any]:
        window = self.windows[index]
        return {
            "sample_id": window.sample_id,
            "split_id": window.split_id,
            "split": window.partition,
            "dataset": window.dataset,
            "task_name": window.task_name,
            "task_id": window.task_id,
            "task_id_old": window.task_id_old,
            "vid": window.vid,
            "source_video_index": window.source_video_index,
            "start_step": window.start_step,
            "end_step": window.end_step,
            "is_padded": window.is_padded,
            "pad_count": window.pad_count,
            "actions": list(window.actions),
        }

    def _load_feature_array(self, window: SequenceWindow) -> np.ndarray:
        record = {
            "dataset": window.dataset,
            "vid": window.vid,
            "task_name": window.task_name,
            "task_id_old": window.task_id_old,
        }
        path = resolve_feature_path(record, self.feature, self.feature_roots)
        if not path.is_file():
            raise FileNotFoundError(f"Feature file does not exist: {path}")
        loaded = np.load(path, allow_pickle=False)
        if self.feature is FeatureKind.S3D:
            if (
                not isinstance(loaded, np.lib.npyio.NpzFile)
                or "frames_features" not in loaded.files
            ):
                raise ValueError(f"S3D feature archive lacks frames_features: {path}")
            values = loaded["frames_features"]
            loaded.close()
        else:
            if not isinstance(loaded, np.ndarray):
                raise ValueError(f"VideoCLIP feature file is not an array: {path}")
            values = loaded
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError(
                f"Feature shape {tuple(values.shape)} does not match {self.feature.value} "
                f"dimension {self.dimension}: {path}"
            )
        return values

    def _endpoint_embedding(
        self, values: np.ndarray, timestamp: float, *, is_start: bool
    ) -> torch.Tensor:
        index = int(timestamp)
        result = torch.zeros((self.observation_count, self.dimension), dtype=torch.float32)
        for offset in range(self.observation_count):
            frame_index = (
                index + offset if is_start else index - self.observation_count + 1 + offset
            )
            if 0 <= frame_index < values.shape[0]:
                result[offset] = torch.as_tensor(values[frame_index], dtype=torch.float32)
        return result.reshape(-1)

    def _action_tensor(self, actions: tuple[str, ...]) -> torch.Tensor:
        vectors: list[torch.Tensor] = []
        for action in actions:
            try:
                vector = torch.as_tensor(self.action_embeddings[action], dtype=torch.float32)
            except KeyError as error:
                raise ValueError(f"Action embedding missing for {action!r}") from error
            vector = vector.reshape(-1)
            if vector.numel() != self.dimension:
                raise ValueError(f"Action embedding has wrong dimension for {action!r}")
            vectors.append(vector)
        return torch.stack(vectors)

    def _labels(self, actions: tuple[str, ...]) -> torch.Tensor:
        try:
            return torch.tensor(
                [self.action_to_label[action] for action in actions], dtype=torch.long
            )
        except KeyError as error:
            raise ValueError(
                f"Action {error.args[0]!r} is absent from the selected action pool"
            ) from error

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        window = self.windows[index]
        values = self._load_feature_array(window)
        start_frames = self._endpoint_embedding(values, window.start_segment[0], is_start=True)
        end_frames = self._endpoint_embedding(values, window.end_segment[1], is_start=False)
        actions = window.actions
        labels = self._labels(actions)
        return (
            window.vid,
            start_frames,
            end_frames,
            torch.cat((start_frames, end_frames)),
            list(actions),
            self._action_tensor(actions),
            list(actions[1:-1]),
            labels,
            labels[1:-1],
        )

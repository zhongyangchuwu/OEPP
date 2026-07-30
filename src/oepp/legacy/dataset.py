import json
import os

import numpy as np
import torch

from .feature_paths import videoclip_feature_path, videoclip_root


class Video(torch.utils.data.Dataset):
    def __init__(self, root, split, feat, is_val=0):
        self.root = root
        self.split = split
        self.is_val = is_val
        self.feat = feat
        self.videoclip_root = videoclip_root() if self.feat == "videoclip" else None
        self.M = 3
        if self.feat == "s3d":
            self.zeros_frame = torch.zeros(512)
        elif self.feat == "videoclip":
            self.zeros_frame = torch.zeros(768)
        else:
            self.zeros_frame = torch.zeros(512)

        # 读取json文件
        if self.is_val == 0:  # train
            video_json = "train_train_base_dataset_" + str(self.split) + ".json"
        elif self.is_val == 1:  # test_novel
            video_json = "novel_dataset_" + str(self.split) + ".json"
        elif self.is_val == 2:  # test_base
            video_json = "test_base_dataset_" + str(self.split) + ".json"
        elif self.is_val == 3:  # val
            video_json = "train_val_base_dataset_" + str(self.split) + ".json"
        video_json_dir = os.path.join(self.root, video_json)
        if os.path.exists(video_json_dir):
            with open(video_json_dir) as f:
                self.json_data = json.load(f)
            print(f"Loaded {video_json_dir}")
        else:
            print("no json data")

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        video_id = self.json_data[index]

        # feature path
        if self.feat == "s3d":
            if video_id["dataset"] == "COIN":
                name = (
                    video_id["task_name"]
                    + "_"
                    + str(video_id["task_id_old"])
                    + "_"
                    + video_id["vid"]
                    + ".npy"
                )
                images_d = np.load(
                    os.path.join("/data0/wuyilu/data/COIN/full_npy", name),
                    encoding="bytes",
                    allow_pickle=True,
                )  # T*512
                images = images_d["frames_features"]
            else:
                name = video_id["task_id_old"] + "_" + video_id["vid"] + ".npy"
                images_d = np.load(
                    os.path.join("/data0/wuyilu/data/ori_processed_data", name),
                    encoding="bytes",
                    allow_pickle=True,
                )
                images = images_d["frames_features"]
        elif self.feat == "videoclip":
            images = np.load(
                videoclip_feature_path(video_id["dataset"], video_id["vid"], self.videoclip_root)
            )
        # print(images.shape)

        start_frames_list = []
        end_frames_list = []
        action_list = []

        for step in video_id["anno"]:
            start_frames = []
            end_frames = []
            segment = step["segment"]
            action = step["action"]
            # print(action)
            action_list.append(str(action))
            start = int(segment[0])
            end = int(segment[1])
            if end >= images.shape[0]:
                end = images.shape[0] - 1
            for i in range(self.M):
                if start + i >= images.shape[0]:  #
                    start_frames.extend(self.zeros_frame)
                else:
                    start_frames.extend(images[start + i])
                if end - self.M + 1 + i < 0:
                    end_frames.extend(self.zeros_frame)
                else:
                    end_frames.extend(images[end - self.M + 1 + i])
            # print(len(start_frames))
            start_frames = torch.tensor(start_frames)
            end_frames = torch.tensor(end_frames)
            start_frames_list.append(start_frames)
            end_frames_list.append(end_frames)

        return video_id["vid"], start_frames_list, end_frames_list, action_list


class Seq_action(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        split,
        feat,
        T,  # T,num of actions
        is_pad,
        is_total,
        is_val,
    ):
        self.root = root
        self.split = split
        self.feat = feat
        self.T = T
        self.is_pad = is_pad
        self.is_total = is_total
        self.is_val = is_val
        self.videos = Video(root=self.root, split=self.split, feat=self.feat, is_val=self.is_val)
        self.partition = {0: "train", 1: "novel", 2: "base", 3: "validation"}[self.is_val]
        self.seq_list = []
        print("len of videos: ", len(self.videos))
        if self.feat == "videoclip":
            with open("data/vc_action_feat_dict.json") as f:
                self.actions_text_dict = json.load(f)
        elif self.feat == "s3d":
            with open("data/s3d_action_feat_dict.json") as f:
                self.actions_text_dict = json.load(f)

        for source_video_index, video in enumerate(self.videos):
            source_video = self.videos.json_data[source_video_index]
            vid, start_frames_list, end_frames_list, action_list = video
            length = len(action_list)
            if self.T <= length:
                for start_id in range(length - self.T + 1):
                    end_id = start_id + self.T - 1
                    actions = list(action_list[start_id : end_id + 1])
                    self.seq_list.append(
                        {
                            "sample_id": (
                                f"split{self.split}_{self.partition}_video{source_video_index}_"
                                f"start{start_id}_T{self.T}"
                            ),
                            "vid": vid,
                            "dataset": source_video.get("dataset", ""),
                            "task_name": source_video.get("task_name", ""),
                            "task_id": source_video.get("task_id", ""),
                            "task_id_old": source_video.get("task_id_old", ""),
                            "source_video_index": source_video_index,
                            "start_step": start_id,
                            "end_step": end_id,
                            "is_padded": False,
                            "pad_count": 0,
                            "start_frames": start_frames_list[start_id],
                            "end_frames": end_frames_list[end_id],
                            "actions": actions,
                        }
                    )
            elif self.is_pad == 1:
                pad_count = self.T - length
                actions = [action_list[0]] * pad_count + list(action_list)
                self.seq_list.append(
                    {
                        "sample_id": (
                            f"split{self.split}_{self.partition}_video{source_video_index}_"
                            f"padded_T{self.T}"
                        ),
                        "vid": vid,
                        "dataset": source_video.get("dataset", ""),
                        "task_name": source_video.get("task_name", ""),
                        "task_id": source_video.get("task_id", ""),
                        "task_id_old": source_video.get("task_id_old", ""),
                        "source_video_index": source_video_index,
                        "start_step": 0,
                        "end_step": length - 1,
                        "is_padded": True,
                        "pad_count": pad_count,
                        "start_frames": start_frames_list[0],
                        "end_frames": end_frames_list[-1],
                        "actions": actions,
                    }
                )
        print("total sequences length:", len(self.seq_list))
        if self.is_val == 0 or self.is_val == 2 or self.is_val == 3:
            action_pool_file = "data/base_action_pool_" + str(self.split) + ".json"
        else:
            action_pool_file = "data/novel_action_pool_" + str(self.split) + ".json"
        if self.is_total == 1:
            action_pool_file = "data/total_action_pool.json"
        with open(action_pool_file) as f:
            self.action_pool = json.load(f)
        self.action_to_label = {action: index for index, action in enumerate(self.action_pool)}

    def get_labels(self, actions):
        labels = torch.empty(self.T, dtype=torch.long)
        for index, action in enumerate(actions):
            try:
                labels[index] = self.action_to_label[action]
            except KeyError as error:
                raise ValueError(
                    f"Action {action!r} is absent from the selected action pool"
                ) from error
        return labels

    def get_action_tensor(self, action_list):
        tensor_list = []
        for action in action_list:
            text_embedding = torch.tensor(self.actions_text_dict[action])
            tensor_list.append(text_embedding)
        text_tensor = torch.cat([tensor for tensor in tensor_list], dim=0)
        return text_tensor

    def __len__(self):
        return len(self.seq_list)

    def metadata_at(self, index):
        seq = self.seq_list[index]
        return {
            "sample_id": seq["sample_id"],
            "split": self.partition,
            "dataset": seq["dataset"],
            "task_name": seq["task_name"],
            "task_id": seq["task_id"],
            "task_id_old": seq["task_id_old"],
            "vid": seq["vid"],
            "source_video_index": seq["source_video_index"],
            "start_step": seq["start_step"],
            "end_step": seq["end_step"],
            "is_padded": seq["is_padded"],
            "pad_count": seq["pad_count"],
            "actions": list(seq["actions"]),
        }

    def __getitem__(self, index):
        seq = self.seq_list[index]
        vid = seq["vid"]
        start_frames = seq["start_frames"]
        end_frames = seq["end_frames"]
        start_end_frames = torch.cat((start_frames, end_frames), dim=0)
        action_list = seq["actions"]
        action_tensor = self.get_action_tensor(action_list)
        mid_action_list = action_list[1:-1]
        labels = self.get_labels(action_list)
        mid_labels = labels[1:-1]
        return (
            vid,
            start_frames,
            end_frames,
            start_end_frames,
            action_list,
            action_tensor,
            mid_action_list,
            labels,
            mid_labels,
        )


if __name__ == "__main__":
    train_dataset = Seq_action(
        root="/data0/wuyilu/otpp/data_p_1114",
        split=1,
        T=6,
        feat="videoclip",
        is_pad=0,
        is_total=0,
        is_val=0,
    )

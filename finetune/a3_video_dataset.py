"""A3 cookie-transfer training dataset (memmap-backed).

Subclasses RobotVideoDataset so the video concat/resize/normalize pipeline,
text-context cache and the returned sample dict stay identical to the official
LIBERO training path. Only the sample source is replaced: BaseLerobotDataset
(LeRobot v2.x layout) is swapped for a shim that reads the memmaps produced
by finetune/prepare_a3_arrays.py.
"""
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from .robot_video_dataset import RobotVideoDataset
from .utils.normalizer import load_dataset_stats_from_json
from ..dataset_utils import CenterCrop, Normalize, ResizeSmallestSideAspectPreserving

DEFAULT_A3_TASK = "transfer 10 cookies into target box"


class _A3SampleSource:
    """Stand-in for BaseLerobotDataset: serves preprocessed windows."""

    def __init__(self, parent):
        self.parent = parent
        self.processor = parent.processor

    def __len__(self):
        return len(self.parent._windows)

    def __getitem__(self, idx):
        return self.parent._make_processed_sample(idx)


class A3VideoDataset(RobotVideoDataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[224, 448],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.0,
        is_training_set=True,
        global_sample_stride=1,
        action_video_freq_ratio=4,
        skip_padding_as_possible=False,
        max_padding_retry=3,
        concat_multi_camera="horizontal",
        override_instruction=None,
    ):
        if global_sample_stride != 1:
            raise ValueError("A3VideoDataset requires global_sample_stride=1")
        if concat_multi_camera != "horizontal":
            raise ValueError("A3VideoDataset requires concat_multi_camera='horizontal'")

        self.num_frames = int(num_frames)
        self.action_video_freq_ratio = int(action_video_freq_ratio)
        assert (self.num_frames - 1) % self.action_video_freq_ratio == 0
        assert ((self.num_frames - 1) // self.action_video_freq_ratio) % 4 == 0
        self.video_sample_indices = list(range(0, self.num_frames, self.action_video_freq_ratio))
        self.camera_key = camera_key
        self.video_size = [int(v) for v in video_size]
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = int(context_len)
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction
        self._ctx_cache = None

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]}
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]}
        )
        self.normalize_transform = Normalize(args={"mean": 0.5, "std": 0.5})

        data_dir = Path(str(dataset_dirs[0]))
        frames_path = data_dir / "frames.dat"
        arrays_path = data_dir / "arrays.npz"
        if not frames_path.exists() or not arrays_path.exists():
            raise FileNotFoundError(
                f"missing frames.dat / arrays.npz in {data_dir}; "
                "run finetune/prepare_a3_arrays.py first"
            )
        arr = np.load(arrays_path, allow_pickle=True)
        self._action = np.ascontiguousarray(arr["action"], dtype=np.float32)  # [N, 7]
        self._proprio = np.ascontiguousarray(arr["proprio"], dtype=np.float32)  # [N, 8]
        self._task_str = str(arr["task_str"]) if "task_str" in arr.files else DEFAULT_A3_TASK

        # frames.dat is headerless raw bytes (not a .npy), so build the memmap explicitly.
        n_total = int(self._action.shape[0])
        frame_shape = (self.video_size[0], self.video_size[1], 3)
        expected_bytes = n_total * int(np.prod(frame_shape))
        actual_bytes = frames_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"frames.dat size mismatch: {actual_bytes} != {expected_bytes} "
                f"for shape ({n_total}, {frame_shape})"
            )
        self._frames = np.memmap(
            frames_path, dtype=np.uint8, mode="r", shape=(n_total, *frame_shape)
        )

        self._windows = []
        for start, length in zip(arr["ep_start"], arr["ep_len"]):
            n = int(length) - self.num_frames + 1
            if n < 1:
                raise ValueError(f"episode shorter than num_frames: {int(length)}")
            self._windows.extend(range(int(start), int(start) + n))

        self._image_keys = [m["key"] for m in shape_meta["images"]]
        self._action_key = shape_meta["action"][0]["key"]
        self._state_key = shape_meta["state"][0]["key"]
        self._cam_w = self._frames.shape[2] // len(self._image_keys)

        if processor is None:
            raise ValueError("`processor` is required")
        if isinstance(processor, DictConfig):
            processor = instantiate(processor)
        if not pretrained_norm_stats:
            raise ValueError("`pretrained_norm_stats` is required (empirical A3 stats)")
        processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(pretrained_norm_stats)))
        if is_training_set:
            processor.train()
        else:
            processor.eval()
        self.processor = processor

        self.lerobot_dataset = _A3SampleSource(self)

    def _make_processed_sample(self, idx):
        start = self._windows[idx]
        s = slice(start, start + self.num_frames)
        imgs = np.array(self._frames[s], copy=True)  # [T, 224, 448, 3] uint8 (writable)
        images = {}
        for i, key in enumerate(self._image_keys):
            cam = imgs[:, :, i * self._cam_w:(i + 1) * self._cam_w, :]
            images[key] = torch.from_numpy(np.ascontiguousarray(cam)).permute(0, 3, 1, 2)
        raw = {
            "idx": int(idx),
            "task": self._task_str,
            "images": images,
            "action": {self._action_key: torch.from_numpy(self._action[s][:-1].copy())},
            "state": {self._state_key: torch.from_numpy(self._proprio[s].copy())},
            "action_is_pad": torch.zeros(self.num_frames - 1, dtype=torch.bool),
            "state_is_pad": torch.zeros(self.num_frames, dtype=torch.bool),
            "image_is_pad": torch.zeros(self.num_frames, dtype=torch.bool),
        }
        return self.processor.preprocess(raw)

    def _get_cached_text_context(self, prompt):
        if self._ctx_cache is None:
            self._ctx_cache = super()._get_cached_text_context(prompt)
        return self._ctx_cache

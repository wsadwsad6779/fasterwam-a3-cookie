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

from fasterwam.utils.logging_config import get_logger

from .robot_video_dataset import RobotVideoDataset
from .utils.normalizer import load_dataset_stats_from_json
from ..dataset_utils import CenterCrop, Normalize, ResizeSmallestSideAspectPreserving

logger = get_logger(__name__)

DEFAULT_A3_TASK = "transfer 10 cookies into target box"

# What `force_windows` adds to a sample; `_extra_sample_fields` forwards exactly
# these through `RobotVideoDataset._get`'s fixed-key view.
FORCE_SAMPLE_KEYS = (
    "force_history",
    "force_history_mask",
    "force_future",
    "force_future_mask",
    "window_offset",
)


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
        force_arrays_path=None,
        force_channels=None,
        force_hist_len=0,
        force_horizon=0,
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
        self._window_ep_start = []
        self._window_ep_end = []
        for start, length in zip(arr["ep_start"], arr["ep_len"]):
            n = int(length) - self.num_frames + 1
            if n < 1:
                raise ValueError(f"episode shorter than num_frames: {int(length)}")
            self._windows.extend(range(int(start), int(start) + n))
            self._window_ep_start.extend([int(start)] * n)
            self._window_ep_end.extend([int(start) + int(length)] * n)

        self._force = None
        self._force_channels = None
        if force_arrays_path is not None:
            self._load_force_arrays(
                str(force_arrays_path),
                force_channels,
                force_hist_len,
                force_horizon,
                n_total,
                arr["ep_start"],
                arr["ep_len"],
            )

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

    def _load_force_arrays(
        self,
        force_arrays_path,
        force_channels,
        force_hist_len,
        force_horizon,
        n_total,
        ep_start,
        ep_len,
    ):
        """Attach the 18-channel force table to the same row index space as the video.

        The force npz is a separate file written by a different script, so
        nothing guarantees it describes the same episodes. If its row count or
        episode boundaries drifted, every force window would silently point at
        another demonstration's data -- the same class of bug as a counter that
        resets mid-training. Therefore the boundaries are compared here and any
        mismatch is fatal, not a warning.
        """
        force_hist_len = int(force_hist_len)
        force_horizon = int(force_horizon)
        if force_hist_len <= 0 or force_horizon <= 0:
            raise ValueError(
                "`force_hist_len` and `force_horizon` must be positive when force arrays are given"
            )
        if force_horizon != self.num_frames - 1:
            raise ValueError(
                f"`force_horizon` must equal the action horizon {self.num_frames - 1} "
                f"(one force step per action token), got {force_horizon}"
            )
        if not force_channels:
            raise ValueError("`force_channels` must list the raw columns to gather")
        fpath = Path(force_arrays_path)
        if not fpath.exists():
            raise FileNotFoundError(f"force arrays not found: {fpath}")
        farr = np.load(fpath, allow_pickle=True)
        # `force18` is the untouched per-frame table in newtons. `force6` /
        # `force6_zeroed` next to it are pre-gathered and pre-zeroed by an older
        # script whose zero reference used frames after `t`, which an online
        # robot cannot compute at its own first step -- so it is deliberately
        # not read here and the model normalises raw values instead.
        if "force18" in farr.files:
            key = "force18"
        elif "force" in farr.files:
            key = "force"
        else:
            raise KeyError(f"{fpath} has no `force18` / `force` entry (has {list(farr.files)})")
        force = np.ascontiguousarray(farr[key], dtype=np.float32)
        if force.ndim != 2:
            raise ValueError(f"`{key}` must be 2D [N, C], got shape {force.shape}")
        if force.shape[0] != n_total:
            raise ValueError(
                f"force table has {force.shape[0]} rows but the video arrays have {n_total}; "
                "the two preprocessing runs are not over the same corpus."
            )
        for key in ("ep_start", "ep_len"):
            if key not in farr.files:
                raise KeyError(f"{fpath} is missing '{key}', so episode boundaries cannot be checked")
            got, want = np.asarray(farr[key]), np.asarray(ep_start if key == "ep_start" else ep_len)
            if got.shape != want.shape or not np.array_equal(got, want):
                raise ValueError(
                    f"force arrays disagree with the video arrays on '{key}': "
                    f"{got[:5].tolist()}... vs {want[:5].tolist()}..."
                )
        channels = [int(c) for c in force_channels]
        if min(channels) < 0 or max(channels) >= force.shape[1]:
            raise ValueError(
                f"`force_channels` {channels} out of range for a {force.shape[1]}-column table"
            )
        if len(set(channels)) != len(channels):
            raise ValueError(f"`force_channels` contains duplicates: {channels}")

        self._force = force
        self._force_channels = channels
        self._force_hist_len = force_hist_len
        self._force_horizon = force_horizon
        logger.info(
            "Force conditioning: gathered columns %s of %d from %s (%d rows, %d episodes).",
            channels, force.shape[1], fpath.name, force.shape[0], len(ep_start),
        )

    def force_windows(self, window_idx):
        """Raw-newton force history and future for one training window.

        `window_idx` indexes `self._windows`; the window's *current* frame is its
        first row `s`, because the action head predicts the 32 transitions
        starting there. History therefore ends at `s` and the future is
        `s+1 .. s+32`, both clipped to the episode so a window can never borrow
        force from a neighbouring demonstration (§六).
        """
        if self._force is None:
            raise RuntimeError("force windows requested but no force arrays were loaded")
        s = int(self._windows[window_idx])
        ep_start = int(self._window_ep_start[window_idx])
        ep_end = int(self._window_ep_end[window_idx])
        hlen = self._force_hist_len
        flen = self._force_horizon

        lo = max(ep_start, s - (hlen - 1))
        hist = self._force[lo:s + 1, self._force_channels]
        hist_len = int(hist.shape[0])
        force_history = np.zeros((hlen, len(self._force_channels)), dtype=np.float32)
        force_history[hlen - hist_len:] = hist
        history_mask = np.zeros(hlen, dtype=bool)
        history_mask[hlen - hist_len:] = True

        fut_hi = min(ep_end - 1, s + flen)
        fut = self._force[s + 1:fut_hi + 1, self._force_channels]
        fut_len = int(fut.shape[0])
        force_future = np.zeros((flen, len(self._force_channels)), dtype=np.float32)
        force_future[:fut_len] = fut
        future_mask = np.zeros(flen, dtype=bool)
        future_mask[:fut_len] = True

        return {
            "force_history": torch.from_numpy(force_history),
            "force_history_mask": torch.from_numpy(history_mask),
            "force_future": torch.from_numpy(force_future),
            "force_future_mask": torch.from_numpy(future_mask),
            "window_offset": int(s),
        }

    def force_rows(self, rows):
        """Gathered raw force for absolute row indices -- the leak gate's retriever.

        Deliberately a separate read path from `force_windows`, so the gate
        compares the dataset against the source table rather than against itself.
        """
        if self._force is None:
            raise RuntimeError("force_rows requested but no force arrays were loaded")
        rows = np.asarray(rows, dtype=np.int64)
        if rows.min() < 0 or rows.max() >= self._force.shape[0]:
            raise IndexError(f"force row index out of range: [{rows.min()}, {rows.max()}]")
        # Gather rows first, then select the channels. Passing both index arrays to
        # `[]` at once would be elementwise advanced indexing, which only broadcasts
        # when the two lengths match -- it never does for a [N, 6] window read.
        gathered = self._force[rows.reshape(-1)][:, self._force_channels]
        return np.ascontiguousarray(gathered).reshape(
            rows.shape + (len(self._force_channels),)
        )

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
        sample = self.processor.preprocess(raw)
        if self._force is not None:
            sample.update(self.force_windows(idx))
        return sample

    def _extra_sample_fields(self, sample):
        if self._force is None:
            return {}
        missing = [key for key in FORCE_SAMPLE_KEYS if key not in sample]
        if missing:
            raise KeyError(
                "force arrays are loaded but the served sample lacks %s; the training "
                "batch would otherwise run without force conditioning while the model "
                "expects it." % missing
            )
        return {key: sample[key] for key in FORCE_SAMPLE_KEYS}

    def _get_cached_text_context(self, prompt):
        if self._ctx_cache is None:
            self._ctx_cache = super()._get_cached_text_context(prompt)
        return self._ctx_cache

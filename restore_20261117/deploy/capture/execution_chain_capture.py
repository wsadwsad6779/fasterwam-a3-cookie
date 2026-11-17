"""Passive execution-chain evidence. No model loader or simulator imports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import uuid

import numpy as np


def copy_cpu(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.array(value, copy=True)


def checked(value, shape, name):
    array = copy_cpu(value)
    if array.shape != shape:
        raise ValueError(f"{name}: shape {array.shape}, expected {shape}")
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name}: expected finite numeric values")
    return array


def hash_files(artifacts):
    """Require actual weight/config/source files; stream rather than loading weights."""
    if not all(artifacts.get(role) for role in ("weights", "config", "source")):
        raise ValueError("artifacts must include weights, config and source paths")
    result = {}
    for role, paths in artifacts.items():
        if isinstance(paths, (str, Path)):
            paths = [paths]
        result[role] = []
        for path in paths:
            path = Path(path).resolve(strict=True)
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            result[role].append({"path": str(path), "sha256": digest.hexdigest(),
                                 "bytes": path.stat().st_size})
    return result


class Capture:
    def __init__(self, artifacts):
        self.run_id = uuid.uuid4().hex  # OS randomness; never Python/NumPy RNG.
        self.artifacts = hash_files(artifacts)
        self.arrays = {}
        self.chunks = []
        self.mapping = []
        self.frame = None
        self.observation = None
        self.opening = None
        self.status = "initialized"
        self.error = None

    def reset(self, obs):
        if self.frame is not None:
            raise ValueError("one Capture represents exactly one reset/run")
        self.opening = self.state(obs)[7].item()
        self.observation = obs
        self.frame = 0

    @staticmethod
    def state(obs):
        if "observation.state" not in obs:
            raise ValueError("missing observation.state")
        return checked(obs["observation.state"], (16,), "observation.state")

    def chunk(self, frame, raw, denorm, rewritten, adapter):
        if frame != self.frame:
            raise ValueError(f"chunk frame {frame} != current observation frame {self.frame}")
        n = len(rewritten)
        if n == 0:
            raise ValueError("empty action chunk")
        values = {
            "raw": checked(raw, (1, n, 7), "raw"),
            "denorm": checked(denorm, (1, n, 7), "denorm"),
            "rewritten": checked(rewritten, (n, 7), "rewritten"),
            "adapter": checked(adapter, (n, 14), "adapter"),
        }
        chunk_id = len(self.chunks)
        for name, value in values.items():
            self.arrays[f"chunk_{chunk_id}_{name}"] = value
        self.chunks.append({"run_id": self.run_id, "chunk_id": chunk_id,
                            "obs_frame": frame, "predicted_rows": n, "executed_rows": 0})

    def selected(self, action):
        if self.frame is None or not self.chunks:
            raise ValueError("step requires reset and a captured chunk")
        chunk = self.chunks[-1]
        row = chunk["executed_rows"]
        if row >= chunk["predicted_rows"]:
            raise ValueError("step exceeds predicted chunk rows")
        if chunk["obs_frame"] + row != self.frame:
            raise ValueError("chunk row / observation frame mapping mismatch")
        handed = checked(action, (14,), "handed14")
        expected = self.arrays[f"chunk_{chunk['chunk_id']}_adapter"][row]
        if not np.array_equal(handed, expected):
            raise ValueError("handed14 differs from selected adapter row")
        return handed

    def executed(self, handed, obs, info):
        for key in ("requested_action", "applied_action"):
            if key not in info:
                raise ValueError(f"missing info.{key}")
        requested = checked(info["requested_action"], (14,), "requested_action")
        applied = checked(info["applied_action"], (16,), "applied_action")
        opening_after = self.state(obs)[7].item()
        step = len(self.mapping)
        chunk = self.chunks[-1]
        self.mapping.append({"run_id": self.run_id, "chunk_id": chunk["chunk_id"],
                             "chunk_row": chunk["executed_rows"], "step": step,
                             "obs_frame_before": self.frame, "obs_frame_after": self.frame + 1,
                             "opening_before": self.opening, "opening_after": opening_after})
        for name, value in (("handed14", handed), ("requested14", requested),
                            ("applied16", applied)):
            self.arrays[f"step_{step}_{name}"] = value
        chunk["executed_rows"] += 1
        self.frame += 1
        self.observation = obs
        self.opening = opening_after

    def save(self, path):
        metadata = {"schema_version": 2, "run_id": self.run_id, "artifacts": self.artifacts,
                    "chunks": self.chunks, "steps": self.mapping, "status": self.status,
                    "error": self.error, "reset_obs_frame": 0,
                    "opening": {"key": "observation.state", "index": 7,
                                "unit": "normalized opening (not meters)",
                                "definition": "1 - mean(finger qpos) / GRIPPER_RANGE_M"}}
        payload = dict(self.arrays)
        payload["metadata"] = np.asarray(json.dumps(metadata, allow_nan=False))
        with open(path, "xb") as stream:
            np.savez_compressed(stream, **payload)

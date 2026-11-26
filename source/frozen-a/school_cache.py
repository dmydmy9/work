"""Validated training checkpoints and projection-specific scores in SDK round state."""

import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np


MAX_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_UPLOAD_FILES = 64
MAX_CHECKPOINT_BYTES = 16 * 1024 * 1024
KEY = re.compile(r"[a-f0-9]{64}")


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def fingerprint(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def array_digest(values, dtype):
    array = np.ascontiguousarray(values, dtype=dtype)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def projection_digest(features, targets, ordered_ids):
    return fingerprint({"ids": list(ordered_ids), "shape": features.shape,
                        "features": array_digest(features, "<f8"), "targets": array_digest(targets, "<i8")})


def file_usage(directory):
    sizes = [path.stat().st_size for path in directory.rglob("*") if path.is_file()]
    return len(sizes), sum(sizes)


def check_candidate_upload(directory, limits):
    count, size = file_usage(directory)
    budget = min(MAX_UPLOAD_BYTES, limits.get("artifact_bytes", MAX_UPLOAD_BYTES),
                 limits.get("candidate_upload_bytes_per_round", MAX_UPLOAD_BYTES))
    if count > MAX_UPLOAD_FILES or size > budget:
        raise ValueError("school A candidate exceeds host upload limits")
    return size


class RoundCache:
    def __init__(self, state, round_index):
        self.directory = state / "exact-cache"
        self.directory.mkdir(exist_ok=True)
        self.index = self.directory / "index.json"
        self.round_index = round_index
        self.entries = self._load()

    def _load(self):
        if not self.index.is_file():
            return {}
        try:
            envelope = json.loads(self.index.read_bytes())
            entries = envelope["entries"]
            if envelope["version"] == 1 and isinstance(entries, dict) and fingerprint(entries) == envelope["digest"]:
                return entries
        except (OSError, UnicodeError, ValueError, KeyError, TypeError):
            pass
        return {}

    def _checkpoint(self, key):
        return self.directory / (key + ".npz")

    def training(self, key):
        entry = self.entries.get(key)
        if entry is None:
            return None
        checkpoint = self._checkpoint(key)
        if (not checkpoint.is_file() or checkpoint.is_symlink()
                or checkpoint.stat().st_size != entry["checkpoint_bytes"]
                or entry["checkpoint_bytes"] > MAX_CHECKPOINT_BYTES
                or file_digest(checkpoint) != entry["checkpoint_sha256"]):
            del self.entries[key]
            checkpoint.unlink(missing_ok=True)
            return None
        entry["last_used_round"] = self.round_index
        return checkpoint, entry

    def add_training(self, key, checkpoint, **metadata):
        saved = self._checkpoint(key)
        shutil.copyfile(checkpoint, saved)
        entry = {**metadata, "checkpoint_sha256": file_digest(saved), "checkpoint_bytes": saved.stat().st_size,
                 "created_round": self.round_index, "last_used_round": self.round_index, "scores": {}}
        self.entries[key] = entry
        return entry

    def add_score(self, entry, key, metrics, prediction_sha256):
        entry["scores"][key] = {"round": self.round_index, "metrics": metrics,
                               "prediction_sha256": prediction_sha256}

    def persist(self, limits, candidate_bytes):
        budget = min(MAX_UPLOAD_BYTES - candidate_bytes, limits.get("state_bytes", MAX_UPLOAD_BYTES),
                     limits.get("state_upload_bytes_per_round", MAX_UPLOAD_BYTES))
        for checkpoint in self.directory.glob("*.npz"):
            if KEY.fullmatch(checkpoint.stem) and checkpoint.stem not in self.entries:
                checkpoint.unlink()
        self.index.unlink(missing_ok=True)
        files, size = file_usage(self.directory.parent)
        checkpoints = [key for key in self.entries if KEY.fullmatch(key) and self._checkpoint(key).is_file()]
        self.entries = {key: self.entries[key] for key in checkpoints}
        evict = iter(sorted(checkpoints, key=lambda key: (self.entries[key]["last_used_round"], key)))
        # The SDK submits the newly written index and each checkpoint as ordinary files.
        while self.entries:
            payload = json_bytes({"version": 1, "entries": self.entries, "digest": fingerprint(self.entries)})
            if files + 1 <= MAX_UPLOAD_FILES and size + len(payload) <= budget:
                self.index.write_bytes(payload)
                return
            key = next(evict)
            size -= self._checkpoint(key).stat().st_size
            files -= 1
            self._checkpoint(key).unlink()
            del self.entries[key]
        if files > MAX_UPLOAD_FILES or size > budget:
            raise ValueError("school A round history exceeds host state upload limits")

"""Configuration and preconditioner support for the KVWeave codec."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Optional

import numpy as np


@dataclass
class KVWeaveCodecConfig:
    """Typed settings shared by the KVWeave codec and preconditioner."""

    quantize: bool = True
    qbit: int = 4
    scaling_method: str = "per_channel"
    rh: bool = True
    asym: bool = True
    log: bool = False
    precond: bool = False
    num_threads: int = 1
    precond_seed: int = 42
    precond_path: Optional[str] = None

    def __post_init__(self) -> None:
        self._pd_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._pd_lock = threading.Lock()
        self._pd_file: Optional[dict[str, object]] = None
        self.scaling_method = str(self.scaling_method)

    def get_pd_matrix(
        self, hadamard_size: int
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return deterministic or file-backed ``(signs, permutation)``."""
        cached = self._pd_cache.get(hadamard_size)
        if cached is not None:
            return cached
        with self._pd_lock:
            cached = self._pd_cache.get(hadamard_size)
            if cached is None:
                cached = (
                    self._load_pd_from_file(hadamard_size)
                    if self.precond_path
                    else self._generate_pd(hadamard_size)
                )
                self._pd_cache[hadamard_size] = cached
        return cached

    def _generate_pd(self, hadamard_size: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.precond_seed)
        signs = rng.choice([-1.0, 1.0], size=hadamard_size).astype(np.float32)
        perm = rng.permutation(hadamard_size).astype(np.int32)
        return signs, perm

    def _load_pd_from_file(
        self, hadamard_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._pd_file is None:
            self._pd_file = self._read_pd_file(self.precond_path)
        signs_key = f"signs_{hadamard_size}"
        perm_key = f"perm_{hadamard_size}"
        if signs_key not in self._pd_file or perm_key not in self._pd_file:
            raise KeyError(
                f"preconditioner has no matrices for hadamard_size={hadamard_size}"
            )
        signs = np.asarray(self._pd_file[signs_key], dtype=np.float32).reshape(-1)
        perm = np.asarray(self._pd_file[perm_key], dtype=np.int32).reshape(-1)
        return signs, perm

    @staticmethod
    def _read_pd_file(path: str | None) -> dict[str, object]:
        if path is None:
            raise ValueError("precond_path is required")
        if path.endswith(".npz"):
            with np.load(path) as data:
                return {key: data[key] for key in data.files}
        if path.endswith((".pt", ".pth")):
            import torch

            data = torch.load(path, map_location="cpu")
            return {
                key: value.numpy() if hasattr(value, "numpy") else np.asarray(value)
                for key, value in data.items()
            }
        raise ValueError(f"Unsupported precond_path extension: {path!r}")
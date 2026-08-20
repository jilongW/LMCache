"""Configuration and preconditioner support for the KVWeave codec."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
import os
import threading
from typing import Any, ClassVar, Optional

import numpy as np
import torch

from lmcache.logging import init_logger

logger = init_logger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    """Parse a conventional boolean environment flag."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_scaling_method(name: str, default: str) -> str:
    """Read and validate a KVWeave scaling-method environment variable."""
    value = os.environ.get(name, default)
    valid = {"per_tensor", "per_channel", "per_token"}
    if value not in valid:
        raise ValueError(f"{name}={value!r} is not one of {sorted(valid)}")
    return value


@dataclass(frozen=True)
class MambaCodecOptions:
    """Resolved per-substate quantization parameters for Mamba groups.

    ``conv``/``ssm`` scaling and randomized-Hadamard (``rh``) settings are
    independently configurable; ``asym`` is shared by both sub-states. See
    ``env_vars.md`` for the full environment variable reference.
    """

    conv_scaling_method: str
    conv_rh: bool
    ssm_scaling_method: str
    ssm_rh: bool
    asym: bool
    conv_qbit: int = 4
    ssm_qbit: int = 16
    conv_quant_enabled: bool = True
    ssm_quant_enabled: bool = True

    @classmethod
    def from_env(cls) -> "MambaCodecOptions":
        """Resolve Mamba conv/ssm quantization options from the environment.

        ``LMCACHE_MP_KVWEAVE_CONV_SCALING_METHOD`` and
        ``SSM_SCALING_METHOD`` have independent defaults
        (``per_channel``), not derived from ``LINEAR_*``. ``CONV_RH`` still
        falls back to ``LINEAR_RH`` while ``SSM_RH`` defaults to ``true``. A
        conv ``rh=True`` combined with ``per_tensor``/
        ``per_channel`` scaling disables RH while preserving the requested
        scaling method, because the transform length is not guaranteed to be
        a power of two.

        ``LMCACHE_MP_KVWEAVE_CONV_QUANT_ENABLED``/``SSM_QUANT_ENABLED``
        (DEBUG ONLY, default ``true``) independently disable
        quantization for one sub-state while leaving the other quantized --
        for isolating which sub-state's quantization causes an accuracy
        regression.
        """
        _env_scaling_method(
            "LMCACHE_MP_KVWEAVE_LINEAR_SCALING_METHOD", "per_tensor"
        )
        linear_rh = _env_flag("LMCACHE_MP_KVWEAVE_LINEAR_RH", False)
        conv_scaling = _env_scaling_method(
            "LMCACHE_MP_KVWEAVE_CONV_SCALING_METHOD", "per_channel"
        )
        conv_rh = _env_flag("LMCACHE_MP_KVWEAVE_CONV_RH", linear_rh)
        if conv_rh and conv_scaling in {"per_tensor", "per_channel"}:
            logger.info(
                "LMCACHE_MP_KVWEAVE_CONV_RH=1 requires scaling_method="
                "per_token; disabling RH while preserving "
                "scaling_method=%s",
                conv_scaling,
            )
            conv_rh = False
        return cls(
            conv_scaling_method=conv_scaling,
            conv_rh=conv_rh,
            ssm_scaling_method=_env_scaling_method(
                "LMCACHE_MP_KVWEAVE_SSM_SCALING_METHOD", "per_channel"
            ),
            ssm_rh=_env_flag("LMCACHE_MP_KVWEAVE_SSM_RH", True),
            asym=_env_flag("LMCACHE_MP_KVWEAVE_LINEAR_ASYM", True),
            conv_qbit=int(os.environ.get("LMCACHE_MP_KVWEAVE_CONV_QBIT", "4")),
            ssm_qbit=int(os.environ.get("LMCACHE_MP_KVWEAVE_SSM_QBIT", "16")),
            conv_quant_enabled=_env_flag(
                "LMCACHE_MP_KVWEAVE_CONV_QUANT_ENABLED", True
            ),
            ssm_quant_enabled=_env_flag(
                "LMCACHE_MP_KVWEAVE_SSM_QUANT_ENABLED", True
            ),
        )


@dataclass(frozen=True)
class KVWeaveRuntimeConfig:
    """Fully resolved L1 KVWeave quantization configuration for one worker.

    The single entry point for every ``LMCACHE_MP_L1_KVWEAVE_QUANT``/
    ``LMCACHE_MP_KVWEAVE_*`` environment variable (see ``env_vars.md``):
    callers should read the environment exactly once via :meth:`from_env`
    and thread the resolved values through, rather than reaching for
    ``os.environ`` themselves.
    """

    enabled: bool
    linear_quant_enabled: bool
    linear_max_size_ratio: float
    attention_codec_kwargs: dict[str, Any] = field(default_factory=dict)
    mamba_options: MambaCodecOptions = field(
        default_factory=lambda: MambaCodecOptions(
            conv_scaling_method="per_channel",
            conv_rh=False,
            ssm_scaling_method="per_channel",
            ssm_rh=True,
            asym=True,
            conv_qbit=4,
            ssm_qbit=16,
        )
    )

    @classmethod
    def from_env(cls) -> "KVWeaveRuntimeConfig":
        """Resolve the full L1 KVWeave runtime configuration from the environment.

        ``enabled`` (``LMCACHE_MP_L1_KVWEAVE_QUANT``) is the overall switch;
        ``linear_quant_enabled`` (``LMCACHE_MP_KVWEAVE_LINEAR_QUANT_ENABLED``,
        default ``true``) independently gates Mamba/linear groups under it.
        ``attention_codec_kwargs`` is ready to pass straight into
        ``KVWeaveCodec(...)``.
        """
        enabled = _env_flag("LMCACHE_MP_L1_KVWEAVE_QUANT", False)
        return cls(
            enabled=enabled,
            linear_quant_enabled=_env_flag(
                "LMCACHE_MP_KVWEAVE_LINEAR_QUANT_ENABLED", True
            ),
            linear_max_size_ratio=float(
                os.environ.get("LMCACHE_MP_KVWEAVE_LINEAR_MAX_SIZE_RATIO", "1.20")
            ),
            attention_codec_kwargs={
                "quantize": True,
                "qbit": 4,
                "num_kv_heads": int(
                    os.environ.get("LMCACHE_MP_KVWEAVE_NUM_KV_HEADS", "1")
                ),
                "head_dim": int(os.environ.get("LMCACHE_MP_KVWEAVE_HEAD_DIM", "0")),
                "scaling_method": os.environ.get(
                    "LMCACHE_MP_KVWEAVE_SCALING_METHOD", "per_channel"
                ),
                "precond": _env_flag("LMCACHE_MP_KVWEAVE_PRECOND", False),
            },
            mamba_options=MambaCodecOptions.from_env(),
        )


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
    MAGIC_RAW: ClassVar[bytes] = b"KVW0"
    MAGIC_QUANT: ClassVar[bytes] = b"KVW3"
    MAMBA_MAGIC: ClassVar[bytes] = b"MQ01"
    MAMBA_QBIT: ClassVar[int] = 4
    MAMBA_FLAG_RH: ClassVar[int] = 1
    MAMBA_FLAG_ASYM: ClassVar[int] = 2
    DEFAULT_BLOCK_SIZE: ClassVar[int] = 64
    DTYPE_TO_CODE: ClassVar[dict[torch.dtype, int]] = {
        torch.float16: 0,
        torch.bfloat16: 1,
        torch.float32: 2,
    }
    CODE_TO_DTYPE: ClassVar[dict[int, torch.dtype]] = {
        0: torch.float16,
        1: torch.bfloat16,
        2: torch.float32,
    }
    SCALING_TO_CODE: ClassVar[dict[str, int]] = {
        "per_tensor": 0,
        "per_token": 1,
        "per_channel": 2,
    }
    SUBSTATE_TO_CODE: ClassVar[dict[str, int]] = {"conv": 0, "ssm": 1}
    SCALE_IDS: ClassVar = count(1)

    def __post_init__(self) -> None:
        self._pd_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._pd_lock = threading.Lock()
        self._pd_file: Optional[dict[str, object]] = None
        self.scaling_method = str(self.scaling_method)

    @classmethod
    def next_scale_id(cls) -> int:
        """Allocate a process-wide scale id for native KVWeave state."""
        return next(cls.SCALE_IDS) & 0xFFFFFFFF

    @classmethod
    def next_scale_ids(cls, count: int) -> list[int]:
        return [cls.next_scale_id() for _ in range(count)]

    @staticmethod
    def quantized_bytes(elements: int, qbit: int) -> int:
        return (elements + 1) // 2 if qbit == 4 else elements if qbit <= 8 else elements * 2

    @classmethod
    def mamba_dtype(cls, dtype_str: str) -> torch.dtype:
        dtype = getattr(torch, dtype_str.removeprefix("torch."), None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"unsupported Mamba wire dtype: {dtype_str!r}")
        return dtype

    @staticmethod
    def mamba_layout(
        substate: str, shape: tuple[int, ...], scaling_method: str
    ) -> tuple[int, int, int, int]:
        blocks = max(int(shape[1]), 1)
        tail = shape[2:]
        head_dim = max(int(tail[-1]) if tail else 1, 1)
        middle = max(
            int(np.prod(tail[:-1])) if len(tail) > 1 else 1, 1
        )
        if scaling_method == "per_channel":
            return blocks, middle, head_dim, head_dim
        if scaling_method == "per_token" and substate == "conv":
            return blocks * middle, 1, head_dim, blocks * middle
        if scaling_method == "per_token":
            return blocks, middle, head_dim, blocks
        return blocks, middle, head_dim, 1

    def mamba_precond_tensors(
        self, size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if size <= 0 or size & (size - 1):
            raise ValueError(f"rh requires a power-of-2 transform length, got {size}")
        signs, perm = self.get_pd_matrix(size)
        return (
            torch.as_tensor(signs, dtype=torch.float32).contiguous(),
            torch.as_tensor(perm, dtype=torch.int32).contiguous(),
        )

    def get_pd_matrix(
        self, hadamard_size: int
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return cached deterministic or file-backed P/D matrices."""
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
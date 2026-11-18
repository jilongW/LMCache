# SPDX-License-Identifier: Apache-2.0
"""Configuration and preconditioner support for the KVWeave L2 serde.

Everything here is resolved from the serde's JSON kwargs (the ``"serde"`` dict
of an L2 adapter spec); nothing reads environment variables or a model
``config.json``.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
from enum import Enum
from itertools import count
from typing import Any, ClassVar, Mapping, Optional
import threading

# Third Party
import numpy as np
import torch

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


class AttentionPlaneLayout(Enum):
    """How one attention group's KV object plane(s) are physically laid out.

    ``SPLIT_KV``: the standard two-plane layout (K and V as separate
    tensors/axis), quantized via the existing K/V-paired codec path.
    ``FUSED_KV``: K and V packed into one tensor's trailing content axis
    (e.g. vLLM's non-MLA blocks-first fused backends) -- still per-head K/V
    data, just packed; quantized via the single-tensor fused codec path.
    ``MLA``: a true compressed latent vector (Multi-head Latent Attention)
    with no per-head K/V structure for the codec to exploit -- never
    quantized.
    ``UNKNOWN``: single-plane (``kv_size == 1``) but the engine KV format
    could not be classified -- never quantized (safe default, no guessing).
    """

    SPLIT_KV = "split_kv"
    FUSED_KV = "fused_kv"
    MLA = "mla"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ConvQKVSplit:
    """Widths of the Q/K/V segments packed in a Mamba ``conv_state`` row.

    The row is ``[query(key_dim) | key(key_dim) | value(value_dim)]``.
    """

    key_dim: int
    value_dim: int


@dataclass(frozen=True)
class MambaCodecOptions:
    """Resolved per-substate quantization parameters for Mamba groups.

    ``conv``/``ssm`` scaling and randomized-Hadamard (``rh``) settings are
    independently configurable; ``asym`` is shared by both sub-states.
    """

    conv_scaling_method: str
    conv_rh: bool
    ssm_scaling_method: str
    ssm_rh: bool
    asym: bool
    conv_qbit: int = 4
    ssm_qbit: int = 4
    conv_quant_enabled: bool = True
    ssm_quant_enabled: bool = True
    ssm_direct_quant_enabled: bool = True
    conv_direct_quant_enabled: bool = True
    conv_qkv_split: ConvQKVSplit = field(
        default_factory=lambda: ConvQKVSplit(key_dim=0, value_dim=0)
    )
    """Filled per group from the engine-reported conv channel segments;
    never read from user configuration."""

    @classmethod
    def from_kwargs(cls, kwargs: Mapping[str, Any]) -> "MambaCodecOptions":
        """Build the Mamba conv/ssm options from the serde's JSON kwargs.

        Recognized keys (all optional): ``conv_scaling_method``,
        ``ssm_scaling_method`` (default ``per_token``), ``conv_rh``,
        ``ssm_rh`` (default ``False``), ``mamba_asym`` (default ``False``),
        ``conv_qbit``, ``ssm_qbit`` (default 4), ``conv_quant_enabled``,
        ``ssm_quant_enabled``, ``ssm_direct_quant``, ``conv_direct_quant``
        (default ``True``). The conv Q/K/V split is not configurable: it is
        reported per group by the engine.
        """
        valid = {"per_tensor", "per_channel", "per_token"}
        conv_scaling = str(kwargs.get("conv_scaling_method", "per_token"))
        ssm_scaling = str(kwargs.get("ssm_scaling_method", "per_token"))
        for name, value in (
            ("conv_scaling_method", conv_scaling),
            ("ssm_scaling_method", ssm_scaling),
        ):
            if value not in valid:
                raise ValueError(f"{name}={value!r} is not one of {sorted(valid)}")
        return cls(
            conv_scaling_method=conv_scaling,
            conv_rh=bool(kwargs.get("conv_rh", False)),
            ssm_scaling_method=ssm_scaling,
            ssm_rh=bool(kwargs.get("ssm_rh", False)),
            asym=bool(kwargs.get("mamba_asym", False)),
            conv_qbit=int(kwargs.get("conv_qbit", 4)),
            ssm_qbit=int(kwargs.get("ssm_qbit", 4)),
            conv_quant_enabled=bool(kwargs.get("conv_quant_enabled", True)),
            ssm_quant_enabled=bool(kwargs.get("ssm_quant_enabled", True)),
            ssm_direct_quant_enabled=bool(kwargs.get("ssm_direct_quant", True)),
            conv_direct_quant_enabled=bool(kwargs.get("conv_direct_quant", True)),
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
    num_threads: int = 8
    precond_seed: int = 42
    precond_path: Optional[str] = None
    MAGIC_RAW: ClassVar[bytes] = b"KVW0"
    MAGIC_QUANT: ClassVar[bytes] = b"KVW3"
    MAGIC_QUANT_FUSED: ClassVar[bytes] = b"KVW4"
    MAMBA_MAGIC: ClassVar[bytes] = b"MQ01"
    MAMBA_QBIT: ClassVar[int] = 4
    MAMBA_FLAG_RH: ClassVar[int] = 1
    MAMBA_FLAG_ASYM: ClassVar[int] = 2
    DEFAULT_BLOCK_SIZE: ClassVar[int] = 64
    DTYPE_TO_CODE: ClassVar[dict[torch.dtype, int]] = {
        torch.float16: 0,
        torch.bfloat16: 1,
        torch.float32: 2,
        # fp8 KV cache (vLLM --kv-cache-dtype fp8); appended so existing payload
        # codes are unchanged.
        torch.float8_e4m3fn: 3,
        torch.float8_e5m2: 4,
    }
    CODE_TO_DTYPE: ClassVar[dict[int, torch.dtype]] = {
        0: torch.float16,
        1: torch.bfloat16,
        2: torch.float32,
        3: torch.float8_e4m3fn,
        4: torch.float8_e5m2,
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
        return (
            (elements + 1) // 2
            if qbit == 4
            else elements
            if qbit <= 8
            else elements * 2
        )

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
        middle = max(int(np.prod(tail[:-1])) if len(tail) > 1 else 1, 1)
        if scaling_method == "per_channel":
            return blocks, middle, head_dim, head_dim
        if scaling_method == "per_token":
            # Token-wise grouping uses one scale per logical token slice
            # across the non-head_dim axes (blocks * middle).
            return blocks * middle, 1, head_dim, blocks * middle
        return blocks, middle, head_dim, 1

    def mamba_precond_tensors(self, size: int) -> tuple[torch.Tensor, torch.Tensor]:
        if size <= 0 or size & (size - 1):
            raise ValueError(f"rh requires a power-of-2 transform length, got {size}")
        signs, perm = self.get_pd_matrix(size)
        return (
            torch.as_tensor(signs, dtype=torch.float32).contiguous(),
            torch.as_tensor(perm, dtype=torch.int32).contiguous(),
        )

    def get_pd_matrix(self, hadamard_size: int) -> tuple[np.ndarray, np.ndarray]:
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

    def _load_pd_from_file(self, hadamard_size: int) -> tuple[np.ndarray, np.ndarray]:
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
            # Third Party
            import torch

            data = torch.load(path, map_location="cpu")
            return {
                key: value.numpy() if hasattr(value, "numpy") else np.asarray(value)
                for key, value in data.items()
            }
        raise ValueError(f"Unsupported precond_path extension: {path!r}")

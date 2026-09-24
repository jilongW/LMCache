"""Configuration and preconditioner support for the KVWeave codec."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import count
import json
import os
import threading
from typing import Any, ClassVar, Optional

import numpy as np
import torch

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

# Fallback model text-config, used when MODEL_PATH/MODEL are unset or the
# model's config.json is missing the fields KVWeave needs. Mirrors
# Qwen3.5-9B's known text_config geometry.
_QWEN35_9B_DEFAULTS: dict[str, int] = {
    "num_key_value_heads": 4,
    "head_dim": 256,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_value_head_dim": 128,
    "linear_num_value_heads": 32,
}


def _load_model_text_config() -> dict[str, int]:
    """Resolve KV/Mamba geometry from ``{MODEL_PATH}/{MODEL}/config.json``.

    Reads the ``text_config`` section (falling back to the top-level object
    for models that don't nest their text config) and returns
    ``num_key_value_heads``/``head_dim``/``linear_*_head_dim``/
    ``linear_num_*_heads``. Any field missing from the environment, the
    model directory, or the model's config falls back to the Qwen3.5-9B
    defaults above.
    """
    result = dict(_QWEN35_9B_DEFAULTS)
    model_path = os.environ.get("MODEL_PATH")
    model = os.environ.get("MODEL")
    if not model_path or not model:
        return result
    config_path = os.path.join(model_path, model, "config.json")
    try:
        with open(config_path) as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(
            "Failed to read model config at %s (%s); falling back to "
            "Qwen3.5-9B KVWeave defaults",
            config_path,
            e,
        )
        return result
    text_config = config.get("text_config", config)
    for key in result:
        if key in text_config:
            result[key] = int(text_config[key])
    return result


@dataclass(frozen=True)
class ConvQKVSplit:
    """Conv-state Q/K/V split widths used by newer codec variants.

    This branch may not actively use the split in runtime code yet, but the
    option is part of the public config surface expected by tests.
    """

    key_dim: int
    value_dim: int


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


def _env_attention_qbit() -> int:
    qbit = int(os.environ.get("LMCACHE_MP_KVWEAVE_QBIT", "4"))
    if qbit not in {4, 8}:
        raise ValueError(
            f"LMCACHE_MP_KVWEAVE_QBIT={qbit!r} is not one of [4, 8]"
        )
    return qbit


def _env_device(name: str = "LMCACHE_MP_KVWEAVE_QUANT_DEVICE") -> str:
    """Read and validate the KVWeave quant/dequant compute device."""
    value = os.environ.get(name, "cpu").strip().lower()
    valid = {"cpu", "xpu"}
    if value not in valid:
        raise ValueError(f"{name}={value!r} is not one of {sorted(valid)}")
    return value


def _env_num_threads() -> int:
    """Read the native KVWeave quant kernel's OpenMP thread count.

    Previously never wired up despite ``LMCACHE_MP_KVWEAVE_NUM_THREADS``
    being referenced by deployment scripts -- the codec always ran with
    ``num_threads=1`` regardless of this env var.
    """
    num_threads = int(os.environ.get("LMCACHE_MP_KVWEAVE_NUM_THREADS", "4"))
    if num_threads < 1:
        raise ValueError(
            f"LMCACHE_MP_KVWEAVE_NUM_THREADS={num_threads!r} must be >= 1"
        )
    return num_threads


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
    ssm_qbit: int = 4
    conv_quant_enabled: bool = True
    ssm_quant_enabled: bool = True
    conv_qkv_split: ConvQKVSplit = field(
        default_factory=lambda: ConvQKVSplit(key_dim=2048, value_dim=4096)
    )

    @classmethod
    def from_env(cls) -> "MambaCodecOptions":
        """Resolve Mamba conv/ssm quantization options from the environment.

        ``LMCACHE_MP_KVWEAVE_CONV_SCALING_METHOD`` and
        ``SSM_SCALING_METHOD`` both default to ``per_token``, independently
        of ``LINEAR_*``.
        ``CONV_RH`` still
        falls back to ``LINEAR_RH`` (now defaulting to ``true``) while
        ``SSM_RH`` defaults to ``true`` independently.

        ``LMCACHE_MP_KVWEAVE_CONV_QUANT_ENABLED``/``SSM_QUANT_ENABLED``
        (DEBUG ONLY, default ``true``) independently disable
        quantization for one sub-state while leaving the other quantized --
        for isolating which sub-state's quantization causes an accuracy
        regression.

        With the default ``conv_scaling_method="per_token"``, RH operates
        across each Q/K/V segment's last dimension. It remains enabled when
        those widths are powers of 2, as they are for Qwen3.5-9B.
        """
        _env_scaling_method(
            "LMCACHE_MP_KVWEAVE_LINEAR_SCALING_METHOD", "per_channel"
        )
        linear_rh = _env_flag("LMCACHE_MP_KVWEAVE_LINEAR_RH", True)
        conv_scaling = _env_scaling_method(
            "LMCACHE_MP_KVWEAVE_CONV_SCALING_METHOD", "per_token"
        )
        conv_rh = _env_flag("LMCACHE_MP_KVWEAVE_CONV_RH", linear_rh)
        text_config = _load_model_text_config()
        conv_qkv_split = ConvQKVSplit(
            key_dim=text_config["linear_key_head_dim"]
            * text_config["linear_num_key_heads"],
            value_dim=text_config["linear_value_head_dim"]
            * text_config["linear_num_value_heads"],
        )
        return cls(
            conv_scaling_method=conv_scaling,
            conv_rh=conv_rh,
            ssm_scaling_method=_env_scaling_method(
                "LMCACHE_MP_KVWEAVE_SSM_SCALING_METHOD", "per_token"
            ),
            ssm_rh=_env_flag("LMCACHE_MP_KVWEAVE_SSM_RH", True),
            asym=_env_flag("LMCACHE_MP_KVWEAVE_LINEAR_ASYM", True),
            conv_qbit=int(os.environ.get("LMCACHE_MP_KVWEAVE_CONV_QBIT", "4")),
            ssm_qbit=int(os.environ.get("LMCACHE_MP_KVWEAVE_SSM_QBIT", "4")),
            conv_quant_enabled=_env_flag(
                "LMCACHE_MP_KVWEAVE_CONV_QUANT_ENABLED", True
            ),
            ssm_quant_enabled=_env_flag(
                "LMCACHE_MP_KVWEAVE_SSM_QUANT_ENABLED", True
            ),
            conv_qkv_split=conv_qkv_split,
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
    split_attention_quant_enabled: bool = True
    fused_attention_quant_enabled: bool = True
    attention_codec_kwargs: dict[str, Any] = field(default_factory=dict)
    mamba_options: MambaCodecOptions = field(
        default_factory=lambda: MambaCodecOptions(
            conv_scaling_method="per_token",
            conv_rh=False,
            ssm_scaling_method="per_token",
            ssm_rh=True,
            asym=True,
            conv_qbit=4,
            ssm_qbit=4,
        )
    )

    @classmethod
    def from_env(cls) -> "KVWeaveRuntimeConfig":
        """Resolve the full L1 KVWeave runtime configuration from the environment.

        ``enabled`` (``LMCACHE_MP_L1_KVWEAVE_QUANT``) is the overall switch;
        ``linear_quant_enabled`` (``LMCACHE_MP_KVWEAVE_LINEAR_QUANT_ENABLED``,
        default ``true``) independently gates Mamba/linear groups under it.
        ``attention_codec_kwargs`` is ready to pass straight into
        ``KVWeaveCodec(...)``. ``rh``/``asym`` were previously only settable
        by constructing ``_KVWeaveCodec`` directly (its own default is
        ``True`` for both); ``LMCACHE_MP_KVWEAVE_RH``/``LMCACHE_MP_KVWEAVE_ASYM``
        expose that same default as an environment override.
        """
        enabled = _env_flag("LMCACHE_MP_L1_KVWEAVE_QUANT", False)
        text_config = _load_model_text_config()
        return cls(
            enabled=enabled,
            linear_quant_enabled=_env_flag(
                "LMCACHE_MP_KVWEAVE_LINEAR_QUANT_ENABLED", True
            ),
            linear_max_size_ratio=float(
                os.environ.get("LMCACHE_MP_KVWEAVE_LINEAR_MAX_SIZE_RATIO", "1.20")
            ),
            split_attention_quant_enabled=_env_flag(
                "LMCACHE_MP_KVWEAVE_SPLIT_ATTENTION_QUANT_ENABLED", True
            ),
            fused_attention_quant_enabled=_env_flag(
                "LMCACHE_MP_KVWEAVE_FUSED_ATTENTION_QUANT_ENABLED", True
            ),
            attention_codec_kwargs={
                "quantize": True,
                "qbit": _env_attention_qbit(),
                "num_kv_heads": text_config["num_key_value_heads"],
                "head_dim": text_config["head_dim"],
                "scaling_method": os.environ.get(
                    "LMCACHE_MP_KVWEAVE_SCALING_METHOD", "per_token"
                ),
                "rh": _env_flag("LMCACHE_MP_KVWEAVE_RH", True),
                "asym": _env_flag("LMCACHE_MP_KVWEAVE_ASYM", True),
                "precond": _env_flag("LMCACHE_MP_KVWEAVE_PRECOND", True),
                "device": _env_device(),
                "num_threads": _env_num_threads(),
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
        if scaling_method == "per_token":
            # Token-wise grouping uses one scale per logical token slice
            # across the non-head_dim axes (blocks * middle).
            return blocks * middle, 1, head_dim, blocks * middle
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
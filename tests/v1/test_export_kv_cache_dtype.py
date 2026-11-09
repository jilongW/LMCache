# SPDX-License-Identifier: Apache-2.0
"""Exporting vLLM's fp8 ``--kv-cache-dtype`` for the KVWeave worker config."""

# Standard
from types import SimpleNamespace
import os

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm.utils import export_kv_cache_dtype
from lmcache.v1.distributed.serde.kvweave.kvweave_config import _env_kv_fp8_dtype

ENV = "LMCACHE_MP_KVWEAVE_KV_CACHE_DTYPE"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)


def _config(cache_dtype: str) -> SimpleNamespace:
    return SimpleNamespace(cache_config=SimpleNamespace(cache_dtype=cache_dtype))


@pytest.mark.parametrize(
    "cache_dtype, expected",
    [
        ("fp8", torch.float8_e4m3fn),
        ("fp8_e4m3", torch.float8_e4m3fn),
        ("fp8_e5m2", torch.float8_e5m2),
    ],
)
def test_fp8_formats_are_exported_and_understood_by_the_config(cache_dtype, expected):
    export_kv_cache_dtype(_config(cache_dtype))

    # The exported string must be one the worker config can resolve.
    assert _env_kv_fp8_dtype() == expected


@pytest.mark.parametrize(
    "cache_dtype",
    ["auto", "float16", "bfloat16", "fp8_inc", "fp8_ds_mla", "turboquant_4bit_nc"],
)
def test_other_cache_dtypes_are_not_exported(cache_dtype):
    """Other ``fp8*`` values are uint8 in layouts of their own; reinterpreting
    them as e4m3 would corrupt the cache."""
    export_kv_cache_dtype(_config(cache_dtype))

    assert ENV not in os.environ
    assert _env_kv_fp8_dtype() is None


def test_an_explicit_setting_is_never_overridden(monkeypatch):
    monkeypatch.setenv(ENV, "fp8_e5m2")

    export_kv_cache_dtype(_config("fp8"))

    assert _env_kv_fp8_dtype() == torch.float8_e5m2


def test_missing_vllm_config_is_ignored():
    broken = SimpleNamespace()  # no ``cache_config``

    export_kv_cache_dtype(broken)

    assert _env_kv_fp8_dtype() is None

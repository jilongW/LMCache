# SPDX-License-Identifier: Apache-2.0
"""Engine-neutral facts about what an object group's chunks contain.

A serde that transforms KV data (e.g. quantization) must know what it is
transforming: attention K/V or recurrent Mamba state, how K and V are packed,
the head geometry, and the Mamba sub-state layout. The engine registration
publishes those facts here; each serde reads them and decides for itself what
to do, so the registration code stays serde-agnostic.

An object group's chunk is one memory object holding one tensor per kernel
group (a *part*), so facts are kept per ``(object_group_id, part)``.
"""

# Standard
from typing import Mapping, Sequence
import threading

# First Party
from lmcache.v1.common.group_view import SerdePartInfo


class SerdeGroupInfoRegistry:
    """Thread-safe map ``(model_name, object_group_id) -> per-part facts``.

    Facts registered under model ``None`` apply to every model.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._parts: dict[tuple[str | None, int], tuple[SerdePartInfo, ...]] = {}

    def register(
        self,
        model_name: str | None,
        object_group_id: int,
        parts: Sequence[SerdePartInfo],
    ) -> None:
        """Register (or replace) one object group's per-part facts."""
        with self._lock:
            self._parts[(model_name, object_group_id)] = tuple(parts)

    def register_all(
        self,
        model_name: str | None,
        object_groups: Mapping[int, Sequence[SerdePartInfo]],
    ) -> None:
        """Register every object group of one engine registration."""
        for object_group_id, parts in object_groups.items():
            self.register(model_name, object_group_id, parts)

    def lookup(
        self, model_name: str, object_group_id: int, part: int
    ) -> SerdePartInfo | None:
        """Return one part's facts, model-specific first, else model-agnostic.

        Returns:
            The facts, or ``None`` when the group or part is unknown.
        """
        with self._lock:
            parts = self._parts.get((model_name, object_group_id)) or self._parts.get(
                (None, object_group_id)
            )
        if parts is None or part >= len(parts):
            return None
        return parts[part]

    def clear(self) -> None:
        """Drop everything (tests / reconfiguration)."""
        with self._lock:
            self._parts.clear()


__all__ = [
    "SerdeGroupInfoRegistry",
    "SerdePartInfo",
    "get_serde_group_info_registry",
]

_REGISTRY = SerdeGroupInfoRegistry()


def get_serde_group_info_registry() -> SerdeGroupInfoRegistry:
    """Return the process-wide registry shared by the engine and serdes."""
    return _REGISTRY

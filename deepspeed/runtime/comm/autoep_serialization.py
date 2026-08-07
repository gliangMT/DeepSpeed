# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

"""Optional process-local serialization for AutoEP communication.

The sequencer preserves the streams selected by each subsystem, but completes
each collective before returning to the caller.  Completion requires waiting
on the communication work handle when the backend returns one; synchronizing a
caller-stream event alone does not necessarily cover work queued by MCCL on an
internal stream.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager

from deepspeed.accelerator import get_accelerator
from deepspeed.utils import logger


class _AutoEPCommunicationSequencer:

    def __init__(self) -> None:
        self._enabled = False
        self._lock = threading.Lock()
        self._local = threading.local()
        self._last_event = None
        self._epoch = 0

    def enable(self) -> None:
        if not self._enabled:
            logger.info("AutoEP communication serialization enabled: multiple streams are preserved, "
                        "but every collective completes before the caller continues.")
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    @contextmanager
    def serialize(self, label: str):
        if not self._enabled:
            yield None
            return

        # Selected call sites wrap a larger operation such as ZeRO gradient
        # partitioning, while the DeepSpeed communication facade wraps the
        # collective itself.  Treat the inner scope as part of the outer one
        # so the process-global lock is safe for nested instrumentation.
        depth = getattr(self._local, "depth", 0)
        if depth:
            self._local.depth = depth + 1
            try:
                yield None
            finally:
                self._local.depth = depth
            return

        with self._lock:
            self._local.depth = 1
            if self._last_event is not None:
                self._last_event.synchronize()

            epoch = self._epoch
            try:
                yield epoch
            except Exception:
                self._last_event = None
                raise
            else:
                if not get_accelerator().is_synchronized_device():
                    event = get_accelerator().Event()
                    event.record()
                    event.synchronize()
                    self._last_event = None
                else:
                    self._last_event = None
                self._epoch += 1
                logger.debug("AutoEP serialized communication epoch=%s label=%s", epoch, label)
            finally:
                self._local.depth = 0

    def reset_for_testing(self) -> None:
        with self._lock:
            self._enabled = False
            self._last_event = None
            self._epoch = 0
            self._local.depth = 0


_SEQUENCER = _AutoEPCommunicationSequencer()


def enable_autoep_communication_serialization() -> None:
    _SEQUENCER.enable()


def autoep_communication_serialization_enabled() -> bool:
    return _SEQUENCER.enabled


def complete_autoep_communication(result) -> None:
    """Wait for backend work returned by a collective while strict mode is on.

    DeepSpeed's communication facade returns either ``None`` for a blocking
    collective or a Work-like object for ``async_op=True``.  Some backends use
    an internal communication stream, so an event recorded on the caller's
    stream is not a substitute for ``Work.wait()``.  Preserve the original
    result for API compatibility; callers may safely observe/wait on it again.
    """
    if not _SEQUENCER.enabled or result is None:
        return

    if isinstance(result, (list, tuple)):
        for item in result:
            complete_autoep_communication(item)
        return

    wait = getattr(result, "wait", None)
    if callable(wait):
        wait()


def serialized_autoep_communication(label: str):
    return _SEQUENCER.serialize(label)


def reset_autoep_communication_serialization_for_testing() -> None:
    _SEQUENCER.reset_for_testing()

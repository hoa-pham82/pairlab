"""Deterministic heap-based event queue ordered by (ts, priority, sequence)."""

from __future__ import annotations

import heapq
from itertools import count
from typing import Iterator

from pairlab.events import Event


class EventQueue:
    """Min-heap queue; events with the same timestamp are ordered by priority then arrival order."""

    def __init__(self) -> None:
        self._heap: list[tuple] = []
        self._counter = count()

    def push(self, event: Event) -> None:
        """Add an event; ties broken by priority then insertion order."""
        seq = next(self._counter)
        heapq.heappush(self._heap, (event.ts, event.priority, seq, event))

    def pop(self) -> Event:
        """Remove and return the next event. Raises IndexError if empty."""
        _, _, _, event = heapq.heappop(self._heap)
        return event

    def peek_ts(self) -> object:
        """Return the timestamp of the next event without removing it."""
        return self._heap[0][0] if self._heap else None

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)

    def __iter__(self) -> Iterator[Event]:
        """Drain all events in order. Destructive."""
        while self._heap:
            yield self.pop()

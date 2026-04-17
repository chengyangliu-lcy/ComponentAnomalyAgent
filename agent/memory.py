from __future__ import annotations

from schemas import Evidence, ToolEvent


class ContextMemory:
    def __init__(self) -> None:
        self.evidence: list[Evidence] = []
        self.events: list[ToolEvent] = []

    def add_evidence(self, items: list[Evidence]) -> None:
        self.evidence.extend(items)

    def add_events(self, items: list[ToolEvent]) -> None:
        self.events.extend(items)


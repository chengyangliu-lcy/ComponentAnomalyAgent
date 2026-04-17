from __future__ import annotations

from schemas import ToolEvent


class TraceLogger:
    def __init__(self) -> None:
        self.events: list[ToolEvent] = []

    def add(self, event: ToolEvent) -> None:
        self.events.append(event)

    def tool_names(self) -> list[str]:
        return sorted({event.tool_name for event in self.events})


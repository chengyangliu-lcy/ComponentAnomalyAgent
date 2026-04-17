from __future__ import annotations

from pathlib import Path

from tools.utils import compact_text


class FileReader:
    def read_text(self, path: Path, max_chars: int = 6000) -> str:
        text = path.read_text(encoding="utf-8", errors="replace")
        return compact_text(text, max_chars=max_chars)


from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")


def read_sample_ids_file(path: str | Path | None) -> list[str]:
    if not path:
        return []
    sample_ids: list[str] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            sample_ids.append(value)
    return list(dict.fromkeys(sample_ids))


def filter_items_by_sample_ids(
    items: Iterable[T],
    sample_ids: list[str],
    key_fn: Callable[[T], str],
) -> list[T]:
    if not sample_ids:
        return list(items)
    by_id = {key_fn(item): item for item in items}
    return [by_id[sample_id] for sample_id in sample_ids if sample_id in by_id]

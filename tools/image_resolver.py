from __future__ import annotations

import base64
from functools import cached_property
from pathlib import Path
from typing import Dict, Optional


class ImageResolver:
    def __init__(self, image_root: Path) -> None:
        self.image_root = image_root

    @cached_property
    def post_dirs(self) -> Dict[str, Path]:
        if not self.image_root.exists():
            return {}
        return {path.name: path for path in self.image_root.glob("*/*") if path.is_dir()}

    def resolve(self, post_id: str, image_url: str) -> Optional[Path]:
        filename = Path(image_url).name
        post_dir = self.post_dirs.get(post_id)
        if not post_dir:
            return None
        direct = post_dir / "images" / filename
        if direct.exists():
            return direct
        matches = list(post_dir.glob(f"**/{filename}"))
        return matches[0] if matches else None

    def to_data_url(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
        elif suffix == ".gif":
            mime = "image/gif"
        elif suffix == ".webp":
            mime = "image/webp"
        else:
            mime = "image/png"
        payload = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{payload}"


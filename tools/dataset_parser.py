from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from schemas import ImageRef, StandardSample
from tools.image_resolver import ImageResolver
from tools.utils import read_jsonl


class DatasetParser:
    def __init__(self, dataset_path: Path, image_root: Path) -> None:
        self.dataset_path = dataset_path
        self.image_resolver = ImageResolver(image_root)

    def load(self) -> List[StandardSample]:
        return [self.parse_row(row) for row in read_jsonl(self.dataset_path)]

    def iter_samples(self) -> Iterable[StandardSample]:
        for row in read_jsonl(self.dataset_path):
            yield self.parse_row(row)

    def parse_row(self, row: Dict[str, Any]) -> StandardSample:
        post_id = str(row["post_id"])
        messages = row.get("messages", [])
        question_parts: List[str] = []
        images: List[ImageRef] = []
        reference_answer = ""
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role == "assistant" and isinstance(content, str):
                reference_answer = content
            if role != "user":
                continue
            if isinstance(content, str):
                question_parts.append(content)
                continue
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        question_parts.append(item.get("text", ""))
                    elif item.get("type") == "image_url":
                        original_url = item.get("image_url", {}).get("url", "")
                        path = self.image_resolver.resolve(post_id, original_url)
                        images.append(ImageRef(original_url=original_url, path=path, exists=bool(path and path.exists())))
        return StandardSample(
            sample_id=post_id,
            post_id=post_id,
            question_text="\n".join(part.strip() for part in question_parts if part.strip()),
            images=images,
            reference_answer=reference_answer,
            raw_messages=messages,
            metadata={"image_count": len(images)},
        )

    def validate(self) -> Dict[str, Any]:
        samples = self.load()
        post_ids = [sample.post_id for sample in samples]
        image_refs = [image for sample in samples for image in sample.images]
        return {
            "total_samples": len(samples),
            "unique_post_ids": len(set(post_ids)),
            "duplicate_post_ids": sorted({pid for pid in post_ids if post_ids.count(pid) > 1}),
            "missing_user_question": [sample.post_id for sample in samples if not sample.question_text],
            "missing_reference_answer": [sample.post_id for sample in samples if not sample.reference_answer],
            "image_refs": len(image_refs),
            "image_refs_found": sum(1 for image in image_refs if image.exists),
            "missing_images": [image.to_json() for image in image_refs if not image.exists][:50],
        }


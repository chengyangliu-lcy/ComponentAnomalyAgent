from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any, Dict, List

from agent.prompts import build_final_answer_system_prompt, build_final_answer_user_prompt
from llm_client import LLMClient
from schemas import Evidence, StandardSample
from tools.utils import compact_text


class AnswerSynthesizer:
    def __init__(self, llm: LLMClient, send_images: bool = True, max_images: int = 4) -> None:
        self.llm = llm
        self.send_images = send_images
        self.max_images = max_images

    def synthesize(self, sample: StandardSample, evidence: List[Evidence]) -> tuple[str, str, Dict[str, Any], list[str]]:
        if self.llm.available:
            return self._llm_answer(sample, evidence)
        return self._fallback_answer(sample, evidence)

    def _llm_answer(self, sample: StandardSample, evidence: List[Evidence]) -> tuple[str, str, Dict[str, Any], list[str]]:
        evidence_text = "\n\n".join(
            f"[{idx}] {item.title}\n来源: {item.source}\n{compact_text(item.content, 1200)}"
            for idx, item in enumerate(evidence[:6], 1)
        )
        user_text = build_final_answer_user_prompt(
            sample.question_text,
            evidence_text,
            _question_hints(sample.question_text),
        )
        user_content: str | list[dict[str, Any]]
        if self.send_images:
            content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
            for image in sample.images[: self.max_images]:
                if image.path and image.exists:
                    data_url = self._image_data_url(image.path)
                    if data_url:
                        content.append({"type": "image_url", "image_url": {"url": data_url}})
            user_content = content
        else:
            user_content = user_text
        messages = [
            {
                "role": "system",
                "content": build_final_answer_system_prompt(),
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]
        response = self.llm.chat(messages)
        if response.content:
            summary = f"使用 {len(evidence)} 条证据生成答案。"
            return response.content, summary, response.token_usage, []
        return self._fallback_answer(sample, evidence, extra_error=response.error)

    def _fallback_answer(
        self,
        sample: StandardSample,
        evidence: List[Evidence],
        extra_error: str | None = None,
    ) -> tuple[str, str, Dict[str, Any], list[str]]:
        snippets = [compact_text(item.content, 600) for item in evidence[:3] if item.content]
        if snippets:
            answer = (
                "基于本地资料和检索证据，建议按以下思路分析：\n"
                "1. 先确认异常现象对应的组件、反馈链路、供电和布线环境，避免只从单个器件判断。\n"
                "2. 对照原帖/资料中的相同问题，优先检查噪声耦合、参数补偿、接地路径、滤波和器件连接方式。\n"
                "3. 结合图片中的实际连接位置复核关键节点，必要时用示波器验证纹波、尖峰、反馈脚或采样脚波形。\n"
                "4. 可尝试调整补偿/滤波参数、优化 PCB 布局和地线回路，并确认修改后稳定性和温升。\n\n"
                "参考证据摘要：\n- " + "\n- ".join(snippets)
            )
        else:
            answer = (
                "当前没有获得可用外部证据。建议先围绕问题中的关键组件做分层排查：确认连接和参数是否符合手册，"
                "检查供电、地线、反馈/采样链路、噪声耦合和 PCB 布局，再通过波形或日志验证异常触发条件。"
            )
        errors = [extra_error] if extra_error else []
        summary = "LLM 不可用或调用失败，使用规则化证据摘要生成答案。"
        return answer, summary, {}, errors

    def _image_data_url(self, path: Path) -> str | None:
        try:
            suffix = path.suffix.lower()
            if suffix in {".jpg", ".jpeg"}:
                mime = "image/jpeg"
            elif suffix == ".gif":
                mime = "image/gif"
            elif suffix == ".webp":
                mime = "image/webp"
            else:
                mime = "image/png"
            encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
            return f"data:{mime};base64,{encoded}"
        except Exception:
            return None


def _question_hints(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z]{1,8}\d{0,5}[A-Za-z0-9_.+-]*|\d+(?:\.\d+)?\s*[A-Za-zΩμ%]*|[\u4e00-\u9fff]{2,}", text or "")
    stop = {"请教", "问题", "为什么", "怎么", "处理", "以及", "这个", "电路", "作用", "哪些"}
    cleaned = [token.strip() for token in tokens if token.strip() and token.strip() not in stop]
    return list(dict.fromkeys(cleaned))[:12]

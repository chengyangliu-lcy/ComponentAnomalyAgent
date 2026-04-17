from __future__ import annotations

import re
from typing import Any, Dict, List

from schemas import AgentPlan, StandardSample


class Planner:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def plan(self, sample: StandardSample) -> AgentPlan:
        question = sample.question_text
        question_type = self._classify(question)
        queries = self._queries(question)
        needs_images = bool(sample.images) and bool(self.config.get("use_images", True))
        needs_local = bool(self.config.get("enable_local_retrieval", False))
        needs_web = bool(self.config.get("enable_web_search", False))
        steps = [
            "解析问题类型和关键信息缺口",
            "必要时读取图片并纳入上下文",
        ]
        if needs_local:
            steps.append("检索允许使用的外部知识库")
        if needs_web:
            steps.append("通过公开网页搜索补充外部证据")
        steps.extend(["反思证据是否覆盖原因、组件、参数和处理建议", "整合证据生成最终答案"])
        return AgentPlan(
            question_type=question_type,
            needs_images=needs_images,
            needs_local_retrieval=needs_local,
            needs_web_search=needs_web,
            queries=queries,
            steps=steps,
        )

    def _classify(self, question: str) -> str:
        buckets = [
            ("power_supply", ["电源", "开关", "反激", "LLC", "DCDC", "纹波", "TL431", "MOS"]),
            ("embedded_software", ["程序", "代码", "编译", "驱动", "I2C", "SPI", "UART", "GPIO"]),
            ("component_selection", ["选型", "型号", "参数", "电阻", "电容", "芯片", "传感器"]),
            ("debugging", ["异常", "故障", "不工作", "报错", "问题", "怎么处理", "为什么"]),
        ]
        upper_question = question.upper()
        for name, keys in buckets:
            if any(key.upper() in upper_question for key in keys):
                return name
        return "general_component_anomaly"

    def _queries(self, question: str) -> List[str]:
        normalized = re.sub(r"\s+", " ", question).strip()
        short = normalized[:120]
        keywords = re.findall(r"[A-Za-z0-9_.+-]+|[\u4e00-\u9fff]{2,}", normalized)
        keyword_query = " ".join(keywords[:12])
        queries = [q for q in [short, keyword_query] if q]
        return list(dict.fromkeys(queries))[:3]

    def _needs_web(self, question: str) -> bool:
        web_hints = ["datasheet", "手册", "官网", "型号", "规格", "寄存器", "报错", "版本"]
        return any(hint.lower() in question.lower() for hint in web_hints)

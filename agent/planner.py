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
        tokens = re.findall(r"[A-Za-z0-9_.+-]+|[\u4e00-\u9fff]{2,}", normalized)
        upper_tokens = {token.upper() for token in tokens}
        electronics_terms = [token for token in tokens if token.upper() in {"TL431", "MOS", "DCDC", "PWM", "PFC"}]
        electronics_terms.extend(
            token
            for token in tokens
            if re.fullmatch(r"[RCLDUQ]\d+[A-Za-z]?", token.upper())
            or re.fullmatch(r"[A-Z]{2,}\d+[A-Z0-9-]*", token.upper())
        )
        for term in ["输出电压", "纹波", "光耦", "反馈", "闭环", "补偿", "滤波", "噪声", "电源", "电路"]:
            if term in question:
                electronics_terms.append(term)
        for term in ["反激", "开关电源", "限流", "电流检测", "采样电阻", "电流采样", "尖峰", "低通滤波"]:
            if term in question:
                electronics_terms.append(term)
        if "LLC" in upper_tokens:
            electronics_terms.extend(["LLC谐振电源", "谐振", "开关电源"])
        core = " ".join(list(dict.fromkeys(electronics_terms))[:12])
        if not core:
            core = " ".join(tokens[:10])
        core_without_ambiguous_llc = " ".join(
            term for term in core.split() if term not in {"LLC", "LLC谐振电源", "谐振"}
        )
        queries = [
            f"{core_without_ambiguous_llc or core} 原因 处理 电子电路",
            f"{core} 原因 处理 开关电源",
        ]
        english_queries: list[str] = []
        if "TL431" in upper_tokens or "光耦" in question:
            english_queries.append("TL431 optocoupler feedback compensation power supply ripple")
        if any(term in question for term in ["反激", "限流", "电流检测", "电流采样", "采样电阻"]):
            english_queries.append("flyback current sense resistor RC filter leading edge spike blanking")
        if any(term in question for term in ["补偿", "反馈", "电阻", "电容"]):
            english_queries.append("power supply feedback loop compensation resistor capacitor")
        english_queries.append(f"{core} feedback compensation ripple power supply")
        queries.extend(english_queries)
        return list(dict.fromkeys(queries))[:3]

    def _needs_web(self, question: str) -> bool:
        web_hints = ["datasheet", "手册", "官网", "型号", "规格", "寄存器", "报错", "版本"]
        return any(hint.lower() in question.lower() for hint in web_hints)

from __future__ import annotations

from typing import List

from schemas import AgentPlan, Evidence, StandardSample


class Reflector:
    def assess(self, sample: StandardSample, plan: AgentPlan, evidence: List[Evidence]) -> tuple[bool, str, list[str]]:
        joined = "\n".join(item.content for item in evidence)
        missing: list[str] = []
        checks = {
            "组件/对象": any(token in joined for token in ["电阻", "电容", "芯片", "MOS", "TL431", "光耦", "组件"]),
            "异常原因": any(token in joined for token in ["原因", "导致", "由于", "干扰", "噪声", "补偿", "滤波"]),
            "处理建议": any(token in joined for token in ["建议", "处理", "检查", "调整", "增加", "减小", "布局"]),
        }
        for name, passed in checks.items():
            if not passed:
                missing.append(name)
        if not evidence:
            missing.append("外部或本地证据")
        enough = len(missing) == 0 or len(evidence) >= 2
        summary = "证据充分，可生成答案。" if enough else f"证据不足，缺少：{', '.join(missing)}。"
        return enough, summary, missing

    def supplemental_queries(self, sample: StandardSample, missing: list[str]) -> list[str]:
        base = sample.question_text[:120]
        return [f"{base} {' '.join(missing)} 处理 建议 原因"]


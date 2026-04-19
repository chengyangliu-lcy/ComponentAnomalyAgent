from __future__ import annotations

import unittest

from agent.prompts import (
    FINAL_ANSWER_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    VISION_SYSTEM_PROMPT,
    build_final_answer_user_prompt,
    build_judge_user_prompt,
    build_vision_user_prompt,
    planner_guidance,
)


class PromptContractTests(unittest.TestCase):
    def test_planner_prompt_preserves_tool_json_contract(self) -> None:
        text = PLANNER_SYSTEM_PROMPT + planner_guidance()

        for tool in [
            "inspect_image",
            "match_domain_skill",
            "web_search",
            "web_read",
            "rank_evidence",
            "review_evidence",
            "finish_answer",
        ]:
            self.assertIn(tool, text)
        self.assertIn("只返回一个合法 JSON 对象", text)
        self.assertIn("隐藏推理", text)
        self.assertIn("不要重复调用失败工具", text)
        self.assertIn("预算", text)
        self.assertIn("查询词必须包含题面中的元件", text)

    def test_vision_prompt_requires_detailed_description_and_no_guessing(self) -> None:
        text = VISION_SYSTEM_PROMPT + build_vision_user_prompt("R1 发热为什么")

        for section in ["整体画面", "元件与标注", "连接与拓扑", "测量与异常线索", "无法确认的信息"]:
            self.assertIn(section, text)
        self.assertIn("详细描述图片内容", text)
        self.assertIn("不要猜测精确型号", text)
        self.assertIn("不要给最终故障结论", text)
        self.assertIn("不要输出 JSON", text)

    def test_final_answer_prompt_requires_evidence_grounded_sections(self) -> None:
        text = FINAL_ANSWER_SYSTEM_PROMPT + build_final_answer_user_prompt(
            "R1 发热为什么",
            "[1] 图片证据\n来源: image_inspect\nR1 位于 MOS 栅极附近",
            ["R1", "MOS"],
        )

        for section in ["结论", "依据", "原因机制", "检查步骤", "处理建议", "不确定性"]:
            self.assertIn(section, text)
        self.assertIn("<题目>", text)
        self.assertIn("<证据>", text)
        self.assertIn("<题面线索>", text)
        self.assertIn("不得编造", text)
        self.assertIn("证据不足时说明缺口", text)

    def test_judge_prompt_preserves_required_json_fields_and_caps(self) -> None:
        text = JUDGE_SYSTEM_PROMPT + build_judge_user_prompt(
            "问题",
            "参考答案",
            "预测答案",
            {"covered": []},
        )

        for field in [
            "score",
            "accuracy",
            "completeness",
            "clarity",
            "usefulness",
            "average_score",
            "factual_consistency",
        ]:
            self.assertIn(field, text)
        self.assertIn("score <= 0.45", text)
        self.assertIn("factual_consistency <= 0.4", text)
        self.assertIn("completeness <= 2", text)
        self.assertIn("不要输出 Markdown", text)


if __name__ == "__main__":
    unittest.main()

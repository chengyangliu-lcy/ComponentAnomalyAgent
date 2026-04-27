from __future__ import annotations

import unittest

from evaluator.evaluate import Evaluator
from evaluator.jaccard_eval import technical_entity_match
from evaluator.rouge_eval import claim_rouge_l
from evaluator.scoring_points import judge_scoring_points


class EvaluationV2MetricTests(unittest.TestCase):
    def test_claim_rouge_scores_equivalent_rewrite_at_claim_level(self) -> None:
        reference = "结论：Q1发热主要是MOS管栅极振荡导致，需要增大栅极电阻并检查驱动回路。"
        prediction = "MOSFET门极有自激振荡，Q1因此发烫。处理上应把栅极串阻加大，同时复核驱动走线。"

        result = claim_rouge_l(reference, prediction)

        self.assertGreater(result.score, 0.0)
        self.assertTrue(result.claim_scores)

    def test_technical_entity_match_reports_unsupported_entities(self) -> None:
        reference = "检查R1和MOS管栅极电阻，避免振荡。"
        prediction = "检查R1、Q7和LM358，可能是运放饱和导致。"

        result = technical_entity_match(reference, prediction)

        self.assertGreater(result.score, 0.0)
        self.assertIn("lm358", result.unsupported_entities)
        self.assertGreater(result.unsupported_entity_rate, 0.0)

    def test_technical_entity_match_extracts_out_of_vocab_terms(self) -> None:
        reference = "肖特基二极管的ESR异常会造成温升异常，需要复核EMI滤波回路。"
        prediction = "肖特基管ESR偏大，EMI滤波部分可能导致过热。"

        result = technical_entity_match(reference, prediction)

        self.assertGreater(result.recall, 0.0)
        self.assertTrue(set(result.reference_entities) & set(result.prediction_entities))
        self.assertEqual(result.unsupported_entities, [])

    def test_technical_entity_match_does_not_reward_generic_terms_only(self) -> None:
        reference = "TL431反馈回路异常导致5V输出过压。"
        prediction = "建议检查，可能异常，需要进一步确认。"

        result = technical_entity_match(reference, prediction)

        self.assertLess(result.score, 0.2)

    def test_technical_entity_match_limits_long_prediction_noise(self) -> None:
        reference = "R9和C5组成RC低通滤波网络，用于抑制MOS导通瞬间的电流尖峰。"
        prediction = (
            "R9与C5是RC滤波，用来压低MOS开通尖峰。"
            "建议检查布局、焊接、温升、示波器探头、测试条件、负载变化、环境温度、记录数据、复核波形。"
        )

        result = technical_entity_match(reference, prediction)

        self.assertGreater(result.score, 0.2)
        self.assertLessEqual(len(result.prediction_entities), 112)

    def test_question_supported_entities_are_not_unsupported(self) -> None:
        reference = "输出纹波与反馈回路和高频噪声耦合有关。"
        prediction = "可以检查题面中的C24、R65和100nF补偿电容。"
        support_text = "原理图里C24和R65位于反馈支路，C24标称100nF。"

        result = technical_entity_match(reference, prediction, support_text=support_text)

        self.assertNotIn("c24", result.unsupported_entities)
        self.assertNotIn("r65", result.unsupported_entities)
        self.assertNotIn("100nf", result.unsupported_entities)

    def test_empty_reference_points_do_not_default_to_full_coverage(self) -> None:
        result = judge_scoring_points("", "泛泛回答")

        self.assertIsNone(result.coverage)

    def test_scoring_points_reward_semantic_rewrite_without_fixed_cue_words(self) -> None:
        reference = "温升异常与MOS管振荡有关。"
        prediction = "MOS管振荡导致过热和发烫。"

        result = judge_scoring_points(reference, prediction).to_json()

        self.assertGreaterEqual(result["coverage"], 0.5)
        self.assertTrue(result["match_evidence"])

    def test_scoring_points_extract_out_of_vocab_reference_point(self) -> None:
        reference = "肖特基二极管反向恢复异常会造成温升异常，需要复核ESR和EMI滤波回路。"
        prediction = "肖特基二极管过热，ESR偏大，同时检查EMI滤波。"

        result = judge_scoring_points(reference, prediction).to_json()

        self.assertGreaterEqual(result["coverage"], 0.5)
        self.assertTrue(result["structured_points"])
        self.assertIn("point_type_confidence", result)

    def test_scoring_points_do_not_reward_generic_answer(self) -> None:
        reference = "TL431反馈回路异常导致5V输出过压。"
        prediction = "建议检查电路，可能存在异常，需要进一步确认。"

        result = judge_scoring_points(reference, prediction).to_json()

        self.assertLess(result["coverage"], 0.5)

    def test_core_contradiction_blocks_fully_correct_and_caps_v2_score(self) -> None:
        scoring_points = judge_scoring_points("结论：上电瞬间上管先导通。", "结论：上电瞬间下管先导通。").to_json()
        evaluator = Evaluator.__new__(Evaluator)
        evaluator.weights = {}

        score = evaluator._final_score(
            {"enabled": True, "score": 1.0},
            semantic=1.0,
            claim_rouge=1.0,
            technical_entity=1.0,
            scoring_points=scoring_points,
        )

        self.assertFalse(evaluator._fully_correct({"enabled": True, "score": 1.0}, scoring_points))
        self.assertLessEqual(score, 0.45)

    def test_final_score_reweights_when_judge_is_disabled(self) -> None:
        evaluator = Evaluator.__new__(Evaluator)
        evaluator.weights = {}

        score = evaluator._final_score(
            {"enabled": False, "score": 0.0},
            semantic=0.8,
            claim_rouge=0.4,
            technical_entity=0.2,
            scoring_points={"coverage": 0.6, "matches": []},
        )

        expected = (0.25 * 0.6 + 0.20 * 0.8 + 0.05 * 0.4 + 0.05 * 0.2) / 0.55
        self.assertAlmostEqual(score, expected, places=6)


if __name__ == "__main__":
    unittest.main()

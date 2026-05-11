# 实验结果汇总

## 综合评分

| 实验 | 配置要点 | final_score | llm_judge | semantic | coverage | 状态 |
|------|----------|-------------|-----------|----------|----------|------|
| exp19_v3_max2_full | v3 KB + max_calls=2 + web_search | 0.5801 | 0.6729 | 0.7668 | 0.4009 | done (368) |
| exp20_qwen_search | 纯qwen_search (禁用web_search+本地KB) + finish_answer 4000tokens + 900s超时 | 0.5846 | 0.6758 | 0.7668 | 0.4115 | done (368) |

## LLM Judge 详细指标

| 实验 | judge_score | accuracy | completeness | factual_consistency | usefulness | clarity |
|------|-------------|----------|--------------|---------------------|------------|---------|
| exp19 | 0.6729 | 3.4132 | 3.5289 | 0.7098 | 3.9394 | 4.5289 |
| exp20 | 0.6758 | 3.4266 | 3.5435 | 0.7098 | 3.9647 | 4.5380 |

## 质量指标

| 实验 | claim_rouge_l | entity_match | entity_precision | entity_recall | entity_f_beta | unsupported_entity_rate | scoring_point_coverage | required_point_coverage |
|------|---------------|--------------|------------------|---------------|---------------|------------------------|------------------------|-------------------------|
| exp19 | 0.2600 | 0.3418 | 0.8304 | 0.2775 | 0.3418 | 0.5829 | 0.4009 | 0.4183 |
| exp20 | 0.2593 | 0.3488 | 0.8261 | 0.2841 | 0.3488 | 0.5767 | 0.4115 | 0.4251 |

## 质量通过率

| 实验 | fully_correct_rate | critical_error_rate | core_conclusion_hit_rate |
|------|--------------------|---------------------|--------------------------|
| exp19 | 0.54% | 40.49% | 70.31% |
| exp20 | 1.09% | 41.30% | 71.25% |

## 指标说明
- **final_score**: 加权综合分 (llm_judge 45% + structured_point_coverage 25% + semantic_similarity 20% + claim_rouge_l 5% + technical_entity_match 5%)
- **llm_judge**: LLM 评判得分 (accuracy 35% + completeness 25% + factual_consistency 20% + usefulness 10% + clarity 10%)
- **semantic**: 语义相似度
- **coverage**: 结构化要点覆盖率
- **claim_rouge_l**: 声明级别的 ROUGE-L 分数
- **entity_match / precision / recall / f_beta**: 技术实体匹配的精确率、召回率和 F-beta 分数
- **unsupported_entity_rate**: 未被证据支持的实体比例（越低越好）
- **fully_correct_rate**: 完全正确的样本比例
- **critical_error_rate**: 包含严重错误的样本比例（越低越好）
- **core_conclusion_hit_rate**: 核心结论命中率

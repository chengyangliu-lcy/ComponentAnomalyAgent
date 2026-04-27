# 评测指标重设计汇报

## 改了什么

本次把原来偏“字面相似”的评测，改成了更适合技术问答的三类指标：

- `rouge_l` -> `claim_rouge_l`
- `bigram_jaccard` -> `technical_entity_match`
- `scoring_point_coverage` -> 结构化采分点覆盖

同时补充了几个更直接的诊断字段：

- `fully_correct_rate`
- `critical_error_rate`
- `core_conclusion_hit_rate`
- `unsupported_entity_rate`
- `required_point_coverage`

## 三个核心指标怎么变了

### 1. `claim_rouge_l`

旧版 `rouge_l` 是整篇答案的字面重合，容易被表达方式影响。

新版改成按 claim 逐条判断：

- 先把参考答案拆成多个关键 claim
- 再看预测答案是否覆盖这些 claim
- 更看重“关键点有没有说到”，而不是“句子像不像”

优势：

- 对同义改写更友好
- 对答案顺序不敏感
- 更适合作为“参考要点覆盖”的辅助指标

### 2. `technical_entity_match`

旧版 `bigram_jaccard` 只看字符片段重合，无法区分技术实体。

新版改成动态技术实体匹配，重点看：

- 元件编号
- 型号和料号
- 数值和单位
- 技术缩写
- 关键技术短语

还会单独记录 `unsupported_entities`，用于发现编造的关键实体。

优势：

- 更贴近电子故障诊断场景
- 能识别真正重要的技术对象
- 能单独暴露编造风险
- 不会被普通字面重复误导

### 3. `scoring_point_coverage`

旧版采分点主要靠正则和简单匹配，结构不够清楚。

新版改成结构化采分点覆盖，采分点分成：

- `core_conclusion`
- `cause_mechanism`
- `component_or_value`
- `diagnostic_step`
- `fix_suggestion`
- `caveat`

每个采分点都有：

- `weight`
- `required`
- `aliases`
- `point_type_confidence`
- `match_evidence`
- `contradiction_evidence`

命中状态也更细：

- `hit`
- `partial`
- `missed`
- `contradicted`

优势：

- 能区分核心结论、原因、步骤和建议
- 支持部分命中，不是非黑即白
- 核心结论错误会被单独标出来
- 空采分点不再默认满分
- 可解释性更强，方便逐样本复盘

## 整体收益

这次改造后，评测从“看起来像不像参考答案”转向了“有没有覆盖关键技术判断”。

主要收益是：

1. 对合理改写更公平。
2. 对技术实体和数值更敏感。
3. 能更清楚地区分核心正确、部分正确和关键错误。
4. 能单独看出编造实体和核心矛盾。
5. baseline 和 agent 终于是在同一套更公平的规则下比较。

## 一句话总结

这次评测重设计的核心，不是让分数更高，而是让分数更像“技术答案是否真的答对了”。

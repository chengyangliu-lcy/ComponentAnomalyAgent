# ComponentAnomalyAgent 历史上下文

## 项目目标

ComponentAnomalyAgent 是面向电子组件异常分析问答任务的 Agent 系统。系统读取 `2025_dataset.jsonl` 中的问题、参考答案和输入图片，生成中文技术答案，并通过统一评测层输出可解释分数、采分点分析、baseline 对比和实验报告。

核心目标：

- 针对电路板、元器件、拓扑、异常现象和维修/分析类问题生成可评测答案。
- 支持图片输入、联网搜索、网页读取、领域技能匹配和模型推理。
- 避免测试集数据泄漏，保证推理证据来源可追踪。
- 用统一评分口径比较 Agent 与 baseline。
- 支持可恢复、可并发、可复现实验流程。

## 重要约束

- `2025/` 目录中的 Markdown、HTML、JSON 是测试集原始资料，默认禁止作为检索知识库读取，避免数据泄漏。
- Agent 允许读取样本输入图片，但不能把测试集原始帖子资料当作外部知识使用。
- 默认本地知识库路径是 `knowledge_base/`，只有显式开启 `enable_local_retrieval` 后才用于检索。
- 当前默认 Agent 策略是 `agentic_tool_loop`，保留 legacy 单 Agent 管线作为 runtime 失败时的 fallback。
- 没有 API key 时可以走 fallback 或本地评测，但真实生成和 LLM Judge 效果不可代表正式结果。

## 已完成功能

### 1. 数据解析与输入图片处理

- 从 `2025_dataset.jsonl` 加载样本。
- 解析 `sample_id/post_id`、题面、参考答案和原始消息。
- 按 `2025/<month>/<post_id>/images/<filename>` 解析图片路径。
- `scripts/validate_data.py` 可校验数据集、参考答案和图片路径。

### 2. Legacy Agent 推理管线

- 保留原始规划、执行、反思、答案合成管线。
- 支持输入图片、联网搜索、本地检索开关、trace 记录和错误收集。
- 当 agentic runtime 报错且配置允许时，可自动 fallback 到 legacy pipeline。

### 3. Agentic Evidence Runtime

当前主要推理策略是 OpenHands 风格的只读证据循环：planner 每轮选择一个动作，runtime 执行工具并记录观察结果，直到证据足够或预算耗尽后生成最终答案。

主要文件：

- `agent/openhands_runtime.py`
- `tools/evidence_tools.py`
- `agent/prompts.py`
- `agent/pipeline.py`

已实现动作：

- `inspect_image`：用视觉模型抽取图片中的元件、丝印、数值、连接、波形和异常线索。
- `match_domain_skill`：匹配 `knowledge_base/domain_skills.yaml` 中的通用电子领域技能。
- `web_search`：通过 Tavily、Brave、Bing、SerpAPI 或 HTML 搜索获取公开证据。
- `web_read`：读取公开网页；对 PDF、慢站或被拦截页面保留搜索摘要作为证据。
- `rank_evidence`：按题面关键词、可信来源、图片证据和领域技能证据排序去重。
- `review_evidence`：检查证据覆盖度，指出图片、元件、原因或处理建议等缺口。
- `finish_answer`：基于已排序证据合成最终中文技术答案。

runtime 能力：

- 控制最大迭代数、总耗时、planner/tool/vision/final answer timeout。
- 限制连续工具错误、planner 失败次数、图片检查重试次数和网页读取页数。
- planner 不可用或返回非法 JSON 时使用确定性 fallback action。
- 图片检查超时或网页读取失败属于可恢复错误，不阻塞最终回答。
- 最终结果中记录 `plan.selected_actions`、`final_stop_reason`、预算和完整 `tool_trace`。

### 4. 提示词集中管理

`agent/prompts.py` 集中管理以下提示词：

- planner 工具选择 JSON 合约。
- 视觉图片描述提示词，要求只描述可见内容，不给最终维修结论。
- 最终答案提示词，强调先直接回答、基于证据、不编造型号/参数/波形。
- LLM Judge 提示词，统一输出 JSON 评分字段。

最终答案提示词特别覆盖常见题型：

- 充电芯片或 LED 状态异常。
- NTC/浪涌限流选型。
- 缓启动/预充电阻计算。
- 运放恒流源/负反馈。
- RC 或三极管振荡/闪烁灯。

### 5. LLM 客户端增强

`llm_client.py` 已支持：

- OpenAI-compatible chat completion。
- timeout 和 max_retries。
- `extra_body`，用于 DashScope/Qwen 的 `enable_thinking: false` 等参数。
- `response_format`，用于 planner/Judge JSON 输出。
- `json_chat` 自动剥离 Markdown code fence 并解析 JSON。
- `_repair_truncated_json`：对截断或格式错误的 JSON 尝试修复——自动补齐闭合括号，逐步移除尾部残缺片段后重新解析。对 `max_tokens` 不足导致的截断（如 score ~4935 chars 处截断）已验证可修复。

默认模型配置位于 `configs/default.yaml`：

- `AGENT_MODEL` / `default_agent_model`
- `VISION_MODEL` / `default_vision_model`
- `JUDGE_MODEL` / `default_judge_model`
- planner、vision、answer 分别可配置 `extra_body`
- judge 专用 `max_tokens` 通过 `evaluation.judge_max_tokens` 配置，默认 4000（独立于 `model.max_tokens`）

### 6. 联网证据检索

已支持多 provider 搜索：

- Tavily
- Brave Search
- Bing Search
- SerpAPI
- HTML fallback，默认 DuckDuckGo

配置入口：

- `web.provider_order`
- `web.max_results_per_query`
- `web.max_pages_to_read`
- `web.api_key_envs`
- `web.api_keys`

网页读取会尽量获取正文；对于 PDF、厂商慢站或被拦截页面，会保留搜索结果摘要，避免工具失败导致整条样本中断。

### 7. 推理脚本与 trace 输出

`scripts/run_infer.py` 已支持：

- `--sample-id` 跑单样本。
- `--sample-ids-file` 跑固定样本集合，按文件顺序过滤。
- `--limit` 限制样本数。
- `--enable-web` / `--disable-web` 临时覆盖联网开关。
- `--no-resume` 重新生成输出。
- `--max-workers` 并发推理。
- 自动写出 `predictions.jsonl`。
- 为每个样本写出 `traces/<sample_id>.trace.json`。
- 区分 hard failed 和 warning samples。

预测输出字段：

- `sample_id`
- `question`
- `answer`
- `tools_used`
- `web_searched`
- `tool_trace`
- `reasoning_summary`
- `elapsed_seconds`
- `token_usage`
- `errors`
- `plan`

### 7.5 知识库检索策略

当前知识库是 `knowledge_base/circuit_diagnosis_fts`（SQLite FTS5 + Dense 向量索引），存储从 elecfans.com 等电子技术网站爬取的电路诊断技术文档（12,111 条候选 chunks）。

**检索方式**：混合检索（Dense 65% + Sparse 35%）

- Dense：Qwen3-Embedding-4B 向量检索，语义匹配
- Sparse：SQLite FTS5 + BM25 重排序，关键词匹配
- 合并后按 hybrid score 排序，每 URL 最多取 2 chunks

**多层过滤**（候选 → 高相关性仅保留 12.2%）：

1. 低价值来源过滤（非技术类 URL）
2. 水文/模板文本过滤
3. 低价值项目过滤
4. Required terms 过滤（query 关键词须在结果中出现）
5. 低相关性过滤（sparse score < 5.0 或 hybrid score < 0.15）

**调用门槛**（`openhands_runtime._should_use_local_retrieval`）：

- 必须同时满足：有强实体（型号/位号/参数值）+ 有故障/公式/拓扑需求
- 如果有图片证据但缺上述条件，优先用图片而跳过 KB
- 每样本最多调用 1 次，返回最多 4 chunks

配置入口：

- `agent.enable_local_retrieval`
- `agent.max_local_retrieval_calls`
- `agent.max_local_docs`
- `agent.max_local_chunks`
- `retrieval.embedding_model`
- `retrieval.dense_weight` / `retrieval.sparse_weight`

### 7.6 最新实验对比（2025-05-01）

两组实验各 368 样本，对比有知识库 vs 无知识库的效果：

| 指标 | 有 KB (kb_rag_full_v4_web) | 无 KB (agent_full_v4) | Delta |
|---|---|---|---|
| final_score mean | 0.5809 | 0.5805 | +0.004 |
| llm_judge score | 0.6877 | 0.677 | +0.011 |
| accuracy | 3.51 | 3.45 | +0.07 |
| factual_consistency | 0.728 | 0.711 | +0.017 |
| semantic_similarity | 0.767 | 0.767 | ~0 |
| scoring_point_coverage | 0.389 | 0.401 | -0.012 |
| fully_correct | 3/368 | 1/368 | +2 |

逐样本胜负：RAG 赢 152 / Agent 赢 149 / 平局 67。

**核心结论**：知识库几乎未带来提升（final_score 差仅 0.004），三大原因：

1. **检索门槛太严**：只在"强实体 + 故障/拓扑"时触发，55% 的题目不调 KB
2. **过滤太狠**：候选 2329 → 保留 283（88% 被丢弃），检索非空率仅 49%
3. **知识利用率为 0**：`kb_evidence_used_rate = 0.27%`，planner 拿到 KB 内容后未有效融入答案

KB 诊断数据（kb_rag 实验）：

| 指标 | 值 |
|---|---|
| local_retrieve 触发率 | 167/368 = 45.4% |
| 检索非空率 | 49.1% |
| avg_chunks 返回 | 1.7 |
| KB 证据用于答案率 | 0.27% |
| 候选 → 高相关性 | 283/2329 = 12.2% |

潜在改进方向：放宽检索触发条件和过滤策略；优化 planner prompt 让其更积极将 KB 知识融入答案。

### 8. 统一评测体系

统一评测入口是 `scripts/run_eval.py` 和 `evaluator/evaluate.py`。

已实现指标：

- Semantic Similarity：优先 BERTScore，可 fallback 到 sentence-transformer。
- Claim ROUGE-L：分 claim 加权评分。
- Scoring point coverage：结构化采分点覆盖率（hit/partial/missed/contradicted）。
- Technical entity match：idf-weighted F-beta，含 precision/recall/unsupported entity 惩罚。
- LLM Judge：多维度质量评判（accuracy/completeness/clarity/usefulness/factual_consistency）。
- Error analysis：自动诊断低分原因和严重级别。

综合分公式（当前权重，与旧版不同）：

```text
FinalScore = 0.45 * LLMJudgeScore
           + 0.25 * ScoringPointCoverage
           + 0.20 * SemanticSimilarity
           + 0.05 * ClaimRougeL
           + 0.05 * TechnicalEntityMatch

LLMJudgeScore = 0.35 * accuracy_norm
              + 0.25 * completeness_norm
              + 0.20 * factual_consistency
              + 0.10 * usefulness_norm
              + 0.10 * clarity_norm
```

当 LLM Judge 被禁用/失败时，对应权重自动重分配到其余指标。当采分点 coverage 为 null 时同样跳过重分配。

评测输出字段：

- `semantic_similarity`（score + backend + error）
- `claim_rouge_l`（score + claims + claim_scores + claim_weights）
- `technical_entity_match`（score/precision/recall/f1/f_beta + 各类实体列表）
- `llm_judge`（enabled + score + accuracy/completeness/clarity/usefulness/average_score/factual_consistency + fully_correct + critical_errors）
- `scoring_points`（reference_points + hit/missed/false_positive + coverage + matches + structured_points）
- `final_score`
- `fully_correct`
- `error_analysis`（reasons + severity）
- `elapsed_seconds`

#### LLM Judge 可靠性保障（三层兜底）

`evaluator/llm_judge.py` 的 `judge()` 方法现在对同一个 LLM 响应依次尝试三层解析：

1. **直接 JSON 解析**：标准 `json.loads`
2. **JSON 修复**（`llm_client.py._repair_truncated_json`）：对截断的 JSON 响应自动补齐闭合括号，并尝试移除尾部残缺片段后重新解析
3. **正则提取**（`llm_judge.py._regex_fallback_extract`）：从原始文本中用正则提取 accuracy/completeness/clarity/usefulness/factual_consistency/score 等核心数值字段

根因修复：judge 专用的 `max_tokens` 从 2000 提升到 4000（通过 `evaluation.judge_max_tokens` 配置项），大幅降低截断概率。

#### 评测修复脚本

`scripts/repair_eval.py` 可对已有实验目录中 LLM Judge 失败的样本单独重新评测：

```bash
cd ComponentAnomalyAgent && .venv/bin/python scripts/repair_eval.py \
  --config configs/kb_diagnosis.yaml \
  --experiment agent_full_v4_rerun \
  --max-workers 4
```

#### 评测已知问题

- `run_eval.py` 在高并发 (`--max-workers > 1`) 时，`append_jsonl` 可能产生重复行（8 workers 时观察到 19/387 重复）。需用 `scripts/dedupe_jsonl.py` 或手动去重。
- DashScope API 在 8 并发下可能触发限流，导致部分样本 LLM Judge 超时（90s timeout），表现为 `llm=0.0000`。建议 `--max-workers 4` 并增大 `request_timeout_seconds`。

### 9. Qwen Baseline 统一化

`qwen_eval.py` 已从原始 baseline 改造为兼容统一实验体系的 baseline 脚本。

已支持：

- 从环境变量读取 API key、base URL、生成模型和输入路径。
- 生成 agent 格式的 `predictions.jsonl`。
- 输出统一 `eval_results.jsonl`。
- 支持 `composite`、`local-only`、`generate-only` 模式。
- 支持 `--sample-ids-file` 固定样本集合。
- 支持 `--retry-failed-only` 只重跑失败样本。
- 支持 `--resume` / `--no-resume`。
- 失败样本写出 `failed_sample_ids.txt` 和 `<output_dir>_failed_ids.txt`。
- dedupe predictions/eval rows，优先保留成功结果。
- 写出 summary 和 `baseline_score.json`。

`scripts/run_baseline.py` 已支持：

- 读取已有 baseline 结果并转换为统一评测格式。
- 复用已经包含统一评测字段的 qwen_eval 输出。
- `--force-reevaluate` 强制重新评测。
- `--sample-ids-file` 固定样本集合。
- 跳过失败 generation。
- 同时写 `eval_results.jsonl` 和兼容旧命名的 `baseline_eval_results.jsonl`。

### 10. 实验辅助工具

已新增：

- `scripts/select_samples.py`：按 seed 从数据集中抽取固定样本 ID。
- `scripts/dedupe_jsonl.py`：按 `sample_id/post_id` 去重 JSONL，可保留 first 或 last。
- `scripts/compare_runs.py`：比较 agent 和 baseline 的统一评测 JSONL。
- `scripts/repair_eval.py`：对已有实验目录中 LLM Judge 失败的样本单独重新评测。
- `tools/sample_ids.py`：读取、去重和按顺序过滤样本 ID。

`scripts/compare_runs.py` 能力：

- 默认发现重复 sample_id 时报错，避免不公平比较。
- 可用 `--duplicates keep-last` 诊断性保留最后一条。
- 默认要求 agent/baseline 样本集合一致。
- 可用 `--sample-set shared` 只比较交集。
- 生成 sample set 报告、win rate、runtime action 统计、fallback finish 统计、final answer timeout 统计和 top tool sequences。

### 11. 一键实验流程

`scripts/run_experiment.py` 可串联：

- Agent 推理。
- 统一评测。
- 可选 baseline 对比。

常用方式：

```bash
uv run python scripts/run_experiment.py --experiment agent_20 --limit 20
```

如果已有 baseline 统一评测结果：

```bash
uv run python scripts/run_experiment.py \
  --experiment agent_20_compare \
  --limit 20 \
  --baseline-eval outputs/baseline_20/eval_results.jsonl
```

### 12. JSON 输出可靠性

`tools/utils.py` 已增强：

- `append_jsonl` 和 `write_json` 会清理不可序列化或不安全 JSON 值。
- 非有限浮点数写为 `null`。
- 字符串中的非法控制字符替换为 `\ufffd`。
- `write_json` 使用临时文件加 `os.replace` 原子写入。

### 13. 测试覆盖

当前已有 unittest 测试：

- agentic runtime 工具循环、非法 action 恢复、图片超时降级、重复工具 guard、trace JSON 输出。
- evidence tools：搜索 provider fallback、API key 优先级、网页读取降级、图片检查、领域技能加载和 forbidden root 防泄漏、最终答案证据压缩去重。
- prompts 合约。
- qwen baseline 统一评测兼容逻辑。
- sample ids 读取和过滤。
- JSONL 去重工具。
- run compare 的重复 ID 和样本集合检查。

当前验证命令：

```bash
uv run python -m unittest discover tests
```

最近一次通过结果：33 个测试 OK。

## 常用命令

安装依赖：

```bash
uv sync
```

下载本地评测模型：

```bash
uv run python scripts/download_models.py
```

校验数据：

```bash
uv run python scripts/validate_data.py
```

单样本推理：

```bash
uv run python scripts/run_infer.py --sample-id 1326045 --experiment test_one --no-resume
```

禁用联网推理：

```bash
uv run python scripts/run_infer.py --sample-id 1326045 --experiment reason_only --disable-web --no-resume
```

按固定样本文件推理：

```bash
uv run python scripts/run_infer.py --sample-ids-file sample_ids.txt --experiment agent_fixed --no-resume
```

推理后评测：

```bash
uv run python scripts/run_eval.py \
  --predictions outputs/agent_fixed/predictions.jsonl \
  --experiment agent_fixed
```

运行 qwen baseline 并直接生成统一评测：

```bash
uv run python qwen_eval.py \
  --output-dir outputs/baseline_fixed \
  --sample-ids-file sample_ids.txt \
  --mode composite
```

只重跑 baseline 失败样本：

```bash
uv run python qwen_eval.py \
  --output-dir outputs/baseline_fixed \
  --sample-ids-file sample_ids.txt \
  --retry-failed-only
```

比较 agent 和 baseline：

```bash
uv run python scripts/compare_runs.py \
  --agent outputs/agent_fixed/eval_results.jsonl \
  --baseline outputs/baseline_fixed/eval_results.jsonl \
  --agent-predictions outputs/agent_fixed/predictions.jsonl \
  --output outputs/agent_fixed/compare_report.json
```

抽样固定样本：

```bash
uv run python scripts/select_samples.py \
  --dataset 2025_dataset.jsonl \
  --limit 20 \
  --seed 42 \
  --output sample_ids.txt
```

去重 JSONL：

```bash
uv run python scripts/dedupe_jsonl.py \
  --input outputs/agent_fixed/predictions.jsonl \
  --output outputs/agent_fixed/predictions.deduped.jsonl \
  --keep last
```

## 关键配置

默认配置文件：

```text
configs/default.yaml
```

重要环境变量：

- `DASHSCOPE_API_KEY` 或 `OPENAI_API_KEY`
- `DASHSCOPE_BASE_URL`
- `AGENT_MODEL`
- `VISION_MODEL`
- `JUDGE_MODEL`
- `BASELINE_GENERATION_MODEL`
- `BASELINE_INPUT_DATASET`
- `BASELINE_IMAGE_ROOT`
- `TAVILY_API_KEY`
- `BRAVE_SEARCH_API_KEY`
- `BING_SEARCH_API_KEY`
- `SERPAPI_API_KEY`

关键 runtime 配置：

- `agent.strategy: agentic_tool_loop`
- `agent.fallback_to_legacy_on_runtime_error: true`
- `agent.max_iterations`
- `agent.max_total_seconds`
- `agent.planner_timeout_seconds`
- `agent.tool_timeout_seconds`
- `agent.vision_timeout_seconds`
- `agent.final_answer_timeout_seconds`
- `agent.web_read_timeout_seconds`
- `agent.max_consecutive_tool_errors`
- `agent.max_planner_failures`
- `agent.enable_web_search`
- `agent.enable_image_inspection_llm`
- `agent.max_llm_images`
- `agent.enable_local_retrieval`
- `agent.max_local_retrieval_calls`
- `evaluation.enable_llm_judge`
- `evaluation.judge_max_tokens`（默认 4000，独立于 model.max_tokens）
- `evaluation.final_weights.llm_judge: 0.45`
- `evaluation.final_weights.structured_point_coverage: 0.25`
- `evaluation.final_weights.semantic_similarity: 0.20`
- `evaluation.final_weights.claim_rouge_l: 0.05`
- `evaluation.final_weights.technical_entity_match: 0.05`
- `retrieval.embedding_model`
- `retrieval.dense_weight: 0.65`
- `retrieval.sparse_weight: 0.35`
- `web.provider_order`
- `web.max_results_per_query`
- `web.max_pages_to_read`
- `runtime.max_workers`
- `runtime.request_timeout_seconds`（judge 和 agent 共用，默认 90）
- `runtime.resume`

## 主要文件地图

- `README.md`：项目说明和常用命令。
- `configs/default.yaml`：默认配置。
- `configs/config.py`：配置加载和模型/API key 属性。
- `schemas.py`：样本、证据、plan、trace、推理结果等结构。
- `llm_client.py`：OpenAI-compatible LLM 客户端。
- `agent/pipeline.py`：AgentPipeline 入口，负责 agentic/legacy 分流。
- `agent/openhands_runtime.py`：agentic action/observation loop。
- `agent/prompts.py`：planner、vision、answer、judge 提示词。
- `agent/synthesizer.py`：legacy final answer synthesizer。
- `tools/evidence_tools.py`：agentic runtime 的工具执行器。
- `tools/web_search.py`：HTML 搜索。
- `tools/web_reader.py`：网页正文读取。
- `tools/dataset_parser.py`：数据集解析。
- `tools/image_resolver.py`：图片路径解析。
- `tools/sample_ids.py`：样本 ID 文件读取和过滤。
- `tools/utils.py`：JSONL/JSON 写入、文本压缩、timer。
- `tools/circuit_kb.py`：知识库混合检索（Dense + Sparse + 多层过滤）。
- `tools/retriever.py`：Dense vector retriever (Qwen3-Embedding-4B)。
- `evaluator/evaluate.py`：统一评测入口。
- `evaluator/llm_judge.py`：LLM Judge（三层兜底：JSON 解析 → 修复 → 正则提取）。
- `evaluator/rouge_eval.py`：Claim ROUGE-L。
- `evaluator/semantic_similarity.py`：BERTScore / sentence-transformer。
- `evaluator/scoring_points.py`：结构化采分点覆盖率。
- `evaluator/jaccard_eval.py`：技术实体 IDF。
- `evaluator/report.py`：评测汇总和错误分析。
- `evaluator/baseline_compare.py`：baseline 对比。
- `qwen_eval.py`：Qwen baseline 生成和统一评测。
- `scripts/run_infer.py`：Agent 推理。
- `scripts/run_eval.py`：评测预测结果。
- `scripts/repair_eval.py`：修复 LLM Judge 失败的评测条目。
- `scripts/run_baseline.py`：baseline 结果统一化。
- `scripts/run_experiment.py`：一键实验。
- `scripts/compare_runs.py`：agent/baseline 对比报告。
- `scripts/dedupe_jsonl.py`：JSONL 厯重。
- `scripts/select_samples.py`：固定样本抽取。
- `tests/`：unittest 测试。

## 最近提交历史

- `9ae75cc feat: unify qwen baseline evaluation`
- `0997084 feat: add agentic evidence runtime`
- `aa07223 feat: add experiment sample utilities`
- `a098ad9 feat: add resumable concurrent run scripts`
- `8dd21d9 feat: unify evaluation scoring`
- `8840b21 feat: improve web evidence retrieval`
- `fca4066 Initial ComponentAnomalyAgent implementation`

## 最近代码改动（2025-05-01，未提交）

- `llm_client.py`：新增 `_repair_truncated_json`，对截断 JSON 尝试闭合修复
- `evaluator/llm_judge.py`：三层解析兜底（JSON → 修复 → 正则提取），只做一次 LLM 调用；新增 `_regex_fallback_extract`；日志记录 fallback 情况
- `evaluator/evaluate.py`：judge `max_tokens` 从 `model.max_tokens` 改为 `evaluation.judge_max_tokens`，默认 4000
- `scripts/repair_eval.py`：新增修复脚本，只重新评测 LLM Judge 失败的样本


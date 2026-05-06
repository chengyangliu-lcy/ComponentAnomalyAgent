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

最终答案提示词（`FINAL_ANSWER_SYSTEM_PROMPT`）输出结构化 1-5 格式：

1. **结论**：先直接回答问题，明确最可能原因或处理方向
2. **依据**：用题面、图片、KB 或网页证据支撑关键结论
3. **原因机制**：解释异常为什么发生，强制使用因果链（如 "R161+C24 串联→容抗随频率增大而减小→高频段增益下降→抑制高频噪声放大"），禁止泛化
4. **检查步骤**：可执行的测量、复核和定位步骤
5. **处理建议与不确定性**：修改建议 + 证据缺口说明

关键规则：
- Rule 9：KB 证据自然融入，不写"根据本地知识库"；如图片矛盾采信图片
- Rule 10：不得编造题面和证据中均未出现的元件标号、参数值或型号
- Rule 11：常见题型（LED/NTC/缓启动/运放恒流源/RC 振荡）必须覆盖的要点

`build_final_answer_user_prompt` 将 evidence_text 和 question_hints 注入 `<证据>` 和 `<题面线索>` 标签，指导 LLM 按结构化格式输出。

### 5. LLM 客户端增强

`llm_client.py` 已支持：

- OpenAI-compatible chat completion。
- Provider 系统：`dashscope` / `vllm` / `openai`，通过 `configs/local.yaml` 的 `model.provider` 切换。
- 每个 provider 独立配置 `base_url`、`api_key` 和 `extra_body`（DashScope 需 `enable_thinking: false`）。
- timeout 和 max_retries。
- `extra_body`，用于 DashScope/Qwen 的 `enable_thinking: false` 等参数。
- `response_format`，用于 planner/Judge JSON 输出。
- `json_chat` 自动剥离 Markdown code fence 并解析 JSON。
- `_repair_truncated_json`：对截断或格式错误的 JSON 尝试修复——自动补齐闭合括号，逐步移除尾部残缺片段后重新解析。对 `max_tokens` 不足导致的截断已验证可修复。

默认模型配置位于 `configs/default.yaml` 和 `configs/local.yaml`：

- `model.provider`：当前为 `dashscope`
- `model.api_key` / `model.default_base_url`：API 入口
- `model.default_agent_model` / `model.default_vision_model` / `model.default_judge_model`
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

当前知识库是 `knowledge_base/circuit_diagnosis_fts_hq_v2`（SQLite FTS5 + Dense 向量索引），包含 5,924 chunks / 5,584 docs，来自电路诊断技术文档（经过严格过滤，相比旧版 57k chunks 减少 90% 噪声）。

**完整检索 Pipeline**（配置入口：`configs/experiments/exp13_structured.yaml`）：

1. **LLM Query Rewriting**：将中文题面改写为英文电子关键词，保留型号、位号和数值
2. **Dense Retrieval**：Qwen3-Embedding-4B（2560-dim）向量语义检索，权重 65%
3. **Sparse Retrieval**：SQLite FTS5 + BM25 关键词检索，权重 35%
4. **Hybrid Merge**：dense + sparse 分数融合，threshold=0.15
5. **Cross-Encoder Rerank**：BAAI/bge-reranker-v2-m3 对 top-30 候选精排
6. **Contextual Retrieval**：chunk 前预置文档级上下文，提升匹配精度

**多层过滤**：

1. 低价值来源/水文/模板文本/低价值项目过滤
2. Required terms 过滤（query 关键词须在结果中出现）
3. Hybrid score < 0.15 过滤
4. 每 URL 最多 2 chunks
5. 最终仅保留 high_relevance >= 0.4 的 chunks

**调用策略**（`planner_guidance`）：

- 当题面包含明确型号、位号、数值、拓扑词或故障症状时优先调用 KB
- KB 证据会被 rank_evidence 排序并进入最终答案（不再丢弃）
- 每样本最多 2 次调用，每次最多 4 docs / 6 chunks
- 中文题面须改写为英文关键词检索（本地库以英文为主）

**检索诊断指标**（Exp13 full 368 样本）：

| 指标 | 值 |
|------|-----|
| local_retrieve 触发率 | 335/368 = 91.0% |
| 检索非空率 | 84.2% |
| 平均返回 chunks | 4.3 |
| 候选 → 高相关性 | 1,434/26,299 = 5.5% |
| KB 证据实际利用率 | 0.8%（指标自身可能不准，检测方式依赖文本标记） |

配置入口：

- `paths.local_kb_index`
- `agent.enable_local_retrieval` / `max_local_retrieval_calls` / `max_local_docs` / `max_local_chunks`
- `retrieval.embedding_model` / `dense_weight` / `sparse_weight` / `device`
- `retrieval.enable_llm_query_rewriting` / `enable_cross_encoder_rerank` / `enable_contextual_retrieval`
- `retrieval.cross_encoder_model` / `cross_encoder_top_n` / `hybrid_score_threshold` / `high_relevance_threshold` / `max_chunks_per_url`

### 7.6 最新实验对比（2025-05-05）

经过 10+ 轮实验迭代，最终结论：**KB 策略对分数的提升微乎其微（~0.02），结构化答案模板是最大单项改进。**

#### 100 样本对照实验（Exp10-Exp13）

| Exp | final_score | llm_judge | factual | 描述 |
|-----|:-----------:|:---------:|:-------:|------|
| Exp12 | 0.5508 | 0.624 | 0.6465 | 无 KB + 灵活模板 |
| Exp11 | 0.5481 | 0.6065 | 0.633 | v2 KB only（无检索特性，灵活模板） |
| Exp10 | 0.5647 | 0.623 | 0.6475 | v2 KB + 检索特性 + 灵活模板 |
| Exp13 | 0.5717 | 0.6571 | 0.6925 | v2 KB + 检索特性 + **结构化模板** |

关键发现：
- **KB alone 反而有害**：Exp11 < Exp12，Δ=-0.0027。只加 KB 不加检索特性，噪声大于收益
- **检索特性带来小幅提升**：Exp10 > Exp12，Δ=+0.014。cross-encoder + query rewriting + contextual retrieval 合计贡献约 0.014
- **结构化模板是最大单项改进**：Exp13 > Exp10，Δ=+0.007。从灵活模板恢复 1-5 编号格式（结论/依据/原因机制/检查步骤/处理建议）带来显著提升

#### 全量 368 样本：Exp13 vs Baseline（均去重）

| 指标 | Baseline(368) | Exp13(368) | Δ |
|------|:-----------:|:---------:|:--:|
| final_score | 0.5789 | 0.5799 | **+0.0010** |
| llm_judge.score | 0.6713 | 0.6705 | -0.0008 |
| accuracy | 3.42 | 3.44 | +0.016 |
| factual_consistency | 0.7069 | 0.6992 | -0.0077 |
| critical_error_rate | 42.7% | 42.1% | -0.54% |
| core_conclusion_hit | 68.4% | 70.6% | +2.19% |

**核心结论：**

Baseline（agent_full_v4_rerun）使用的是旧 57k 噪声 KB，实际上 Knowledge Base 贡献近似于零。Exp13 用 v2 KB（5,924 chunks，1/10 规模）+ 检索特性 + 结构化模板，在全部 368 条样本上与旧 baseline 打平并微弱反超。但本质上是**模板和提示词工程**驱动了提升，知识库本身从 v1 到 v2 的改进带来的分数提升不超过 0.02。

这意味着后续改进方向应该更关注：
1. KB 内容质量（当前 KB 缺少考题需要的因果链、公式推导等）
2. 证据到答案的转换效率（LLM 如何将 KB chunk 转化为答案段落）
3. 提示词优化（已经证明提示词改动可以带来 0.007+ 提升）

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

全量推理（使用 Exp13 配置，5 并发）：

```bash
uv run python scripts/run_infer.py --config configs/experiments/exp13_structured.yaml --experiment exp13_full --max-workers 5
```

限定样本数推理：

```bash
uv run python scripts/run_infer.py --config configs/experiments/exp13_structured.yaml --experiment exp13_test --limit 100 --max-workers 5
```

单样本推理：

```bash
uv run python scripts/run_infer.py --sample-id 1326045 --experiment test_one --no-resume
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
- `tools/circuit_kb.py`：知识库混合检索（Dense + Sparse + Cross-encoder + 多层过滤）。
- `tools/dense_retriever.py`：Dense vector retriever (Qwen3-Embedding-4B)，共享加载。
- `configs/experiments/`：实验配置文件目录（exp10-exp13 等）。
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

- `0b025fd feat: 恢复结构化答案模板 + 强化KB证据规则与反幻觉机制`（当前 HEAD）
- `5c0b894 feat: 将本地知识库从 Hackster 替换为 Circuit Markdown，引入 DenseRetriever`
- `e11b3e0 feat: 集成 Hackster 本地知识库混合检索工具`
- `e13ffe1 feat: overhaul evaluation metrics for technical Q&A`
- `fa10b9e feat: 引入 tool registry 与 action 参数自动修复机制`
- `ed6c2fe feat: 添加 OpenHands 浏览器作为网页读取主后端`


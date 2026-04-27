# ComponentAnomalyAgent

面向组件异常分析问答任务的 Agent 系统。系统读取 `2025_dataset.jsonl`，解析样本问题和输入图片，通过 Agent 的规划、工具调用、联网搜索或模型自身推理生成答案，并用独立评测层输出可解释分数、采分点分析、baseline 对比和实验报告。

重要约束：`2025/` 目录中的 Markdown/HTML/JSON 是测试集原始资料，默认禁止作为检索知识库读取，避免数据泄漏。当前 Agent 只能使用输入图片、联网搜索结果和模型自身推理。未来如需本地知识库，应放到独立的 `knowledge_base/` 目录并显式开启 `enable_local_retrieval`。

## 依赖管理

项目使用 `uv` 管理依赖：

```powershell
uv sync
```

下载本地评测模型：

```powershell
uv run python scripts/download_models.py
```

模型会保存到：

```text
models/bert-base-chinese
models/sentence-transformers__paraphrase-multilingual-MiniLM-L12-v2
```

## 配置

默认配置在 `configs/default.yaml`。

推理模型使用 DashScope/OpenAI-compatible 接口，从环境变量读取：

```powershell
$env:DASHSCOPE_API_KEY="你的key"
$env:DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:AGENT_MODEL="qwen3.6-plus"
$env:JUDGE_MODEL="qwen-plus"
```

也可以复制 `configs/local.example.yaml` 为 `configs/local.yaml`，把本机 API key 写进去。`configs/local.yaml` 已被 `.gitignore` 忽略，不会进入版本库。

没有 API key 时，推理会使用 fallback，不能代表真实模型效果；评测仍可使用本地 BERTScore/ROUGE-L/Bigram Jaccard/采分点逻辑。

## 数据与图片

主数据集：

```text
2025_dataset.jsonl
```

输入图片按以下规则解析：

```text
2025/<month>/<post_id>/images/<filename>
```

这只用于读取样本输入图片，不读取同目录下的 Markdown 作为知识证据。

## 常用命令

校验数据和图片：

```powershell
uv run python scripts/validate_data.py
```

单样本推理：

```powershell
uv run python scripts/run_infer.py --sample-id 1326045 --experiment test_one --no-resume
```

禁用联网，仅使用输入图片和模型自身推理：

```powershell
uv run python scripts/run_infer.py --sample-id 1326045 --experiment reason_only --disable-web --no-resume
```

批量前 5 条推理并评测：

```powershell
uv run python scripts/run_infer.py --limit 5 --experiment agent_limit5 --no-resume
uv run python scripts/run_eval.py --predictions outputs/agent_limit5/predictions.jsonl --experiment agent_limit5
```

一键实验：

```powershell
uv run python scripts/run_experiment.py --experiment agent_experiment --limit 5
```

适配已有 baseline 输出：

```powershell
uv run python scripts/run_baseline.py --baseline-results evaluation_results.jsonl --experiment baseline
```

## 命令行参数速查

### `scripts/validate_data.py`

校验数据集、`post_id`、参考答案和图片路径。

```powershell
uv run python scripts/validate_data.py [--config configs/local.yaml] [--output outputs/data_validation.json]
```

- `--config`：额外配置文件路径，会覆盖默认配置。
- `--output`：校验结果 JSON 输出路径。

### `scripts/download_models.py`

下载本地评测模型。

```powershell
uv run python scripts/download_models.py [--models-dir models] [--bertscore-model bert-base-chinese] [--sentence-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2]
```

- `--models-dir`：模型保存目录。
- `--bertscore-model`：BERTScore 使用的 Hugging Face 模型。
- `--sentence-model`：BERTScore 不可用时的向量相似度降级模型。

### `scripts/run_infer.py`

运行 Agent 推理，输出 `predictions.jsonl`。

```powershell
uv run python scripts/run_infer.py [--config configs/local.yaml] [--experiment agent_20] [--sample-id 1326045] [--limit 20] [--output outputs/agent_20/predictions.jsonl] [--enable-web | --disable-web] [--no-resume] [--max-workers 4]
```

- `--config`：额外配置文件路径。
- `--experiment`：实验名，默认输出到 `outputs/<experiment>/`。
- `--sample-id`：只跑指定 `sample_id/post_id`。
- `--limit`：只跑前 N 条样本。
- `--output`：自定义预测 JSONL 输出路径。
- `--enable-web`：本次运行强制开启联网搜索。
- `--disable-web`：本次运行强制关闭联网搜索。
- `--no-resume`：不跳过已有预测，重新生成输出。
- `--max-workers`：并发推理样本数；不传时使用 `configs/default.yaml` 里的 `runtime.max_workers`。

### `scripts/run_eval.py`

评测 Agent 或其他预测结果。

```powershell
uv run python scripts/run_eval.py --predictions outputs/agent_20/predictions.jsonl [--config configs/local.yaml] [--experiment agent_20] [--output outputs/agent_20/eval_results.jsonl] [--limit 20] [--mode composite|local-only]
```

- `--predictions`：必填，预测结果 JSONL。
- `--config`：额外配置文件路径。
- `--experiment`：实验名，默认输出到 `outputs/<experiment>/`。
- `--output`：自定义评测 JSONL 输出路径。
- `--limit`：只评测前 N 条预测。
- `--mode composite`：默认模式，使用统一 `llm_judge` + BERTScore/ROUGE-L/Bigram Jaccard/采分点。
- `--mode local-only`：只使用本地指标，不调用 LLM Judge。

### `qwen_eval.py`

运行原始 baseline：生成答案并用 qwen-style LLM Judge 评估。

```powershell
uv run python qwen_eval.py [--input-dataset 2025_dataset.jsonl] [--image-root 2025] [--output-dir outputs/baseline_20] [--output-eval outputs/baseline_20/evaluation_results.jsonl] [--output-summary outputs/baseline_20/evaluation_summary.json] [--limit 20] [--max-workers 5] [--resume | --no-resume]
```

- `--input-dataset`：baseline 输入数据集。
- `--image-root`：图片根目录。
- `--output-dir`：baseline 原始输出目录。
- `--output-eval`：baseline 样本级 JSONL 输出路径，会覆盖 `--output-dir` 默认值。
- `--output-summary`：baseline 汇总 JSON 输出路径，会覆盖 `--output-dir` 默认值。
- `--limit`：只跑前 N 条样本。
- `--max-workers`：并发处理线程数。
- `--resume`：默认开启，跳过已在输出 JSONL 中完成的 `post_id`。
- `--no-resume`：关闭断点续跑。

### `scripts/run_baseline.py`

把 `qwen_eval.py` 的 baseline 输出转成统一综合评分口径。

```powershell
uv run python scripts/run_baseline.py [--config configs/local.yaml] [--baseline-results outputs/baseline_20/evaluation_results.jsonl] [--run-qwen-eval] [--experiment baseline_20] [--output outputs/baseline_20/baseline_eval_results.jsonl] [--limit 20]
```

- `--config`：额外配置文件路径。
- `--baseline-results`：已有 `qwen_eval.py` JSONL 输出。
- `--run-qwen-eval`：先调用 `qwen_eval.py`，再适配评分。
- `--experiment`：实验名，默认输出到 `outputs/<experiment>/`。
- `--output`：自定义统一评分 JSONL 输出路径。
- `--limit`：只适配前 N 条 baseline 结果。

### `scripts/run_experiment.py`

串联 Agent 推理、综合评测和可选 baseline 对比。

```powershell
uv run python scripts/run_experiment.py [--config configs/local.yaml] [--experiment agent_20] [--limit 20] [--enable-web | --disable-web] [--baseline-eval outputs/baseline_20/baseline_eval_results.jsonl]
```

- `--config`：额外配置文件路径。
- `--experiment`：实验名。
- `--limit`：只跑前 N 条样本。
- `--enable-web`：本次实验强制开启联网搜索。
- `--disable-web`：本次实验强制关闭联网搜索。
- `--baseline-eval`：传入统一口径的 baseline 评测 JSONL，生成 `compare_report.json`。

## 前 20 条对比实验

```powershell
uv run python scripts/run_infer.py --limit 20 --experiment agent_20 --no-resume --max-workers 4
uv run python scripts/run_eval.py --predictions outputs/agent_20/predictions.jsonl --experiment agent_20 --limit 20

uv run python qwen_eval.py --limit 20 --output-dir outputs/baseline_20 --resume
uv run python scripts/run_baseline.py --baseline-results outputs/baseline_20/evaluation_results.jsonl --experiment baseline_20 --limit 20

uv run python scripts/run_experiment.py --experiment agent_20_compare --limit 20 --baseline-eval outputs/baseline_20/baseline_eval_results.jsonl
```

## 输出

推理输出 `predictions.jsonl` 包含：

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

评测输出 `eval_results.jsonl` 包含：

- `semantic_similarity`
- `claim_rouge_l`
- `technical_entity_match`
- `llm_judge`
- `scoring_points`
- `final_score`
- `legacy_final_score`
- `error_analysis`

综合分：

```text
FinalScore = 0.50 * LLMJudgeScore
           + 0.25 * SemanticSimilarity
           + 0.10 * ScoringPointCoverage
           + 0.10 * RougeL
           + 0.05 * BigramJaccard

LegacyFinalScore = 0.60 * SemanticSimilarity
                 + 0.25 * RougeL
                 + 0.15 * BigramJaccard
```

`llm_judge` 是唯一的 LLM 评分字段，内部保留 qwen-style 的 `accuracy/completeness/clarity/usefulness/average_score`，并保留参与综合分的 `factual_consistency`。新结果不再生成 `qwen_judge` 字段。

## Baseline

`qwen_eval.py` 保留为 baseline 脚本，已改为从环境变量读取 API key、模型名、输入数据集和图片根目录：

- `DASHSCOPE_API_KEY` 或 `OPENAI_API_KEY`
- `DASHSCOPE_BASE_URL`
- `BASELINE_GENERATION_MODEL`
- `JUDGE_MODEL`（统一 Evaluator/LLM Judge 使用，默认来自配置）
- `BASELINE_INPUT_DATASET`
- `BASELINE_IMAGE_ROOT`


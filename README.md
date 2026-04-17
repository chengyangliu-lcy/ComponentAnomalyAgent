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
- `rouge_l`
- `bigram_jaccard`
- `llm_judge`
- `scoring_points`
- `final_score`
- `error_analysis`

综合分：

```text
FinalScore = 0.60 * SemanticSimilarity + 0.25 * RougeL + 0.15 * BigramJaccard
```

## Baseline

`qwen_eval.py` 保留为 baseline 脚本，已改为从环境变量读取 API key、模型名、输入数据集和图片根目录：

- `DASHSCOPE_API_KEY` 或 `OPENAI_API_KEY`
- `DASHSCOPE_BASE_URL`
- `BASELINE_GENERATION_MODEL`
- `BASELINE_EVAL_MODEL`
- `BASELINE_INPUT_DATASET`
- `BASELINE_IMAGE_ROOT`

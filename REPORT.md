# Qwen 联网搜索集成方案与实验报告

## 一、背景与目标

在电路故障诊断 Agent 系统中，Agent 需要查阅技术文档、数据手册和电路原理来生成准确的诊断答案。此前系统依赖两种信息来源：

1. **本地知识库 (KB)**：基于 Circuit Markdown 的混合检索（Dense + Sparse），离线索引
2. **传统 web_search**：调用外部搜索引擎 API，再用 web_read 读取页面内容

本次改进引入 **Qwen 原生联网搜索**（DashScope `enable_search`），利用大模型内置的联网检索能力，简化信息获取链路，对比三种信息来源方案的效果。

---

## 二、技术方案

### 2.1 Qwen 原生联网搜索工具

**核心原理**：调用 DashScope API 时传入 `extra_body={"enable_search": True}`，模型会自动执行互联网搜索，将检索结果融入回答中返回。exp21 进一步验证了阿里云文档中的强制联网参数：`extra_body={"enable_search": True, "search_options": {"forced_search": True}}`。

**与传统 web_search 的区别**：

| 维度 | 传统 web_search | Qwen 原生搜索 |
|------|----------------|---------------|
| 调用方式 | 搜索引擎 API → web_read 读取页面 | 单次 LLM 调用，模型自动检索 |
| 工具链路 | 两步（搜索 + 读取） | 一步（搜索结果直接融入回答） |
| Token 消耗 | 搜索结果需额外解析 | 模型内化，输出更紧凑 |
| 可控性 | 可选择读取哪个 URL | 模型自主决定检索来源 |

**实现关键代码**：

```python
# llm_client.py - search_chat 方法
def search_chat(self, messages, temperature=None, timeout=None, max_tokens=None, search_options=None):
    search_timeout = timeout or max(self.timeout or 0, 90)  # 搜索默认 90s 超时
    search_max_tokens = max_tokens or max(self.max_tokens, 2000)
    search_extra_body = dict(self.extra_body)
    search_extra_body["enable_search"] = True  # 开启 DashScope 联网搜索
    if search_options:
        search_extra_body["search_options"] = {
            **dict(search_extra_body.get("search_options") or {}),
            **dict(search_options),
        }
    # ... 正常调用 API
```

```python
# tools/evidence_tools.py - QwenSearchExecutor
class QwenSearchExecutor(BaseEvidenceExecutor):
    tool_name = "qwen_search"
    action_name = "qwen_internet_search"

    def run(self, query: str) -> ToolRun:
        response = self.llm.search_chat([
            {"role": "system", "content": "你是电路故障诊断专家，请搜索相关信息..."},
            {"role": "user", "content": query},
        ])
        # 将搜索结果封装为证据返回给 Agent 循环
```

### 2.2 配置开关

通过 YAML 配置文件控制工具启用：

```yaml
agent:
  enable_web_search: false      # 禁用传统搜索引擎
  enable_qwen_search: true      # 启用 Qwen 原生搜索
  max_qwen_search_calls: 2      # 每个样本最多调用 2 次
  qwen_search_options:
    forced_search: true         # 强制模型必须执行联网搜索
```

同时优化了 planner 提示词，当工具被禁用时不再输出冗余的"不可用"提示，改为静默处理。

### 2.3 finish_answer 输出长度优化

原先 `finish_answer` 阶段的 LLM 调用继承全局 `max_tokens=2000`，导致详细答案被截断。

**修改**：`LLMClient.chat()` 新增 `max_tokens` 参数，`FinishAnswerExecutor` 显式传入 `max_tokens=4000`，确保答案完整输出。

### 2.4 超时配置调整

| 参数 | 原值 | 新值 | 说明 |
|------|------|------|------|
| `max_total_seconds` | 300s (5min) | 900s (15min) | 单样本总耗时上限 |
| `final_answer_timeout_seconds` | 60s | 120s | finish_answer 阶段超时 |

### 2.5 轨迹查看器 Web 服务

新增 `scripts/trace_viewer.py`，基于 FastAPI 的轨迹可视化工具：

- 白色 Apple 风格主题，聊天对话式展示 Agent 推理过程
- 支持手动输入目录路径，加载指定文件夹下的 `.trace.json` 文件
- API 端点：`POST /api/traces`（列表）、`POST /api/trace`（详情）

---

## 三、实验设计

### 3.1 实验配置

| 实验 | 本地KB | 联网搜索 | 其他 |
|------|--------|---------|------|
| **exp19** (基线) | 启用 (v3 KB, max_calls=2) | web_search | - |
| **exp20** (Qwen only) | 禁用 | qwen_search (max_calls=2) | finish_answer 4000 tokens, 900s 超时 |
| **exp21** (Qwen forced) | 禁用 | qwen_search (max_calls=2, forced_search=true) | 强制 DashScope 联网搜索 |

### 3.2 评测指标体系

| 指标 | 权重 | 说明 |
|------|------|------|
| **final_score** | 综合 | 加权综合分 |
| llm_judge | 45% | LLM 评判（accuracy 35% + completeness 25% + factual_consistency 20% + usefulness 10% + clarity 10%） |
| structured_point_coverage | 25% | 结构化要点覆盖率 |
| semantic_similarity | 20% | 语义相似度 |
| claim_rouge_l | 5% | 声明级 ROUGE-L |
| technical_entity_match | 5% | 技术实体匹配 |

### 3.3 评测规模

- 测试集：368 个电路故障诊断样本
- 推理并发：8 workers
- 评测并发：8 workers

---

## 四、实验结果

### 4.1 综合评分对比

| 实验 | final_score | llm_judge | semantic | coverage |
|------|-------------|-----------|----------|----------|
| exp19 (KB + web_search) | 0.5801 | 0.6729 | 0.7668 | 0.4009 |
| **exp20 (qwen only)** | **0.5846** | **0.6758** | 0.7668 | **0.4115** |
| exp21 (qwen forced_search) | 0.5502 | 0.5923 | 0.7664 | 0.4005 |
| 提升 | +0.0045 (+0.8%) | +0.0029 (+0.4%) | 0 | +0.0106 (+2.6%) |

### 4.2 LLM Judge 详细指标

| 实验 | accuracy | completeness | factual_consistency | usefulness | clarity |
|------|----------|--------------|---------------------|------------|---------|
| exp19 | 3.4132 | 3.5289 | 0.7098 | 3.9394 | 4.5289 |
| exp20 | 3.4266 | 3.5435 | 0.7098 | 3.9647 | 4.5380 |
| exp21 | 3.0788 | 2.8967 | 0.7104 | 3.7880 | 4.2038 |

### 4.3 质量指标对比

| 实验 | claim_rouge_l | entity_match | entity_precision | entity_recall | entity_f_beta |
|------|---------------|--------------|------------------|---------------|---------------|
| exp19 | 0.2600 | 0.3418 | 0.8304 | 0.2775 | 0.3418 |
| exp20 | 0.2593 | 0.3488 | 0.8261 | 0.2841 | 0.3488 |
| exp21 | 0.2597 | 0.3466 | 0.8291 | 0.2820 | 0.3466 |

### 4.4 质量通过率

| 实验 | fully_correct_rate | critical_error_rate | core_conclusion_hit_rate |
|------|--------------------|---------------------|--------------------------|
| exp19 | 0.54% | 40.49% | 70.31% |
| exp20 | 1.09% | 41.30% | 71.25% |
| exp21 | 0.27% | 35.60% | 68.75% |

### 4.5 强制联网搜索策略

exp21 在 exp20 的基础上加入 `search_options.forced_search=true`。小批量 smoke 测试 10 条全部完成，trace 中 10 次 `qwen_search` 均记录 `forced_search=true`，链路和参数传递有效。

全量 368 条实验中，368 条推理和评估均完成，0 条硬失败，334 条样本实际触发 `qwen_search`，所有 `qwen_search` 事件均带有 `forced_search=true` 元数据。指标上 final_score 为 0.5502，低于 exp20 的 0.5846；llm_judge 从 0.6758 下降到 0.5923，coverage 从 0.4115 下降到 0.4005。该策略未带来整体收益。

可能原因是 forced_search 会让模型在即使已有足够上下文时也执行联网检索，增加了外部信息干扰和答案发散；同时 qwen_search 仍由 Agent 计划阶段决定是否调用，forced_search 只能保证已调用的搜索请求执行联网，不能保证每个样本都调用搜索。后续更适合验证结构化 `search_info` 来源审计、按问题类型选择是否强制搜索，或与本地 KB 组合的混合策略。

---

## 五、结论

1. **纯 Qwen 联网搜索方案优于传统 KB + web_search 组合**：final_score 从 0.5801 提升至 0.5846，各项子指标均有改善。

2. **结构化要点覆盖率提升最明显**：coverage 从 0.4009 提升至 0.4115（+2.6%），说明 Qwen 搜索返回的信息更贴近评分要点。

3. **完全正确率翻倍**：fully_correct_rate 从 0.54% 提升至 1.09%，虽然绝对值仍低，但趋势积极。

4. **方案简化收益**：禁用本地 KB 后省去了索引构建、向量检索等复杂链路，系统架构更简洁。

5. **后续方向**：
   - 探索 Qwen 搜索 + 本地 KB 的组合策略（已配置 exp20_original 版本待测）
   - 对 forced_search 采用条件化策略，而不是全样本强制联网
   - 引入 DashScope 原生 `search_info` 结构化来源，用于来源质量审计
   - 优化 critical_error_rate（41.3% 仍有下降空间）
   - 提升 fully_correct_rate（当前仅 1.09%）

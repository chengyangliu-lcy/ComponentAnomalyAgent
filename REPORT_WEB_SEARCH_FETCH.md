# Web Search & Fetch 免费方案调研 + 实验分析

> 调研日期：2026-05-11
> 实验分析：exp19_v3_max2_full (368 samples, qwen3.6-plus)

---

## 一、实验 exp19 网络搜索失败分析

### 1.1 总体数据

| 指标 | 数值 |
|------|------|
| 总样本数 | 368 |
| 尝试搜索的样本 | 99 (27%) |
| 搜索成功 | 40 (40%) |
| 搜索失败 | 59 (60%) |
| 未尝试搜索 | 269 (73%) |
| **web_read 使用次数** | **0** |

### 1.2 搜索失败根因

所有 80 次搜索失败的根因完全一致：**所有 provider 全部失败**。

| Provider | 失败原因 | 出现次数 |
|----------|----------|----------|
| **Tavily** | 432 错误（API Key 过期/无效） | 80/80 |
| **Brave** | 缺少 API Key (BRAVE_SEARCH_API_KEY) | 80/80 |
| **Bing API** | 缺少 API Key (BING_SEARCH_API_KEY) | 80/80 |
| **SerpAPI** | 缺少 API Key (SERPAPI_API_KEY) | 80/80 |
| **DuckDuckGo (HTML)** | 返回无可解析结果 | 80/80 |
| **DuckDuckGo (HTML)** | 读取超时 (timeout=20s) | 33/80 |
| **DuckDuckGo (HTML)** | 403 Forbidden | 2/80 |
| **Bing (HTML)** | 结果通过相关性过滤后为空 | 79/80 |
| **Yahoo (HTML)** | 500 服务器错误 | 41/80 |
| **Yahoo (HTML)** | 返回无可解析结果 | 80/80 |

**核心问题：**
1. **API Key 问题**：Tavily Key 已失效 (432)，其余三个 API 完全没有配置 Key
2. **HTML 爬取全面崩溃**：DuckDuckGo、Bing、Yahoo 三个引擎同时失败
3. **DuckDuckGo 超时率高**：33/80 次超时（41%），说明反爬或网络不稳定

### 1.3 失败查询特征

| 特征 | 失败查询 | 成功查询 |
|------|----------|----------|
| 含中文 | 73% (58/80) | 55% (22/40) |
| 平均长度 | 99 字符 | 类似 |
| 最长 | 172 字符 | - |

含中文的查询更容易失败，可能是因为：
- 中文+英文混合查询对 DuckDuckGo HTML 解析更困难
- Yahoo/Bing 对中文查询的结果质量更差

### 1.4 web_read 从未使用

**这是一个严重的架构问题：**

- `web_read` 在所有 368 个样本的 `allowed_tools` 中都存在
- 但 Planner LLM **从未选择** `web_read` 工具（0 次）
- 搜索成功后，Planner 直接跳到 `rank_evidence → finish_answer`
- 这意味着搜索结果只有 snippet，从未获取完整网页内容

**影响：** 即使搜索成功，Agent 也只能拿到搜索引擎的摘要片段（通常 100-200 字），无法获取 datasheet、技术文章的详细内容。

### 1.5 搜索对分数的影响

| 组别 | 样本数 | 平均分 | <0.4 | 0.4-0.6 | >=0.6 |
|------|--------|--------|------|---------|-------|
| 搜索成功 | 40 | 0.488 | 25% | 48% | 28% |
| 搜索失败 | 59 | 0.497 | 31% | 42% | 27% |
| 未搜索 | 269 | 0.612 | 13% | 27% | 61% |

搜索成功和失败的样本分数几乎一样（0.488 vs 0.497），说明：
- 搜索失败的样本并非因为搜索失败而丢分 — 它们本身就是难题
- 搜索成功的样本也没有从搜索中获得显著提升 — **因为没读取网页内容**

---

## 二、改善方案

### 2.1 搜索层改善

#### 方案 A：SearXNG 自部署（推荐）

**解决的问题：** 替代全部 HTML 爬取，提供稳定的 JSON API

```bash
docker run -d --name searxng -p 8888:8080 \
  -e SEARXNG_BASE_URL=http://localhost:8888 \
  searxng/searxng
```

在 `APIWebSearchExecutor` 中新增 provider：

```python
def _search_searxng(self, query: str, limit: int) -> tuple[list[Evidence], str | None]:
    resp = requests.get(
        self.searxng_url,  # http://localhost:8888
        params={"q": query, "format": "json", "categories": "general"},
        timeout=self.timeout
    )
    results = resp.json().get("results", [])[:limit]
    if not results:
        return [], "searxng: no results"
    evidence = []
    for r in results:
        evidence.append(Evidence(
            title=r.get("title", ""),
            url=r.get("url", ""),
            snippet=r.get("content", ""),
            source="searxng"
        ))
    return evidence, None
```

**预期效果：**
- 聚合 70+ 搜索引擎，单引擎被反爬不影响整体
- 提供结构化 JSON，无需解析 HTML
- 完全免费，无调用限制
- 对中文查询支持更好（可配置中文引擎如百度）

#### 方案 B：DuckDuckGo Python 库替代 HTML 爬取

**解决的问题：** 零配置，比直接爬 HTML 稳定

```bash
pip install duckduckgo-search
```

```python
from duckduckgo_search import DDGS

def _search_ddgs_lib(self, query: str, limit: int) -> tuple[list[Evidence], str | None]:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=limit))
        if not results:
            return [], "ddgs: no results"
        evidence = []
        for r in results:
            evidence.append(Evidence(
                title=r["title"], url=r["href"], snippet=r["body"], source="ddgs_lib"
            ))
        return evidence, None
    except Exception as e:
        return [], f"ddgs: {e}"
```

**预期效果：**
- 比 HTML 爱爬稳定（库维护者跟踪接口变化）
- 零成本、零配置
- 缺点：仍可能被封，中文结果质量一般

#### 方案 C：配置 Brave Search API Key

**解决的问题：** 最小改动，立即生效

1. 注册 https://api.search.brave.com （免费 2,000次/月）
2. 在 `configs/local.yaml` 添加：
```yaml
web:
  api_keys:
    brave: "BSAxxxxxxxxxxxxx"
  provider_order: [brave, html]
```

**预期效果：**
- 2,000 次/月免费额度，覆盖 368 个样本绰绰有余
- 官方 API，稳定可靠
- 改动量最小（只改配置）

### 2.2 抓取层改善

#### 方案 D：Jina Reader 作为 fetch fallback

**解决的问题：** web_read 从未被使用，即使搜索成功也只拿到 snippet

在 `RobustWebReadExecutor` 中新增 Jina Reader 层：

```python
def _fetch_jina(self, url: str, max_chars: int = 8000) -> str | None:
    """Jina Reader: URL 前加 r.jina.ai/ 获取干净 markdown"""
    try:
        resp = requests.get(
            f"https://r.jina.ai/{url}",
            headers={"Accept": "text/plain"},
            timeout=20
        )
        if resp.status_code == 200 and len(resp.text) > 100:
            return resp.text[:max_chars]
    except:
        pass
    return None
```

**集成到 RobustWebReadExecutor.run() 的 fallback 链：**
1. 跳过 PDF 和已知慢站点
2. OpenHandsBrowserFetcher（如果启用）
3. **Jina Reader（新增）**
4. WebReader (requests + BeautifulSoup)
5. snippet-only fallback

**预期效果：**
- 无需 API Key（基础版免费）
- 服务端处理 JS 渲染和反爬
- 返回干净 Markdown，LLM 可直接消费
- 比 OpenHandsBrowserFetcher 更轻量

#### 方案 E：强制 Planner 使用 web_read

**解决的问题：** Planner 从未选择 web_read，搜索只有 snippet

这需要修改 `_fallback_action()` 或在 planner prompt 中强调：

```python
# 在 planner system prompt 中添加：
# "When web_search returns results, you MUST use web_read to fetch at least 
#  one relevant page for detailed content before finishing the answer."
```

或者在 `_guard_action()` 中强制：如果本轮有 web_search 成功但未 web_read，自动插入 web_read。

### 2.3 推荐组合方案

| 优先级 | 方案 | 改动量 | 预期效果 |
|--------|------|--------|----------|
| **P0** | 配置 Brave API Key | 5 行配置 | 搜索成功率从 33% → 90%+ |
| **P1** | SearXNG 部署 + 新增 provider | ~80 行代码 | 搜索完全免费，无限额 |
| **P1** | Jina Reader 作为 fetch fallback | ~30 行代码 | 网页抓取成功率大幅提升 |
| **P2** | DuckDuckGo 库替代 HTML 爬取 | ~30 行代码 | 零配置备选方案 |
| **P2** | 强制 Planner 使用 web_read | ~20 行代码 | 搜索后获取完整内容 |
| **P3** | Crawl4AI 替代 OpenHands | ~50 行代码 | 更轻量的 JS 渲染抓取 |

---

## 三、免费方案全景

### 3.1 Web Search 方案对比

| 方案 | 免费额度 | 需要 API Key | 需要部署 | 反爬处理 | 搜索质量 | 推荐度 |
|------|----------|-------------|----------|----------|----------|--------|
| **SearXNG** | 无限制 | 否 | 是 (Docker) | 聚合多引擎，分散风险 | 高 | ⭐⭐⭐⭐⭐ |
| **DuckDuckGo 库** | 无限制 | 否 | 否 | 非官方，可能被封 | 中 | ⭐⭐⭐⭐ |
| **Brave Search** | 2,000次/月 | 是 | 否 | 官方 API | 高 | ⭐⭐⭐⭐ |
| **Tavily** | 1,000次/月 | 是 | 否 | 官方 API | 高 | ⭐⭐⭐ |
| **Google Custom** | 100次/天 | 是 | 否 | 官方 API | 最高 | ⭐⭐⭐ |
| **Bing API** | 1,000次/月 | 是 | 否 | 官方 API | 高 | ⭐⭐⭐ |
| **SerpAPI** | 100次/月 | 是 | 否 | 服务端处理 | 高 | ⭐⭐ |

### 3.2 Web Fetch 方案对比

| 方案 | 免费 | JS 渲染 | 反爬处理 | 集成复杂度 | 推荐度 |
|------|------|---------|----------|-----------|--------|
| **Jina Reader** | 有限免费 | 是 | 服务端处理 | 极低 | ⭐⭐⭐⭐⭐ |
| **Crawl4AI** | 完全免费 | 是 (Playwright) | 自行管理 | 低 | ⭐⭐⭐⭐ |
| **Firecrawl 自部署** | 完全免费 | 是 (Playwright) | 自行管理 | 中 | ⭐⭐⭐ |

### 3.3 反爬问题总结

当前 HTML 爱爬方式的根本问题：直接爬搜索引擎结果页，被反爬是必然的。

解决思路（按优先级）：
1. **用官方 API 替代爬取** — Brave/Tavily/Google 都有免费 API
2. **自部署 SearXNG** — 聚合多引擎，分散风险
3. **用 Jina Reader/Crawl4AI 替代直接抓取** — 服务端处理反爬
4. **自己处理反爬**（不推荐）— 复杂度高，维护成本大

---

## 四、总结

### 实验 exp19 的核心问题

1. **API Key 全面失效**：Tavily Key 过期，Brave/Bing/SerpAPI 未配置
2. **HTML 爬取全面崩溃**：DDG 超时+无结果，Bing 无相关结果，Yahoo 500 错误
3. **web_read 从未使用**：Planner 不选择读取网页，搜索只有 snippet
4. **搜索成功也没用**：成功/失败样本分数几乎一样（0.488 vs 0.497）

### 最小成本最大收益的改善路径

```
Step 1: 注册 Brave Search API Key → 搜索成功率 33% → 90%+  (5分钟)
Step 2: 部署 SearXNG Docker → 搜索完全免费无限额          (15分钟)
Step 3: 新增 Jina Reader fallback → 网页抓取能力恢复       (30分钟)
Step 4: 修改 Planner prompt → 强制搜索后读取网页           (10分钟)
```

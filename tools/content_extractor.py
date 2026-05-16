from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Iterable

from bs4 import BeautifulSoup

from tools.utils import compact_text

try:  # Optional at runtime; declared as a project dependency.
    import trafilatura
except Exception:  # noqa: BLE001
    trafilatura = None  # type: ignore[assignment]

try:  # Optional at runtime; declared as a project dependency.
    from readability import Document
except Exception:  # noqa: BLE001
    Document = None  # type: ignore[assignment]


_FAILURE_PATTERNS = (
    "cloudflare",
    "captcha",
    "access denied",
    "403 forbidden",
    "forbidden",
    "required part couldn't load",
    "target url returned error 404",
    "http 错误 404",
    "404.0 - not found",
    "enable javascript",
    "please enable cookies",
    "checking your browser",
    "verify you are human",
)

_BOILERPLATE_PATTERNS = (
    "accept cookies",
    "cookie policy",
    "privacy policy",
    "terms of use",
    "sign in",
    "log in",
    "subscribe",
    "newsletter",
    "advertisement",
    "skip to content",
    "toggle navigation",
    "select language",
    "all rights reserved",
    "contact us",
    "share this",
    "社区首页",
    "全部新帖",
    "资料区",
    "社区活动",
    "联系管理员",
    "专业技术中心",
    "公司介绍",
    "产品选型",
    "联系我们",
    "当前位置",
    "返回首页",
)

_NAV_WORDS = {
    "home",
    "products",
    "product",
    "solutions",
    "support",
    "resources",
    "applications",
    "company",
    "about",
    "login",
    "register",
    "cart",
    "menu",
    "search",
    "forum",
    "blog",
    "download",
    "pricing",
}

_MOJIBAKE_MARKERS = ("Ã", "Â", "�", "ï¼", "ã€", "å", "æ", "ç", "é", "è")

_FORUM_NOISE_PATTERNS = (
    "\u8bf7 [\u767b\u5f55]",  # 请 [登录]
    "\u5feb\u6377\u5bfc\u822a",  # 快捷导航
    "\u6ca1\u6709\u5e10\u53f7",  # 没有帐号
    "\u6700\u540e\u767b\u5f55",  # 最后登录
    "\u5728\u7ebf\u65f6\u95f4",  # 在线时间
    "\u82af\u79ef\u5206",  # 芯积分
    "E\u91d1\u5e01",  # E金币
    "\u590d\u5236\u94fe\u63a5",  # 复制链接
    "\u4e0b\u8f7d\u9644\u4ef6",  # 下载附件
    "\u4fdd\u5b58\u5230\u76f8\u518c",  # 保存到相册
    "\u53ea\u770b\u8be5\u4f5c\u8005",  # 只看该作者
    "\u5f53\u524d\u79bb\u7ebf",  # 当前离线
    "\u626b\u4e00\u626b",  # 扫一扫
)

_NOISE_CONTENT_PATTERNS = (
    "网站地图",
    "站点地图",
    "sitemap",
    "精品范文",
    "学术之家",
    "最新作品发布时间",
    "综合视频",
    "认证徽章",
    "抖音综合搜索",
)


@dataclass
class ExtractedContent:
    content: str
    quality_score: float
    quality_reason: str
    extractor: str
    raw_chars: int
    clean_chars: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def acceptable(self, min_score: float, min_chars: int) -> bool:
        return self.clean_chars >= min_chars and self.quality_score >= min_score


def extract_llm_markdown(
    raw: str,
    *,
    title: str = "",
    snippet: str = "",
    max_chars: int = 8000,
    source_format: str = "text",
) -> ExtractedContent:
    """Extract readable Markdown-like content and score it for LLM consumption."""

    raw = raw or ""
    if _looks_like_failure(raw):
        return ExtractedContent(
            content="",
            quality_score=0.0,
            quality_reason="blocked_or_failure_page",
            extractor="failure_detector",
            raw_chars=len(raw),
            clean_chars=0,
        )

    candidates: list[tuple[str, str]] = []
    is_html_input = source_format in {"html", "auto"} or _looks_like_html(raw)
    if is_html_input:
        trafilatura_text = _extract_with_trafilatura(raw)
        if trafilatura_text:
            candidates.append(("trafilatura", trafilatura_text))
        readability_text = _extract_with_readability(raw)
        if readability_text:
            candidates.append(("readability", readability_text))
        soup_text = _extract_with_bs4(raw)
        if soup_text:
            candidates.append(("bs4_main_text", soup_text))

    if not is_html_input or not candidates:
        candidates.append(("generic_line_filter", raw))

    best: ExtractedContent | None = None
    candidate_scores: list[dict[str, Any]] = []
    for extractor, text in candidates:
        cleaned = _clean_markdownish_text(text)
        cleaned = _trim_to_context(cleaned, title=title, snippet=snippet)
        score, reason = _score_content(cleaned, raw, title=title, snippet=snippet)
        block_filtered = _select_relevant_blocks(cleaned, title=title, snippet=snippet)
        if block_filtered and block_filtered != cleaned:
            block_score, block_reason = _score_content(block_filtered, raw, title=title, snippet=snippet)
            if _candidate_rank(block_score, len(block_filtered)) >= _candidate_rank(score, len(cleaned)):
                cleaned = block_filtered
                score = block_score
                reason = f"{block_reason};block_selected=true"
        candidate_scores.append(
            {
                "extractor": extractor,
                "score": round(score, 4),
                "clean_chars": len(cleaned),
                "reason": reason,
            }
        )
        result = ExtractedContent(
            content=compact_text(cleaned, max_chars=max_chars),
            quality_score=score,
            quality_reason=reason,
            extractor=extractor,
            raw_chars=len(raw),
            clean_chars=len(cleaned),
            metadata={"candidate_scores": candidate_scores},
        )
        if best is None or _candidate_rank(result.quality_score, result.clean_chars) > _candidate_rank(
            best.quality_score, best.clean_chars
        ):
            best = result

    if best:
        best.metadata = {"candidate_scores": candidate_scores, "selected_extractor": best.extractor}
    return best or ExtractedContent("", 0.0, "empty", "none", len(raw), 0)


def _extract_with_trafilatura(raw: str) -> str:
    if trafilatura is None:
        return ""
    try:
        return (
            trafilatura.extract(
                raw,
                output_format="markdown",
                favor_precision=True,
                include_comments=False,
                include_tables=True,
                deduplicate=True,
            )
            or ""
        )
    except Exception:  # noqa: BLE001
        return ""


def _extract_with_readability(raw: str) -> str:
    if Document is None:
        return ""
    try:
        doc = Document(raw)
        html = doc.summary(html_partial=True)
        soup = BeautifulSoup(html or "", "html.parser")
        title = doc.short_title() or ""
        text = soup.get_text("\n", strip=True)
        return "\n\n".join(part for part in [title.strip(), text.strip()] if part)
    except Exception:  # noqa: BLE001
        return ""


def _extract_with_bs4(raw: str) -> str:
    try:
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "iframe"]):
            tag.decompose()
        main = (
            soup.find("article")
            or soup.find("main")
            or soup.find(attrs={"role": "main"})
            or soup.find(id=re.compile("content|main|article|post", re.I))
            or soup.find(class_=re.compile("content|main|article|post|entry", re.I))
            or soup.body
            or soup
        )
        return main.get_text("\n", strip=True)
    except Exception:  # noqa: BLE001
        return ""


def _clean_markdownish_text(text: str) -> str:
    lines = [line.strip() for line in (text or "").replace("\r", "\n").split("\n")]
    kept: list[str] = []
    seen_counts: dict[str, int] = {}

    for line in lines:
        line = _normalize_line(line)
        if not line:
            continue
        normalized_key = re.sub(r"\W+", " ", line.lower()).strip()
        if not normalized_key:
            continue
        seen_counts[normalized_key] = seen_counts.get(normalized_key, 0) + 1
        if seen_counts[normalized_key] > 2:
            continue
        if _is_boilerplate_line(line):
            continue
        kept.append(line)

    return "\n\n".join(_merge_short_runs(kept)).strip()


def _select_relevant_blocks(cleaned: str, *, title: str, snippet: str) -> str:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", cleaned or "") if block.strip()]
    if len(blocks) < 4:
        return cleaned

    scored: list[tuple[float, int, str]] = []
    for idx, block in enumerate(blocks):
        score = _block_score(block, title=title, snippet=snippet)
        scored.append((score, idx, block))
    positive = [(score, idx, block) for score, idx, block in scored if score > 0.12]
    if not positive:
        return cleaned

    keep_indexes: set[int] = set()
    for score, idx, _block in sorted(positive, reverse=True)[: max(6, len(blocks) // 3)]:
        keep_indexes.add(idx)
        if idx > 0 and _block_score(blocks[idx - 1], title=title, snippet=snippet) > 0.02:
            keep_indexes.add(idx - 1)
        if idx + 1 < len(blocks) and _block_score(blocks[idx + 1], title=title, snippet=snippet) > 0.02:
            keep_indexes.add(idx + 1)

    selected = "\n\n".join(blocks[idx] for idx in sorted(keep_indexes))
    if len(selected) < 500 and len(cleaned) > 1200:
        return cleaned
    return selected


def _block_score(block: str, *, title: str, snippet: str) -> float:
    lowered = block.lower()
    tokens = _tokens(block)
    if not tokens:
        return 0.0
    link_density = (lowered.count("](") + lowered.count("http://") + lowered.count("https://")) / max(len(tokens), 1)
    length_score = min(len(block) / 900.0, 1.0) * 0.35
    context_score = min(_context_overlap(block, f"{title} {snippet}") / 0.16, 1.0) * 0.30 if title or snippet else 0.08
    technical_score = min(_technical_signal(tokens) / 4.0, 1.0) * 0.20
    density_score = max(0.0, 1.0 - min(link_density * 10.0, 1.0)) * 0.15
    nav_hits = sum(1 for token in tokens if token in _NAV_WORDS)
    boilerplate_hits = sum(lowered.count(pattern) for pattern in _BOILERPLATE_PATTERNS)
    penalty = min((nav_hits / max(len(tokens), 1) * 2.0) + boilerplate_hits * 0.04, 0.7)
    if _is_boilerplate_line(block):
        penalty += 0.3
    return max(0.0, length_score + context_score + technical_score + density_score - penalty)


def _trim_to_context(cleaned: str, *, title: str, snippet: str) -> str:
    context = (snippet or "").strip()
    if not cleaned or len(context) < 20:
        return cleaned

    fragments = [
        context[:120].strip(),
        context[:80].strip(),
        context[:50].strip(),
        context[:35].strip(),
        context[:25].strip(),
        context[:18].strip(),
        context[:12].strip(),
    ]
    fragments.extend(part.strip() for part in re.split(r"[。.!！?？\n]", context) if len(part.strip()) >= 20)

    best_idx: int | None = None
    for fragment in fragments:
        min_len = 8 if _has_cjk(fragment) else 20
        if len(fragment) < min_len:
            continue
        idx = cleaned.find(fragment)
        if idx > 250 and (best_idx is None or idx < best_idx):
            best_idx = idx
    if best_idx is None:
        return cleaned

    prefix = cleaned[:best_idx]
    keepers = []
    for line in prefix.split("\n\n")[:4]:
        stripped = line.strip()
        if stripped.startswith("Title:") or stripped.startswith("#"):
            keepers.append(stripped)
    if not keepers and title:
        keepers.append(title.strip())
    return "\n\n".join([*keepers[:2], cleaned[best_idx:].lstrip()]).strip()


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _normalize_line(line: str) -> str:
    line = re.sub(r"\s+", " ", line).strip()
    line = re.sub(r"^\s*[-*|]+\s*$", "", line)
    return line.strip()


def _merge_short_runs(lines: Iterable[str]) -> list[str]:
    merged: list[str] = []
    buffer: list[str] = []
    for line in lines:
        word_count = len(_tokens(line))
        if 0 < word_count <= 5 and not line.startswith(("#", "|")):
            buffer.append(line)
            if len(buffer) < 3:
                continue
        if buffer:
            merged.append(" ".join(buffer))
            buffer = []
        merged.append(line)
    if buffer:
        merged.append(" ".join(buffer))
    return merged


def _is_boilerplate_line(line: str) -> bool:
    lowered = line.lower()
    words = _tokens(lowered)
    if len(line) <= 2:
        return True
    if lowered.startswith(("url source:", "published time:", "markdown content:")):
        return True
    if lowered.startswith("warning: target url returned error"):
        return True
    if _is_markdown_link_boilerplate(line):
        return True
    if len(words) <= 8 and any(pattern in lowered for pattern in _BOILERPLATE_PATTERNS):
        return True
    if any(pattern in line for pattern in _BOILERPLATE_PATTERNS) and len(line) <= 40:
        return True
    nav_hits = sum(1 for word in words if word in _NAV_WORDS)
    if len(words) <= 12 and nav_hits >= 3:
        return True
    markdown_links = lowered.count("](")
    plain_urls = lowered.count("http://") + lowered.count("https://")
    if len(words) <= 20 and markdown_links + plain_urls >= 3:
        return True
    if re.fullmatch(r"[\W_]+", line):
        return True
    return False


def _is_markdown_link_boilerplate(line: str) -> bool:
    link_count = line.count("](")
    image_count = line.count("![")
    if not link_count and not image_count:
        return False
    without_images = re.sub(r"!\[[^\]]*]\([^)]+\)", "", line)
    without_links = re.sub(r"\[[^\]]*]\([^)]+\)", "", without_images)
    residue = without_links.strip(" -*_>|:;，,。")
    link_text = " ".join(re.findall(r"\[([^\]]*)]\([^)]+\)", line))
    if not residue and len(link_text) <= 80:
        return True
    if link_count >= 2 and len(residue) <= 30:
        return True
    if image_count and len(residue) <= 20:
        return True
    if any(pattern in link_text for pattern in _BOILERPLATE_PATTERNS) and len(residue) <= 30:
        return True
    return False


def _score_content(cleaned: str, raw: str, *, title: str, snippet: str) -> tuple[float, str]:
    if not cleaned:
        return 0.0, "empty_after_cleaning"
    lowered = cleaned.lower()
    if _looks_like_failure(cleaned):
        return 0.0, "blocked_or_failure_page"
    hard_noise = _hard_noise_reason(cleaned, title=title)
    if hard_noise:
        return 0.0, hard_noise

    clean_chars = len(cleaned)
    raw_chars = max(len(raw), 1)
    tokens = _tokens(cleaned)
    unique_ratio = len(set(tokens)) / max(len(tokens), 1)
    clean_ratio = min(clean_chars / raw_chars, 1.0)
    link_density = (lowered.count("](") + lowered.count("http://") + lowered.count("https://")) / max(len(tokens), 1)
    boilerplate_hits = sum(lowered.count(pattern) for pattern in _BOILERPLATE_PATTERNS)
    nav_hits = sum(1 for token in tokens if token in _NAV_WORDS)
    context_overlap = _context_overlap(cleaned, f"{title} {snippet}")
    mojibake_ratio = _mojibake_ratio(cleaned)
    repeated_line_ratio = _repeated_line_ratio(cleaned)
    forum_noise = _forum_noise_score(cleaned)

    length_score = min(clean_chars / 1800.0, 1.0) * 0.35
    uniqueness_score = min(unique_ratio / 0.45, 1.0) * 0.15
    density_score = max(0.0, 1.0 - min(link_density * 12.0, 1.0)) * 0.15
    clean_ratio_score = min(clean_ratio / 0.35, 1.0) * 0.10
    context_score = min(context_overlap / 0.12, 1.0) * 0.15 if title or snippet else 0.08
    technical_score = min(_technical_signal(tokens) / 6.0, 1.0) * 0.10

    penalty = min(
        (boilerplate_hits * 0.03)
        + (nav_hits / max(len(tokens), 1) * 2.0)
        + min(mojibake_ratio * 10.0, 0.55)
        + min(repeated_line_ratio * 0.4, 0.25)
        + min(forum_noise * 0.08, 0.45),
        0.75,
    )
    score = max(0.0, min(1.0, length_score + uniqueness_score + density_score + clean_ratio_score + context_score + technical_score - penalty))

    reason = (
        f"chars={clean_chars};unique={unique_ratio:.2f};links={link_density:.3f};"
        f"context={context_overlap:.2f};tech={_technical_signal(tokens)};"
        f"mojibake={mojibake_ratio:.2f};forum_noise={forum_noise:.1f};"
        f"repeat={repeated_line_ratio:.2f};penalty={penalty:.2f}"
    )
    return score, reason


def _candidate_rank(score: float, clean_chars: int) -> tuple[float, float]:
    # Avoid letting very short, clean boilerplate beat a slightly lower-scoring full article.
    sufficiency = min(clean_chars / 1200.0, 1.0) * 0.08
    return (score + sufficiency, min(clean_chars, 20000))


def _hard_noise_reason(cleaned: str, *, title: str) -> str:
    lowered = cleaned.lower()
    title_lowered = (title or "").lower()
    if _mojibake_ratio(cleaned) >= 0.08 and not _has_cjk(cleaned):
        return "mojibake_or_wrong_encoding"
    link_like = lowered.count("](") + lowered.count("http://") + lowered.count("https://")
    if ("sitemap" in lowered or "网站地图" in cleaned or "站点地图" in cleaned or "sitemap" in title_lowered) and link_like >= 10:
        return "site_map_or_link_index"
    if re.search(r"\|\s*0\.\d+\s*\|\s*daily\s*\|", lowered) and link_like >= 5:
        return "site_map_or_link_index"
    if any(pattern.lower() in lowered for pattern in _NOISE_CONTENT_PATTERNS):
        if link_like >= 5 or _repeated_line_ratio(cleaned) >= 0.35 or len(cleaned) < 800:
            return "search_or_aggregation_page"
    if _repeated_line_ratio(cleaned) >= 0.65 and len(cleaned) > 500:
        return "repetitive_boilerplate"
    return ""


def _mojibake_ratio(text: str) -> float:
    if not text:
        return 0.0
    marker_chars = sum(text.count(marker) for marker in _MOJIBAKE_MARKERS)
    latin1_runs = re.findall(r"[\u00c0-\u00ff]{2,}", text)
    latin1_chars = sum(len(run) for run in latin1_runs)
    replacement_chars = text.count("\ufffd")
    # GBK/UTF-8 mojibake in Chinese pages often appears as dense Latin-1 runs
    # such as "ÔÚµÚ¶þÐÐÏÔÊ¾"; weight those runs higher than isolated symbols.
    weighted = marker_chars + replacement_chars * 4 + latin1_chars * 4
    return weighted / max(len(text), 1)


def _forum_noise_score(text: str) -> float:
    if not text:
        return 0.0
    window = text[:2500]
    hits = sum(window.count(pattern) for pattern in _FORUM_NOISE_PATTERNS)
    if "\u8bba\u575b" in window and ("\u53d1\u5e16" in window or "\u8fd4\u56de\u5217\u8868" in window):
        hits += 2
    return float(hits)


def _repeated_line_ratio(text: str) -> float:
    lines = [line.strip().lower() for line in text.splitlines() if len(line.strip()) >= 20]
    if not lines:
        return 0.0
    unique = set(lines)
    return 1.0 - (len(unique) / len(lines))


def _context_overlap(content: str, context: str) -> float:
    context_tokens = {token for token in _tokens(context) if len(token) >= 4}
    if not context_tokens:
        return 0.0
    content_tokens = set(_tokens(content))
    return len(context_tokens & content_tokens) / max(len(context_tokens), 1)


def _technical_signal(tokens: list[str]) -> int:
    technical_terms = {
        "voltage",
        "current",
        "input",
        "output",
        "datasheet",
        "amplifier",
        "comparator",
        "resistor",
        "capacitor",
        "frequency",
        "gain",
        "offset",
        "power",
        "supply",
        "circuit",
        "package",
        "temperature",
        "typical",
        "maximum",
        "minimum",
        "电压",
        "电流",
        "电源",
        "电路",
        "电感",
        "电容",
        "反馈",
        "纹波",
        "开关",
        "运放",
        "比较",
        "布局",
        "布线",
    }
    hits = sum(1 for token in tokens if token in technical_terms)
    part_like = sum(1 for token in tokens if re.search(r"[a-z]+\d+|\d+[a-z]+", token))
    return hits + min(part_like, 4)


def _tokens(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_+\-.]{1,}", text.lower())
    for run in re.findall(r"[\u4e00-\u9fff]{2,}", text or ""):
        if len(run) <= 4:
            tokens.append(run)
        else:
            tokens.extend(run[idx : idx + 2] for idx in range(0, len(run) - 1))
    return tokens


def _looks_like_html(text: str) -> bool:
    return bool(re.search(r"<\s*(html|body|main|article|div|p|section|table|meta)\b", text or "", re.I))


def _looks_like_failure(text: str) -> bool:
    lowered = (text or "").lower()
    if any(pattern in lowered for pattern in _FAILURE_PATTERNS):
        useful_chars = len(re.sub(r"\s+", "", text))
        if useful_chars < 3000 or "captcha" in lowered or "cloudflare" in lowered:
            return True
    return False

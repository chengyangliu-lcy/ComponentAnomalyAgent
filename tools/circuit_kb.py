from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import urlparse

from schemas import Evidence
from tools.retriever import tokenize
from tools.utils import compact_text


DEFAULT_CIRCUIT_MD_ROOT = Path(
    "/media/work/1ECC291B3E106A4A/xinyang/circuit/warc_output/021150c8-bb23-4a74-9298-0e4be68a6eef"
)
DEFAULT_MAX_DOCS = 10000
DEFAULT_CHUNK_CHARS = 900
DEFAULT_CHUNK_OVERLAP = 120
DEFAULT_MIN_CHUNK_CHARS = 160
DEFAULT_CANDIDATE_LIMIT = 80
DEFAULT_MIN_RELEVANCE_SCORE = 5.0
DEFAULT_MAX_SOURCE_FILE_BYTES = 2_000_000
DB_FILENAME = "circuit_md.sqlite"

PUBLIC_TECH_ALLOWED_PATHS = {
    "www.elecfans.com": (
        "/article/",
        "/dianlutu/",
        "/analog/",
        "/power/",
        "/emc/",
        "/eda/",
        "/soft/",
        "/yuanqijian/",
    ),
    "elecfans.com": (
        "/article/",
        "/dianlutu/",
        "/analog/",
        "/power/",
        "/emc/",
        "/eda/",
        "/soft/",
        "/yuanqijian/",
    ),
    "www.electronicsforu.com": (
        "/electronics-projects/",
        "/electronicsforu/circuitarchives/",
        "/design-guides/",
        "/resources/",
    ),
    "electronicsforu.com": (
        "/electronics-projects/",
        "/electronicsforu/circuitarchives/",
        "/design-guides/",
        "/resources/",
    ),
    "radiokot.ru": (
        "/articles/",
        "/circuit/",
    ),
    "radiokot.ru:81": (
        "/articles/",
        "/circuit/",
    ),
}

PUBLIC_TECH_REJECT_PATH_PARTS = (
    "/tag/",
    "/category/",
    "/news/",
    "/company/",
    "/market",
    "/forum/",
    "/user/",
    "/users/",
    "/projects/details/",
    "/components/",
    "/contest",
    "/products/",
    "/download/",
    "/soft/study/",
)

PUBLIC_TECH_REJECT_TITLES = (
    "404 页面不存在",
    "page not found",
    "301 moved permanently",
    "object moved",
    "提示信息",
    "subscription corner",
    "archives |",
)

BOILERPLATE_LINE_MARKERS = (
    "网站导航",
    "登录 | 注册",
    "欢迎来电子发烧友网",
    "请输入元器件型号",
    "编辑推荐",
    "热门推荐",
    "相关推荐",
    "相关阅读",
    "上一篇",
    "下一篇",
    "copyright",
    "privacy policy",
    "cookie policy",
    "sign in",
    "log in",
    "join",
    "please ensure that javascript is enabled",
    "download files",
    "add new maker",
)

PUBLIC_TECH_SECTION_STOP_MARKERS = (
    "相关技术文章",
    "相关资料下载",
    "相关阅读",
    "上周热点文章排行榜",
    "上周资料下载排行榜",
    "热门文章",
    "热门下载",
    "热门推荐",
    "热门博文",
    "热门社区",
    "论坛热帖",
    "用户评论",
    "发表评论",
    "深度阅读",
    "电子资料下载",
    "相关下载",
    "阅读排行",
    "热门词",
    "创新实用技术专题",
)

PUBLIC_TECH_PREARTICLE_MARKERS = (
    "编辑推荐",
    "推荐帖子",
    "资料下载",
    "技术特刊",
    "社区活动",
    "厂商社区",
    "华强pcb",
    "华强芯城",
    "电子搜索",
    "邮件订阅",
    "文章:新闻",
    "下载:eda",
    "栏目导航",
)

PUBLIC_TECH_CONTENT_TERMS = (
    "emc",
    "emi",
    "pcb",
    "ad",
    "da",
    "adc",
    "dac",
    "模拟地",
    "数字地",
    "布线",
    "退耦",
    "去耦",
    "电流回路",
)

BROAD_QUERY_TERMS = {
    "ac",
    "dc",
    "ce",
    "re",
    "port",
    "common",
    "mode",
    "optimization",
    "architecture",
    "design",
    "differences",
    "from",
    "sense",
    "sensing",
    "charging",
    "battery",
    "cell",
    "circuit",
    "circuits",
    "project",
    "projects",
    "design",
    "selection",
    "select",
    "power",
    "supply",
    "current",
    "voltage",
    "capacitor",
    "resistor",
    "diode",
    "schematic",
    "electronics",
    "troubleshooting",
}

GENERIC_ACRONYM_TERMS = {
    "AC",
    "DC",
    "LED",
    "MOS",
    "FET",
    "MOSFET",
    "BJT",
    "IGBT",
    "PNP",
    "NPN",
    "PWM",
    "PFC",
    "SMPS",
    "PCB",
    "GND",
    "NTC",
    "PTC",
    "TVS",
    "MOV",
    "GDT",
    "EMC",
    "EMI",
}

GENERIC_MODEL_LIKE_TERMS = {
    "type-c",
    "usb-c",
}

LOW_VALUE_PROJECT_TITLES = (
    "lab supply2",
    "flexipower",
    "buck-boost multi-chemistry battery charger",
)

NOISE_MARKERS = (
    "object moved",
    "404 not found",
    "404 - file or directory not found",
    "server error in '/' application",
    "server error",
    "the resource cannot be found",
    "the resource you are looking for might have been removed",
    "configuration error",
    "temporarily unavailable",
    "access denied",
    "captcha",
    "recaptcha",
    "please log in",
    "sign in",
    "login",
    "redirect",
    "temporarily moved",
    "permanently moved",
    "add the following snippet to your html",
    "<iframe",
    "please wait",
    "/images/ajax-loader.gif",
    "images/ajax-loader.gif",
    "content/images/ajax-loader",
    "home * projects * hubs * components * forum",
    "copyright ©",
    "copyright (c)",
    "privacy policy",
    "cookie policy",
    "terms of use",
    "gnu general public license",
    "gnu library general public license",
    "gnu lesser general public license",
    "open source initiative",
    "do not show this message again",
    "circuitmaker-prod",
    "projectmetadata",
    "add new maker",
    "fabricate",
    "download files",
    "download files delete components files",
    "prev * * * * next",
    "community maker",
)

LOW_VALUE_SOURCE_MARKERS = (
    "circuitmaker.com/user/",
    "circuitmaker.com/users/",
    "circuitmaker.com/components/",
    "circuitmaker.com/forum/",
    "circuitmaker.com/stream",
)

CIRCUITMAKER_PROJECT_MARKERS = (
    "circuitmaker.com/projects/",
    "circuitmaker.com/projects/details/",
)

HIGH_VALUE_TEXT_MARKERS = (
    "datasheet",
    "application note",
    "app note",
    "schematic",
    "layout",
    "calculation",
    "equation",
    "formula",
    "failure",
    "fault",
    "debug",
    "troubleshoot",
    "measurement",
    "waveform",
    "scope",
    "引脚",
    "原理图",
    "布局",
    "公式",
    "计算",
    "故障",
    "异常",
    "波形",
)

ELECTRONIC_TERMS = (
    "mosfet",
    "bjt",
    "igbt",
    "opamp",
    "op-amp",
    "lm358",
    "tl431",
    "ntc",
    "ptc",
    "led",
    "pwm",
    "adc",
    "dac",
    "buck",
    "boost",
    "flyback",
    "charger",
    "battery",
    "current",
    "sense",
    "feedback",
    "gate",
    "source",
    "drain",
    "oscillator",
    "filter",
    "ground",
    "thermal",
    "电流",
    "电压",
    "反馈",
    "采样",
    "充电",
    "振荡",
    "滤波",
    "栅极",
    "运放",
)

TOPOLOGY_TERMS = {
    "buck",
    "boost",
    "flyback",
    "charger",
    "constant current",
    "current source",
    "rectifier",
    "bridge rectifier",
    "inrush",
    "soft start",
    "precharge",
    "feedback",
    "current sense",
    "op amp",
    "oscillator",
    "led driver",
    "power supply",
    "switching power supply",
    "mosfet",
    "op amp",
    "opamp",
    "tvs",
    "mov",
    "gdt",
}

FAULT_TERMS = {
    "burn",
    "burns",
    "burned",
    "overheat",
    "overheating",
    "ripple",
    "noise",
    "spike",
    "oscillation",
    "unstable",
    "short",
    "open",
    "inrush",
    "surge",
    "发热",
    "烧",
    "烧毁",
    "纹波",
    "噪声",
    "尖峰",
    "振荡",
    "短路",
    "开路",
}

CIRCUIT_DIAGNOSIS_CONTENT_MARKERS = (
    "故障分析",
    "异常原因",
    "失效机制",
    "失效分析",
    "烧毁原因",
    "过热原因",
    "不工作原因",
    "维修方法",
    "维修步骤",
    "排查步骤",
    "故障诊断",
    "反馈环路",
    "补偿计算",
    "恒流原理",
    "恒压原理",
    "反激原理",
    "开关电源原理",
    "BUCK电路",
    "BOOST电路",
    "LLC谐振",
    "选型计算",
    "参数计算",
    "NTC选型",
    "浪涌计算",
    "缓启动设计",
    "纹波计算",
    "尖峰抑制",
    "EMI滤波",
    "datasheet",
    "application note",
    "app note",
    "pin function",
    "typical application",
    "reference design",
    "evaluation board",
    "design guide",
    "TL431",
    "LM358",
    "UC3842",
    "SG3525",
    "IR2110",
    "电路原理",
    "电路分析",
    "电路设计",
    "波形分析",
    "电路故障",
)

ZH_EN_QUERY_EXPANSIONS = {
    "电流采样": ("current sense", "current sensing", "sense resistor", "shunt"),
    "采样电阻": ("sense resistor", "shunt resistor", "current sense"),
    "采样": ("sense", "sensing", "feedback"),
    "电流检测": ("current sense", "current monitor", "current measurement"),
    "运放": ("op amp", "opamp", "operational amplifier", "amplifier"),
    "放大器": ("amplifier", "op amp"),
    "负反馈": ("negative feedback", "feedback loop", "closed loop"),
    "反馈": ("feedback", "feedback loop"),
    "补偿": ("compensation", "loop compensation"),
    "振荡": ("oscillation", "oscillator", "unstable"),
    "震荡": ("oscillation", "oscillator", "unstable"),
    "噪声": ("noise", "ripple", "spike"),
    "纹波": ("ripple", "noise"),
    "滤波": ("filter", "filtering", "low pass"),
    "充电": ("charger", "charging", "battery charger"),
    "电池": ("battery", "cell"),
    "指示灯": ("LED", "status LED", "indicator"),
    "状态灯": ("status LED", "indicator", "LED"),
    "发光二极管": ("LED", "light emitting diode"),
    "浪涌": ("inrush", "surge", "inrush current"),
    "冲击电流": ("inrush current", "surge current"),
    "整流桥": ("bridge rectifier", "rectifier bridge"),
    "整流": ("rectifier", "rectification"),
    "保险丝": ("fuse", "fusing"),
    "热敏": ("NTC", "thermistor"),
    "缓启动": ("soft start", "soft-start", "startup"),
    "软启动": ("soft start", "soft-start", "startup"),
    "预充": ("precharge", "pre-charge"),
    "恒流": ("constant current", "current source"),
    "恒压": ("constant voltage", "voltage regulation"),
    "电源": ("power supply", "supply", "power"),
    "开关电源": ("switching power supply", "SMPS", "switch mode power supply"),
    "降压": ("buck", "step down"),
    "升压": ("boost", "step up"),
    "反激": ("flyback",),
    "栅极": ("gate", "gate drive"),
    "漏极": ("drain",),
    "源极": ("source",),
    "场效应管": ("MOSFET", "FET"),
    "三极管": ("transistor", "BJT"),
    "二极管": ("diode",),
    "稳压": ("regulator", "voltage regulation", "zener"),
    "过热": ("overheat", "overheating", "thermal"),
    "发热": ("heat", "heating", "thermal"),
    "短路": ("short circuit", "short"),
    "开路": ("open circuit", "open"),
    "接地": ("ground", "GND"),
    "地线": ("ground", "GND"),
    "上拉": ("pull up", "pull-up"),
    "下拉": ("pull down", "pull-down"),
}


@dataclass(frozen=True)
class CircuitMarkdownPage:
    path: Path
    title: str
    url: str
    published_at: str
    text: str
    text_hash: str


@dataclass(frozen=True)
class KbIndexStatus:
    index_dir: str
    db_path: str
    exists: bool
    readable: bool
    chunk_count: int
    error: str | None = None

    @property
    def usable(self) -> bool:
        return self.exists and self.readable and self.chunk_count > 0 and not self.error

    def to_json(self) -> dict[str, Any]:
        return {
            "index_dir": self.index_dir,
            "db_path": self.db_path,
            "exists": self.exists,
            "readable": self.readable,
            "chunk_count": self.chunk_count,
            "usable": self.usable,
            "error": self.error,
        }


def iter_markdown_files(root: Path, limit: int | None = None) -> Iterator[Path]:
    paths = sorted(root.glob("page_*.md"), key=_page_sort_key)
    yielded = 0
    for path in paths:
        if path.is_file():
            yield path
            yielded += 1
            if limit is not None and yielded >= limit:
                break


def iter_numbered_markdown_files(root: Path, *, max_page_num: int = 350000) -> Iterator[Path]:
    """Iterate page_N.md without listing huge single-directory corpora."""
    for index in range(1, max_page_num + 1):
        path = root / f"page_{index}.md"
        if path.is_file():
            yield path


def parse_markdown_page(path: Path) -> CircuitMarkdownPage:
    raw = path.read_text(encoding="utf-8", errors="replace")
    title = _extract_labeled_value(raw, "标题") or path.stem
    url = _extract_labeled_value(raw, "链接")
    published_at = _extract_labeled_value(raw, "发布时间") or "未知"
    body_lines: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if re.match(r"^\*\*(标题|链接|发布时间)\*\*\s*:", stripped):
            continue
        body_lines.append(line)
    text = "\n".join(line.strip() for line in body_lines if line.strip())
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    text_hash = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
    return CircuitMarkdownPage(
        path=path,
        title=compact_text(title, 300),
        url=url.strip(),
        published_at=published_at.strip(),
        text=text,
        text_hash=text_hash,
    )


def parse_markdown_metadata(path: Path, *, max_header_bytes: int = 8192) -> dict[str, str]:
    with path.open("rb") as f:
        raw = f.read(max_header_bytes)
    text = raw.decode("utf-8", errors="replace")
    return {
        "title": (_extract_labeled_value(text, "标题") or path.stem).strip(),
        "url": _extract_labeled_value(text, "链接").strip(),
        "published_at": (_extract_labeled_value(text, "发布时间") or "未知").strip(),
    }


def is_useful_page(page: CircuitMarkdownPage, min_chars: int = DEFAULT_MIN_CHUNK_CHARS) -> bool:
    text = " ".join(page.text.split())
    if len(text) < min_chars:
        return False
    lowered = f"{page.title}\n{page.url}\n{text}".lower()
    if is_low_value_source(page.url):
        return False
    noise_hits = sum(1 for marker in NOISE_MARKERS if marker in lowered)
    if noise_hits and len(text) < 1200:
        return False
    if is_boilerplate_text(f"{page.title}\n{text}"):
        return False
    alnum_or_cjk = sum(char.isalnum() or "\u4e00" <= char <= "\u9fff" for char in text)
    return alnum_or_cjk / max(len(text), 1) >= 0.35


def is_boilerplate_text(text: str) -> bool:
    lowered = " ".join((text or "").lower().split())
    if not lowered:
        return True
    marker_hits = sum(1 for marker in NOISE_MARKERS if marker in lowered)
    nav_or_legal = any(
        marker in lowered
        for marker in (
            "home * projects * hubs * components * forum",
            "/images/ajax-loader.gif",
            "gnu general public license",
            "gnu library general public license",
            "gnu lesser general public license",
            "privacy policy",
            "cookie policy",
            "do not show this message again",
            "add new maker",
            "download files delete components files",
        )
    )
    if nav_or_legal:
        high_value_hits = sum(1 for marker in HIGH_VALUE_TEXT_MARKERS if marker in lowered)
        technical_hits = sum(1 for term in ELECTRONIC_TERMS if term.lower() in lowered)
        if high_value_hits == 0 or technical_hits < 3:
            return True
    if marker_hits >= 2:
        return True
    if "add the following snippet to your html" in lowered and len(lowered) < 1800:
        return True
    technical_hits = sum(1 for term in ELECTRONIC_TERMS if term.lower() in lowered)
    alnum_or_cjk = sum(char.isalnum() or "\u4e00" <= char <= "\u9fff" for char in lowered)
    return marker_hits >= 1 and technical_hits == 0 and alnum_or_cjk / max(len(lowered), 1) < 0.65


def is_low_value_source(source: str) -> bool:
    lowered = (source or "").lower()
    return any(marker in lowered for marker in LOW_VALUE_SOURCE_MARKERS)


def is_deep_circuit_page(page: CircuitMarkdownPage, min_chars: int = DEFAULT_MIN_CHUNK_CHARS) -> bool:
    """Check if a page contains deep circuit analysis or diagnosis content."""
    lowered = f"{page.title}\n{page.text}".lower()
    diagnosis_hits = sum(1 for marker in CIRCUIT_DIAGNOSIS_CONTENT_MARKERS if marker.lower() in lowered)
    if diagnosis_hits >= 2:
        return True
    has_model = any(term.lower() in lowered for term in ELECTRONIC_TERMS[:20])
    has_circuit_context = any(term in lowered for term in ("schematic", "原理图", "电路图", "波形", "scope", "故障", "异常"))
    if has_model and has_circuit_context:
        return True
    return False


def is_circuitmaker_project_source(source: str) -> bool:
    lowered = (source or "").lower()
    return any(marker in lowered for marker in CIRCUITMAKER_PROJECT_MARKERS)


def is_public_tech_kb_page(page: CircuitMarkdownPage, min_chars: int = DEFAULT_MIN_CHUNK_CHARS) -> bool:
    title = (page.title or "").strip().lower()
    url = (page.url or "").strip()
    if not is_public_tech_source_allowed(title, url):
        return False
    if not is_useful_page(page, min_chars):
        return False
    cleaned = clean_public_tech_text(page.text)
    if len(cleaned) < min_chars:
        return False
    lowered = f"{page.title}\n{cleaned}".lower()
    technical_hits = sum(1 for term in ELECTRONIC_TERMS if term.lower() in lowered)
    technical_hits += sum(1 for term in PUBLIC_TECH_CONTENT_TERMS if term.lower() in lowered)
    high_value_hits = sum(1 for marker in HIGH_VALUE_TEXT_MARKERS if marker in lowered)
    return technical_hits >= 2 or high_value_hits >= 1


def is_public_tech_source_allowed(title: str, url: str) -> bool:
    title = (title or "").strip().lower()
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    if not host or host not in PUBLIC_TECH_ALLOWED_PATHS:
        return False
    if any(marker in title for marker in PUBLIC_TECH_REJECT_TITLES):
        return False
    if any(part in path for part in PUBLIC_TECH_REJECT_PATH_PARTS):
        return False
    if not any(path.startswith(prefix) or prefix in path for prefix in PUBLIC_TECH_ALLOWED_PATHS[host]):
        return False
    if host.endswith("elecfans.com") and not path.endswith(".html"):
        return False
    if "�" in title:
        return False
    return True


def clean_public_tech_text(text: str) -> str:
    article_started = False
    kept: list[str] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if _is_public_tech_article_heading(line):
            article_started = True
        elif not article_started and any(marker in line for marker in ("技术资料介绍", "资料介绍")):
            article_started = True
            continue
        if article_started and any(marker.lower() in lowered for marker in PUBLIC_TECH_SECTION_STOP_MARKERS):
            break
        if not article_started:
            continue
        if any(marker in lowered for marker in BOILERPLATE_LINE_MARKERS):
            continue
        if any(marker.lower() in lowered for marker in PUBLIC_TECH_PREARTICLE_MARKERS):
            continue
        if _is_public_tech_navigation_line(line):
            continue
        if line.startswith("!") or lowered.startswith("!["):
            continue
        if len(line) < 8 and not any(char.isdigit() for char in line):
            continue
        kept.append(line)
    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _is_public_tech_article_heading(line: str) -> bool:
    stripped = line.strip()
    heading_match = re.match(r"^(#{1,2})\s+(.+)$", stripped)
    return bool(heading_match and len(heading_match.group(2).strip()) >= 6)


def _is_public_tech_navigation_line(line: str) -> bool:
    stripped = line.strip()
    lowered = stripped.lower()
    if stripped.startswith("* ") and len(stripped) <= 24:
        return True
    if stripped.startswith("####") or stripped.startswith("#####"):
        return True
    if lowered.startswith("!["):
        return True
    return False


def _clean_public_tech_page(page: CircuitMarkdownPage) -> CircuitMarkdownPage:
    cleaned_text = clean_public_tech_text(page.text)
    text_hash = hashlib.sha1(cleaned_text.encode("utf-8", errors="ignore")).hexdigest()
    return CircuitMarkdownPage(
        path=page.path,
        title=page.title,
        url=page.url,
        published_at=page.published_at,
        text=cleaned_text,
        text_hash=text_hash,
    )


def chunk_text(
    text: str,
    *,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_chars: int = DEFAULT_MIN_CHUNK_CHARS,
) -> list[tuple[str, int, int]]:
    normalized = "\n".join(line.strip() for line in (text or "").splitlines() if line.strip())
    if len(normalized) < min_chars:
        return []
    if len(normalized) <= chunk_chars:
        return [(normalized, 0, len(normalized))]
    chunks: list[tuple[str, int, int]] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + chunk_chars)
        if end < len(normalized):
            boundary = max(
                normalized.rfind("\n", start, end),
                normalized.rfind(". ", start, end),
                normalized.rfind("。", start, end),
                normalized.rfind("；", start, end),
            )
            if boundary > start + chunk_chars // 2:
                end = boundary + 1
        chunk = normalized[start:end].strip()
        if len(chunk) >= min_chars:
            chunks.append((chunk, start, end))
        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_circuit_md_kb(
    source_dir: Path,
    output_dir: Path,
    *,
    max_docs: int = DEFAULT_MAX_DOCS,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / DB_FILENAME
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    try:
        _init_schema(conn)
        stats: Counter[str] = Counter()
        seen_hashes: set[str] = set()
        seen_urls: set[str] = set()
        docs = 0
        chunks = 0
        with conn:
            for path in iter_markdown_files(source_dir):
                stats["candidate_files"] += 1
                try:
                    page = parse_markdown_page(path)
                except OSError:
                    stats["read_errors"] += 1
                    continue
                if not is_useful_page(page, min_chunk_chars):
                    stats["filtered_noise_or_short"] += 1
                    continue
                dedupe_key = page.url or page.text_hash
                if dedupe_key in seen_urls or page.text_hash in seen_hashes:
                    stats["duplicates"] += 1
                    continue
                page_chunks = chunk_text(
                    page.text,
                    chunk_chars=chunk_chars,
                    overlap=chunk_overlap,
                    min_chars=min_chunk_chars,
                )
                if not page_chunks:
                    stats["no_chunks"] += 1
                    continue
                page_id = _insert_page(conn, page)
                seen_urls.add(dedupe_key)
                seen_hashes.add(page.text_hash)
                docs += 1
                for index, (chunk, char_start, char_end) in enumerate(page_chunks):
                    chunk_id = _insert_chunk(conn, page_id, index, page, chunk, char_start, char_end)
                    conn.execute(
                        "INSERT INTO chunks_fts(rowid, title, url, text) VALUES (?, ?, ?, ?)",
                        (chunk_id, page.title, page.url, chunk),
                    )
                    chunks += 1
                if docs >= max_docs:
                    break
        conn.execute("INSERT INTO kb_meta(key, value) VALUES (?, ?)", ("tokenizer", "trigram"))
        conn.execute("INSERT INTO kb_meta(key, value) VALUES (?, ?)", ("max_docs", str(max_docs)))
        conn.commit()
    finally:
        conn.close()

    meta = {
        "source_dir": str(source_dir),
        "db_path": str(db_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "max_docs": max_docs,
        "chunk_chars": chunk_chars,
        "chunk_overlap": chunk_overlap,
        "min_chunk_chars": min_chunk_chars,
        "documents": docs,
        "chunks": chunks,
        "stats": dict(stats),
    }
    (output_dir / "build_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def build_filtered_public_kb(
    source_dirs: Iterable[Path | str],
    output_dir: Path | str,
    *,
    max_docs: int = DEFAULT_MAX_DOCS,
    max_page_num: int = 350000,
    max_file_bytes: int = DEFAULT_MAX_SOURCE_FILE_BYTES,
    progress_every: int = 0,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS,
) -> dict[str, Any]:
    """Build a high-precision KB from public electronics article sources."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    db_path = output_path / DB_FILENAME
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    sources = [Path(path) for path in source_dirs]
    stats: Counter[str] = Counter()
    seen_hashes: set[str] = set()
    seen_urls: set[str] = set()
    docs = 0
    chunks = 0
    try:
        _init_schema(conn)
        with conn:
            for source_dir in sources:
                if not source_dir.exists():
                    stats["missing_source_dirs"] += 1
                    continue
                for path in iter_numbered_markdown_files(source_dir, max_page_num=max_page_num):
                    stats["candidate_files"] += 1
                    if progress_every > 0 and stats["candidate_files"] % progress_every == 0:
                        print(
                            f"scanned={stats['candidate_files']} docs={docs} chunks={chunks} "
                            f"filtered={stats['filtered_not_public_tech']} oversized={stats['oversized_files']}",
                            flush=True,
                        )
                    try:
                        if max_file_bytes > 0 and path.stat().st_size > max_file_bytes:
                            stats["oversized_files"] += 1
                            continue
                    except OSError:
                        stats["read_errors"] += 1
                        continue
                    try:
                        metadata = parse_markdown_metadata(path)
                    except OSError:
                        stats["read_errors"] += 1
                        continue
                    if not is_public_tech_source_allowed(metadata["title"], metadata["url"]):
                        stats["filtered_not_public_tech"] += 1
                        continue
                    try:
                        page = parse_markdown_page(path)
                    except OSError:
                        stats["read_errors"] += 1
                        continue
                    if not is_public_tech_kb_page(page, min_chunk_chars):
                        stats["filtered_not_public_tech"] += 1
                        continue
                    cleaned_page = _clean_public_tech_page(page)
                    if not is_useful_page(cleaned_page, min_chunk_chars):
                        stats["filtered_noise_or_short"] += 1
                        continue
                    dedupe_key = cleaned_page.url or cleaned_page.text_hash
                    if dedupe_key in seen_urls or cleaned_page.text_hash in seen_hashes:
                        stats["duplicates"] += 1
                        continue
                    page_chunks = chunk_text(
                        cleaned_page.text,
                        chunk_chars=chunk_chars,
                        overlap=chunk_overlap,
                        min_chars=min_chunk_chars,
                    )
                    if not page_chunks:
                        stats["no_chunks"] += 1
                        continue
                    page_id = _insert_page(conn, cleaned_page)
                    seen_urls.add(dedupe_key)
                    seen_hashes.add(cleaned_page.text_hash)
                    docs += 1
                    for index, (chunk, char_start, char_end) in enumerate(page_chunks):
                        chunk_id = _insert_chunk(conn, page_id, index, cleaned_page, chunk, char_start, char_end)
                        conn.execute(
                            "INSERT INTO chunks_fts(rowid, title, url, text) VALUES (?, ?, ?, ?)",
                            (chunk_id, cleaned_page.title, cleaned_page.url, chunk),
                        )
                        chunks += 1
                    if docs >= max_docs:
                        break
                if docs >= max_docs:
                    break
        conn.execute("INSERT INTO kb_meta(key, value) VALUES (?, ?)", ("tokenizer", "trigram"))
        conn.execute("INSERT INTO kb_meta(key, value) VALUES (?, ?)", ("builder", "filtered_public"))
        conn.execute("INSERT INTO kb_meta(key, value) VALUES (?, ?)", ("max_docs", str(max_docs)))
        conn.commit()
    finally:
        conn.close()
    meta = {
        "source_dirs": [str(path) for path in sources],
        "db_path": str(db_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "max_docs": max_docs,
        "max_page_num": max_page_num,
        "max_file_bytes": max_file_bytes,
        "chunk_chars": chunk_chars,
        "chunk_overlap": chunk_overlap,
        "min_chunk_chars": min_chunk_chars,
        "documents": docs,
        "chunks": chunks,
        "stats": dict(stats),
        "allowed_hosts": sorted(PUBLIC_TECH_ALLOWED_PATHS),
    }
    (output_path / "build_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def build_diagnosis_kb(
    source_dirs: Iterable[Path | str],
    output_dir: Path | str,
    *,
    max_docs: int = DEFAULT_MAX_DOCS,
    max_file_bytes: int = DEFAULT_MAX_SOURCE_FILE_BYTES,
    progress_every: int = 0,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS,
) -> dict[str, Any]:
    """Build a high-precision KB focused on circuit fault diagnosis and deep analysis content."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    db_path = output_path / DB_FILENAME
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    sources = [Path(path) for path in source_dirs]
    stats: Counter[str] = Counter()
    seen_hashes: set[str] = set()
    seen_urls: set[str] = set()
    docs = 0
    chunks = 0
    try:
        _init_schema(conn)
        with conn:
            for source_dir in sources:
                if not source_dir.exists():
                    stats["missing_source_dirs"] += 1
                    continue
                # Phase 1: fast metadata scan to filter candidates
                # Use unsorted glob for speed on huge directories (226K+ files)
                candidate_paths: list[Path] = []
                stats["phase1_scanned"] = 0
                for path in source_dir.glob("page_*.md"):
                    stats["phase1_scanned"] += 1
                    if progress_every > 0 and stats["phase1_scanned"] % 2000 == 0:
                        print(
                            f"phase1: scanned={stats['phase1_scanned']} candidates={len(candidate_paths)} "
                            f"docs={docs} chunks={chunks}",
                            flush=True,
                        )
                    if not path.is_file():
                        continue
                    try:
                        if max_file_bytes > 0 and path.stat().st_size > max_file_bytes:
                            stats["oversized_files"] += 1
                            continue
                    except OSError:
                        stats["read_errors"] += 1
                        continue
                    try:
                        metadata = parse_markdown_metadata(path)
                    except OSError:
                        stats["read_errors"] += 1
                        continue
                    # Fast pre-filter: reject obvious junk by URL/title alone
                    title = metadata["title"].strip().lower()
                    url = metadata["url"].strip()
                    if is_low_value_source(url):
                        stats["phase1_filtered_low_value"] += 1
                        continue
                    if any(marker in title for marker in PUBLIC_TECH_REJECT_TITLES):
                        stats["phase1_filtered_title"] += 1
                        continue
                    # For public tech sources, check URL path eligibility early
                    if not is_public_tech_source_allowed(title, url):
                        # Not an allowed public tech source — check if URL suggests circuit content
                        lowered_meta = f"{title}\n{url}".lower()
                        if not any(term.lower() in lowered_meta for term in CIRCUIT_DIAGNOSIS_CONTENT_MARKERS[:10]):
                            stats["phase1_filtered_no_diagnosis_markers"] += 1
                            continue
                    candidate_paths.append(path)
                if progress_every > 0:
                    print(
                        f"phase1 done for {source_dir.name}: "
                        f"scanned={stats['phase1_scanned']} candidates={len(candidate_paths)}",
                        flush=True,
                    )

                # Phase 2: full parse + deep content filtering on candidates only
                stats["phase2_scanned"] = 0
                for path in candidate_paths:
                    stats["phase2_scanned"] += 1
                    if progress_every > 0 and stats["phase2_scanned"] % 500 == 0:
                        print(
                            f"phase2: processed={stats['phase2_scanned']}/{len(candidate_paths)} "
                            f"docs={docs} chunks={chunks} "
                            f"filtered={stats['filtered_not_deep_circuit']}",
                            flush=True,
                        )
                    # For Chinese sources (elecfans, eet-china), use public tech source filtering
                    metadata = parse_markdown_metadata(path)
                    if is_public_tech_source_allowed(metadata["title"], metadata["url"]):
                        try:
                            page = parse_markdown_page(path)
                        except OSError:
                            stats["read_errors"] += 1
                            continue
                        if not is_public_tech_kb_page(page, min_chunk_chars):
                            stats["filtered_not_deep_circuit"] += 1
                            continue
                        cleaned_page = _clean_public_tech_page(page)
                        if not is_deep_circuit_page(cleaned_page, min_chunk_chars):
                            stats["filtered_not_deep_circuit"] += 1
                            continue
                        if not is_useful_page(cleaned_page, min_chunk_chars):
                            stats["filtered_noise_or_short"] += 1
                            continue
                        page_to_index = cleaned_page
                    else:
                        # For non-Chinese-allowed sources, use general useful + deep circuit checks
                        try:
                            page = parse_markdown_page(path)
                        except OSError:
                            stats["read_errors"] += 1
                            continue
                        if not is_useful_page(page, min_chunk_chars):
                            stats["filtered_not_deep_circuit"] += 1
                            continue
                        if not is_deep_circuit_page(page, min_chunk_chars):
                            stats["filtered_not_deep_circuit"] += 1
                            continue
                        page_to_index = page
                    dedupe_key = page_to_index.url or page_to_index.text_hash
                    if dedupe_key in seen_urls or page_to_index.text_hash in seen_hashes:
                        stats["duplicates"] += 1
                        continue
                    page_chunks = chunk_text(
                        page_to_index.text,
                        chunk_chars=chunk_chars,
                        overlap=chunk_overlap,
                        min_chars=min_chunk_chars,
                    )
                    if not page_chunks:
                        stats["no_chunks"] += 1
                        continue
                    page_id = _insert_page(conn, page_to_index)
                    seen_urls.add(dedupe_key)
                    seen_hashes.add(page_to_index.text_hash)
                    docs += 1
                    for index, (chunk, char_start, char_end) in enumerate(page_chunks):
                        chunk_id = _insert_chunk(conn, page_id, index, page_to_index, chunk, char_start, char_end)
                        conn.execute(
                            "INSERT INTO chunks_fts(rowid, title, url, text) VALUES (?, ?, ?, ?)",
                            (chunk_id, page_to_index.title, page_to_index.url, chunk),
                        )
                        chunks += 1
                    if docs >= max_docs:
                        break
                if docs >= max_docs:
                    break
        conn.execute("INSERT INTO kb_meta(key, value) VALUES (?, ?)", ("tokenizer", "trigram"))
        conn.execute("INSERT INTO kb_meta(key, value) VALUES (?, ?)", ("builder", "diagnosis"))
        conn.execute("INSERT INTO kb_meta(key, value) VALUES (?, ?)", ("max_docs", str(max_docs)))
        conn.commit()
    finally:
        conn.close()
    meta = {
        "source_dirs": [str(path) for path in sources],
        "db_path": str(db_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "max_docs": max_docs,
        "max_file_bytes": max_file_bytes,
        "chunk_chars": chunk_chars,
        "chunk_overlap": chunk_overlap,
        "min_chunk_chars": min_chunk_chars,
        "documents": docs,
        "chunks": chunks,
        "stats": dict(stats),
    }
    (output_path / "build_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


class CircuitMarkdownRetriever:
    def __init__(
        self,
        index_dir: Path | str,
        *,
        candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
        max_chunks_per_url: int = 2,
        min_relevance_score: float = DEFAULT_MIN_RELEVANCE_SCORE,
        dense_weight: float = 0.65,
        sparse_weight: float = 0.35,
        dense_retriever: Any | None = None,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.db_path = self.index_dir / DB_FILENAME
        self.candidate_limit = candidate_limit
        self.max_chunks_per_url = max_chunks_per_url
        self.min_relevance_score = min_relevance_score
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self._dense_retriever = dense_retriever

    def status(self) -> KbIndexStatus:
        if not self.db_path.exists():
            return KbIndexStatus(
                index_dir=str(self.index_dir),
                db_path=str(self.db_path),
                exists=False,
                readable=False,
                chunk_count=0,
                error=f"missing {DB_FILENAME}",
            )
        try:
            with sqlite3.connect(self.db_path) as conn:
                chunk_count = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
                conn.execute("SELECT 1 FROM chunks_fts LIMIT 1").fetchone()
        except sqlite3.Error as exc:
            return KbIndexStatus(
                index_dir=str(self.index_dir),
                db_path=str(self.db_path),
                exists=True,
                readable=False,
                chunk_count=0,
                error=str(exc),
            )
        return KbIndexStatus(
            index_dir=str(self.index_dir),
            db_path=str(self.db_path),
            exists=True,
            readable=True,
            chunk_count=chunk_count,
            error=None if chunk_count > 0 else "empty chunks table",
        )

    def search(self, query: str, limit: int = 4) -> list[Evidence]:
        result = self.search_with_diagnostics(query, limit=limit)
        return result["evidence"]

    def search_with_diagnostics(self, query: str, limit: int = 4) -> dict[str, Any]:
        query = (query or "").strip()
        if not query:
            return _empty_search_diagnostics(query, limit, "empty_query")
        fts_query = build_fts_query(query)
        if not fts_query and self._dense_retriever is None:
            return _empty_search_diagnostics(query, limit, "empty_fts_query")
        status = self.status()
        if not status.usable:
            result = _empty_search_diagnostics(query, limit, "index_unusable")
            result["index_status"] = status.to_json()
            return result

        # Dense search results (chunk_id -> normalized score 0-1)
        dense_scores: dict[int, float] = {}
        if self._dense_retriever is not None:
            self._dense_retriever.load_index()
            dense_results = self._dense_retriever.search(query, limit=self.candidate_limit)
            for score, chunk_id in dense_results:
                dense_scores[chunk_id] = max(0.0, min(1.0, score))

        # Sparse search (BM25 + rerank)
        sparse_rows: list[sqlite3.Row] = []
        if fts_query:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    rows = self._search_fts(conn, fts_query, self.candidate_limit)
                    if not rows:
                        rows = self._search_like(conn, query, self.candidate_limit)
                    sparse_rows = rows
            except sqlite3.Error:
                sparse_rows = []
        ranked = rerank_rows(query, sparse_rows)

        # Normalize sparse scores to 0-1 range for hybrid merge
        max_sparse = max((score for score, _ in ranked), default=self.min_relevance_score)
        if max_sparse <= 0:
            max_sparse = self.min_relevance_score

        # Merge dense and sparse: collect all candidate chunk_ids with hybrid scores
        candidate_chunks: dict[int, float] = {}  # chunk_id -> hybrid_score
        for score, row in ranked:
            chunk_id = int(row["chunk_id"])
            sparse_norm = max(0.0, min(1.0, score / max(max_sparse, 1.0)))
            dense_norm = dense_scores.get(chunk_id, 0.0)
            if self._dense_retriever is not None and dense_scores:
                hybrid = self.dense_weight * dense_norm + self.sparse_weight * sparse_norm
            else:
                hybrid = score  # pure sparse mode, keep original rerank score
            candidate_chunks[chunk_id] = hybrid

        # Add dense-only hits (chunks not in sparse results but semantically relevant)
        if self._dense_retriever is not None and dense_scores:
            for chunk_id, dense_norm in dense_scores.items():
                if chunk_id not in candidate_chunks and dense_norm >= 0.3:
                    candidate_chunks[chunk_id] = self.dense_weight * dense_norm

        # Fetch full chunk rows for candidates and apply quality filters
        if candidate_chunks:
            chunk_id_list = sorted(candidate_chunks.keys())
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    placeholders = ",".join("?" for _ in chunk_id_list)
                    rows = list(conn.execute(f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", chunk_id_list))
            except sqlite3.Error:
                rows = []
        else:
            rows = sparse_rows

        # Re-score rows using hybrid scores and apply quality filters
        row_by_id = {int(row["chunk_id"]): row for row in rows}
        kept: list[Evidence] = []
        per_url: Counter[str] = Counter()
        query_profile = classify_query_terms(query)
        diagnostics: dict[str, Any] = {
            "query": query,
            "limit": limit,
            "fts_query": fts_query,
            "candidate_count": len(sparse_rows),
            "dense_candidate_count": len(dense_scores),
            "merged_candidate_count": len(candidate_chunks),
            "ranked_count": len(ranked),
            "kept_count": 0,
            "discarded_low_relevance": 0,
            "discarded_noise": 0,
            "discarded_low_value_source": 0,
            "discarded_low_value_project": 0,
            "discarded_required_terms": 0,
            "discarded_duplicate_source": 0,
            "high_relevance_count": 0,
            "discarded_kb": 0,
            "retrieval_mode": "hybrid" if (self._dense_retriever is not None and dense_scores) else "sparse",
        }

        # Sort candidates by hybrid score
        sorted_candidates = sorted(candidate_chunks.items(), key=lambda item: item[1], reverse=True)
        for chunk_id, hybrid_score in sorted_candidates:
            row = row_by_id.get(chunk_id)
            if row is None:
                continue
            if is_low_value_source(str(row["url"] or row["path"] or "")):
                diagnostics["discarded_low_value_source"] += 1
                continue
            if is_boilerplate_text(f"{row['title']}\n{row['url']}\n{row['text']}"):
                diagnostics["discarded_noise"] += 1
                continue
            if is_low_value_project_row(query_profile, row):
                diagnostics["discarded_low_value_project"] += 1
                continue
            # In hybrid mode, skip required-terms filter for dense hits above threshold
            # (dense retrieval already validates semantic relevance)
            dense_hit_score = dense_scores.get(chunk_id, 0.0)
            is_hybrid = self._dense_retriever is not None and dense_scores
            if not (is_hybrid and dense_hit_score >= 0.3) and not _row_matches_required_terms(query_profile, row):
                diagnostics["discarded_required_terms"] += 1
                continue
            url_key = str(row["url"] or row["path"] or row["page_id"])
            if per_url[url_key] >= self.max_chunks_per_url:
                diagnostics["discarded_duplicate_source"] += 1
                continue
            per_url[url_key] += 1
            # For pure sparse mode, use original rerank score; for hybrid, use hybrid score
            final_score = hybrid_score
            if self._dense_retriever is None or not dense_scores:
                if final_score < self.min_relevance_score:
                    diagnostics["discarded_low_relevance"] += 1
                    continue
            elif hybrid_score < 0.15:
                diagnostics["discarded_low_relevance"] += 1
                continue
            high_relevance = final_score >= self.min_relevance_score + 2.0 if (self._dense_retriever is None or not dense_scores) else hybrid_score >= 0.6
            evidence = self._to_evidence(row, final_score, query_profile=query_profile, high_relevance=high_relevance)
            kept.append(evidence)
            if high_relevance:
                diagnostics["high_relevance_count"] += 1
            if len(kept) >= limit:
                break
        diagnostics["kept_count"] = len(kept)
        diagnostics["discarded_kb"] = (
            diagnostics["discarded_low_relevance"]
            + diagnostics["discarded_noise"]
            + diagnostics["discarded_low_value_source"]
            + diagnostics["discarded_low_value_project"]
            + diagnostics["discarded_required_terms"]
            + diagnostics["discarded_duplicate_source"]
        )
        diagnostics["high_relevance_rate"] = round(diagnostics["high_relevance_count"] / len(kept), 4) if kept else 0.0
        return {"evidence": kept, "diagnostics": diagnostics}

    def _search_fts(self, conn: sqlite3.Connection, fts_query: str, limit: int) -> list[sqlite3.Row]:
        return list(
            conn.execute(
                """
                SELECT
                    c.chunk_id,
                    c.page_id,
                    c.chunk_index,
                    c.title,
                    c.url,
                    c.path,
                    c.text,
                    c.char_start,
                    c.char_end,
                    bm25(chunks_fts) AS bm25_score
                FROM chunks_fts
                JOIN chunks c ON c.chunk_id = chunks_fts.rowid
                WHERE chunks_fts MATCH ?
                ORDER BY bm25_score
                LIMIT ?
                """,
                (fts_query, limit),
            )
        )

    def _search_like(self, conn: sqlite3.Connection, query: str, limit: int) -> list[sqlite3.Row]:
        terms = _drop_broad_terms(extract_query_terms(query))[:4]
        if not terms:
            return []
        where = " OR ".join(["c.title LIKE ? OR c.text LIKE ?" for _ in terms])
        params: list[Any] = []
        for term in terms:
            like = f"%{term}%"
            params.extend([like, like])
        params.append(limit)
        return list(
            conn.execute(
                f"""
                SELECT
                    c.chunk_id,
                    c.page_id,
                    c.chunk_index,
                    c.title,
                    c.url,
                    c.path,
                    c.text,
                    c.char_start,
                    c.char_end,
                    10.0 AS bm25_score
                FROM chunks c
                WHERE {where}
                LIMIT ?
                """,
                params,
            )
        )

    def _to_evidence(
        self,
        row: sqlite3.Row,
        score: float,
        query_profile: dict[str, list[str]] | None = None,
        high_relevance: bool = False,
    ) -> Evidence:
        bm25_score = float(row["bm25_score"]) if "bm25_score" in row.keys() else 0.0
        profile = query_profile or {}
        text = f"{row['title']} {row['url']} {row['text']}".lower()
        matched_terms = {
            key: [term for term in values if term.lower() in text]
            for key, values in profile.items()
            if isinstance(values, list)
        }
        metadata = {
            "kind": "local_kb_chunk",
            "chunk_id": int(row["chunk_id"]),
            "page_id": int(row["page_id"]),
            "chunk_index": int(row["chunk_index"]),
            "path": str(row["path"] or ""),
            "bm25_score": round(bm25_score, 6),
            "rerank_score": round(float(score), 4),
            "kb_relevance": round(float(score), 4),
            "high_relevance": bool(high_relevance),
            "matched_query_terms": matched_terms,
            "char_start": int(row["char_start"] or 0),
            "char_end": int(row["char_end"] or 0),
        }
        return Evidence(
            source=str(row["url"] or row["path"] or f"circuit_md:{row['page_id']}"),
            title=str(row["title"] or row["url"] or "Circuit Markdown KB"),
            content=compact_text(str(row["text"] or ""), 4000),
            score=round(float(score), 4),
            metadata=metadata,
        )


def build_fts_query(query: str, max_terms: int = 12) -> str:
    profile = classify_query_terms(query)
    ordered = [
        *profile["models"],
        *profile["refdes"],
        *profile["values"],
        *profile["topology"],
        *profile["fault"],
        *profile["other"][:4],
    ]
    ordered = _drop_broad_terms(ordered)
    terms = list(dict.fromkeys(ordered))
    if not terms:
        return ""
    strong_terms = profile["models"] + profile["refdes"] + profile["values"]
    if strong_terms:
        return " OR ".join(_quote_fts(term) for term in strong_terms[:6])
    return " OR ".join(_quote_fts(term) for term in terms[:max_terms])


def classify_query_terms(query: str) -> dict[str, list[str]]:
    terms = extract_query_terms(query)
    profile: dict[str, list[str]] = {
        "models": [],
        "refdes": [],
        "values": [],
        "topology": [],
        "fault": [],
        "other": [],
    }
    for term in terms:
        lowered = term.lower()
        if re.match(r"^[rcldqutjf]\d+[a-z0-9_.+\-]*$", lowered):
            profile["refdes"].append(term)
        elif re.match(r"^\d+(\.\d+)?\s*(r|k|m|ohm|v|a|ma|ua|w|uf|nf|pf|mh|uh|hz|khz|mhz)$", lowered):
            profile["values"].append(term)
        elif re.match(r"^\d+(\.\d+)?(r|k|m|ohm|v|a|ma|ua|w|uf|nf|pf|mh|uh|hz|khz|mhz)$", lowered):
            profile["values"].append(term)
        elif _is_model_like(term):
            profile["models"].append(term)
        elif lowered in TOPOLOGY_TERMS:
            profile["topology"].append(term)
        elif lowered in FAULT_TERMS:
            profile["fault"].append(term)
        else:
            profile["other"].append(term)
    return profile


def _is_model_like(term: str) -> bool:
    raw = (term or "").strip()
    lowered = raw.lower()
    upper = raw.upper()
    if lowered in GENERIC_MODEL_LIKE_TERMS:
        return False
    if upper in GENERIC_ACRONYM_TERMS:
        return False
    if re.match(r"^[a-z]{2,8}\d{2,}[a-z0-9_.+\-]*$", lowered):
        return True
    if re.match(r"^\d+[a-z]{1,}[a-z0-9_.+\-]*\d*[a-z0-9_.+\-]*$", lowered):
        return True
    if re.match(r"^[a-z]{2,}\d+[a-z0-9_.+\-]*$", lowered):
        return True
    if re.match(r"^[A-Z][A-Za-z0-9_.+\-]{3,}$", raw) and any(char.isupper() for char in raw[1:]):
        return True
    if re.match(r"^[A-Z]{3,8}$", raw) and upper not in GENERIC_ACRONYM_TERMS:
        return True
    return False


def _drop_broad_terms(terms: list[str], *, keep_if_only: bool = True) -> list[str]:
    specific = [term for term in terms if term.lower() not in BROAD_QUERY_TERMS]
    if specific or not keep_if_only:
        return specific
    return terms


def extract_query_terms(query: str) -> list[str]:
    parts = re.findall(r"[A-Za-z0-9_./+\-#%℃Ωμu\u4e00-\u9fff]{2,}", query or "")
    tokens: list[str] = []
    seen: set[str] = set()
    for part in parts:
        cleaned = part.strip("-_./+ ")
        if len(cleaned) < 2:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        tokens.append(cleaned)
    for token in tokenize(query):
        if len(token) >= 2 and token.lower() not in seen:
            seen.add(token.lower())
            tokens.append(token)
    for expanded in expand_chinese_electronics_terms(query):
        key = expanded.lower()
        if key not in seen:
            seen.add(key)
            tokens.append(expanded)
    return tokens


def expand_chinese_electronics_terms(query: str) -> list[str]:
    expanded: list[str] = []
    text = query or ""
    for zh_term, en_terms in ZH_EN_QUERY_EXPANSIONS.items():
        if zh_term in text:
            expanded.extend(en_terms)
    return expanded


def rerank_rows(query: str, rows: Iterable[sqlite3.Row]) -> list[tuple[float, sqlite3.Row]]:
    query_profile = classify_query_terms(query)
    query_terms = [
        *query_profile["models"],
        *query_profile["refdes"],
        *query_profile["values"],
        *query_profile["topology"],
        *query_profile["fault"],
        *query_profile["other"],
    ]
    query_tokens = {term.lower() for term in _drop_broad_terms(query_terms)}
    ranked: list[tuple[float, sqlite3.Row]] = []
    for row in rows:
        bm25_score = float(row["bm25_score"])
        score = 1.0 / (1.0 + abs(bm25_score))
        title_url = f"{row['title']} {row['url']}".lower()
        text = str(row["text"] or "")
        lowered = f"{title_url} {text.lower()}"
        title = str(row["title"] or "").lower()
        if is_boilerplate_text(lowered):
            score -= 8.0
        if is_low_value_source(str(row["url"] or row["path"] or "")):
            score -= 8.0
        if is_circuitmaker_project_source(str(row["url"] or row["path"] or "")):
            score -= 2.5
            if any(marker in title for marker in LOW_VALUE_PROJECT_TITLES):
                score -= 4.0
            if _technical_content_score(lowered) < 2:
                score -= 2.0
        for term in query_tokens:
            if term in title_url:
                score += 3.0
            elif term in lowered:
                score += 1.0
        for term in (token.lower() for token in query_profile["models"]):
            score += 5.0 if term in lowered else -1.5
        for term in (token.lower() for token in query_profile["refdes"]):
            if term in lowered:
                score += 2.0
        for term in (token.lower() for token in query_profile["topology"]):
            if term in title_url:
                score += 2.0
            elif term in lowered:
                score += 1.0
        for term in (token.lower() for token in query_profile["fault"]):
            if term in lowered:
                score += 1.2
        high_value_match_count = _high_value_match_count(query_profile, lowered)
        score += 1.0 * high_value_match_count
        if not query_profile["models"] and _has_high_value_query(query_profile) and high_value_match_count < 2:
            score -= 2.0
        if any(marker in lowered for marker in HIGH_VALUE_TEXT_MARKERS):
            score += 0.8
        for term in ELECTRONIC_TERMS:
            if term.lower() in lowered and (term.lower() in query.lower() or term.lower() in title_url):
                score += 0.6
        score += 1.2 * len(_component_like_terms(query_tokens).intersection(_component_like_terms(set(re.findall(r"[A-Za-z0-9_.+\-]+", lowered)))))
        if any(marker in lowered for marker in NOISE_MARKERS):
            score -= 3.0
        if len(text) < DEFAULT_MIN_CHUNK_CHARS:
            score -= 2.0
        ranked.append((score, row))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def _row_matches_required_terms(profile: dict[str, list[str]], row: sqlite3.Row) -> bool:
    text = f"{row['title']} {row['url']} {row['text']}".lower()
    strong_terms = [
        term.lower()
        for term in [*profile.get("models", []), *profile.get("refdes", []), *profile.get("values", [])]
    ]
    if strong_terms and not any(term in text for term in strong_terms):
        return False
    if strong_terms:
        return True
    if _has_high_value_query(profile):
        return _high_value_match_count(profile, text) >= 2
    return True


def is_low_value_project_row(profile: dict[str, list[str]], row: sqlite3.Row) -> bool:
    source = str(row["url"] or row["path"] or "")
    if not is_circuitmaker_project_source(source):
        return False
    text = f"{row['title']} {source} {row['text']}".lower()
    title = str(row["title"] or "").lower()
    if any(marker in title for marker in LOW_VALUE_PROJECT_TITLES):
        return True
    strong_terms = [term.lower() for term in [*profile.get("models", []), *profile.get("refdes", []), *profile.get("values", [])]]
    strong_match = bool(strong_terms) and any(term in text for term in strong_terms)
    if strong_match and _technical_content_score(text) >= 2:
        return False
    if not strong_terms and _high_value_match_count(profile, text) >= 2 and _technical_content_score(text) >= 3:
        return False
    return True


def _has_high_value_query(profile: dict[str, list[str]]) -> bool:
    return any(profile.get(key) for key in ("refdes", "values", "topology", "fault"))


def _high_value_match_count(profile: dict[str, list[str]], text: str) -> int:
    lowered = text.lower()
    matched: set[str] = set()
    for key in ("refdes", "values", "topology", "fault"):
        for term in profile.get(key, []):
            normalized = str(term).lower()
            if normalized and normalized in lowered:
                matched.add(normalized)
    return len(matched)


def _technical_content_score(text: str) -> int:
    lowered = text.lower()
    score = 0
    score += min(3, sum(1 for marker in HIGH_VALUE_TEXT_MARKERS if marker in lowered))
    score += min(3, sum(1 for term in ELECTRONIC_TERMS if term.lower() in lowered))
    if re.search(r"\b(r|c|l|d|q|u|j|t)\d+\b", lowered):
        score += 1
    if re.search(r"\d+(\.\d+)?\s*(v|a|ma|ua|w|uf|nf|pf|ohm|khz|mhz)\b", lowered):
        score += 1
    return score


def _empty_search_diagnostics(query: str, limit: int, reason: str) -> dict[str, Any]:
    return {
        "evidence": [],
        "diagnostics": {
            "query": query,
            "limit": limit,
            "reason": reason,
            "candidate_count": 0,
            "ranked_count": 0,
            "kept_count": 0,
            "discarded_low_relevance": 0,
            "discarded_noise": 0,
            "discarded_low_value_source": 0,
            "discarded_low_value_project": 0,
            "discarded_required_terms": 0,
            "discarded_duplicate_source": 0,
            "discarded_kb": 0,
            "high_relevance_count": 0,
            "high_relevance_rate": 0.0,
        },
    }


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE pages(
            page_id INTEGER PRIMARY KEY,
            path TEXT UNIQUE NOT NULL,
            title TEXT,
            url TEXT,
            published_at TEXT,
            text TEXT,
            text_hash TEXT
        )
        """
    )
    conn.execute("CREATE UNIQUE INDEX idx_pages_text_hash ON pages(text_hash)")
    conn.execute(
        """
        CREATE TABLE chunks(
            chunk_id INTEGER PRIMARY KEY,
            page_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            title TEXT,
            url TEXT,
            path TEXT,
            text TEXT,
            char_start INTEGER,
            char_end INTEGER,
            FOREIGN KEY(page_id) REFERENCES pages(page_id)
        )
        """
    )
    conn.execute("CREATE INDEX idx_chunks_page_id ON chunks(page_id)")
    conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(title, url, text, tokenize='trigram')")
    conn.execute("CREATE TABLE kb_meta(key TEXT PRIMARY KEY, value TEXT)")


def _insert_page(conn: sqlite3.Connection, page: CircuitMarkdownPage) -> int:
    cursor = conn.execute(
        """
        INSERT INTO pages(path, title, url, published_at, text, text_hash)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (str(page.path), page.title, page.url, page.published_at, page.text, page.text_hash),
    )
    return int(cursor.lastrowid)


def _insert_chunk(
    conn: sqlite3.Connection,
    page_id: int,
    chunk_index: int,
    page: CircuitMarkdownPage,
    text: str,
    char_start: int,
    char_end: int,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO chunks(page_id, chunk_index, title, url, path, text, char_start, char_end)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (page_id, chunk_index, page.title, page.url, str(page.path), text, char_start, char_end),
    )
    return int(cursor.lastrowid)


def _extract_labeled_value(raw: str, label: str) -> str:
    pattern = rf"^\s*\*\*{re.escape(label)}\*\*\s*:\s*(.*?)\s*$"
    match = re.search(pattern, raw, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def _quote_fts(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


def _component_like_terms(tokens: set[str]) -> set[str]:
    return {
        token.lower()
        for token in tokens
        if re.match(r"^[a-z]{1,4}\d+[a-z0-9_.+\-]*$", token.lower())
        or re.match(r"^\d+(\.\d+)?\s*(r|k|m|ohm|v|a|ma|ua|w|uf|nf|pf|mh|uh|hz|khz|mhz)$", token.lower())
    }


def _page_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"page_(\d+)\.md$", path.name)
    if match:
        return int(match.group(1)), path.name
    return 10**12, path.name

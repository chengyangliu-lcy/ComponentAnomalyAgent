from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.circuit_kb import (
    DEFAULT_CHUNK_CHARS,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_MAX_SOURCE_FILE_BYTES,
    DEFAULT_MIN_CHUNK_CHARS,
    CIRCUIT_DIAGNOSIS_CONTENT_MARKERS,
    parse_markdown_metadata,
    parse_markdown_page,
    is_low_value_source,
    is_boilerplate_text,
    is_circuitmaker_project_source,
    _clean_public_tech_page,
    _init_schema,
    DB_FILENAME,
)
from tools.utils import compact_text

import json
import os
import re
import sqlite3
from collections import Counter
from urllib.parse import urlparse
from typing import Any


# ── Extended high-quality content signals ──

TECHNICAL_DEPTH_MARKERS = (
    # Equations and formulas
    "=", "/", "*", "+", "-", "approx", "delta",
    # Circuit-specific high-signal terms
    "计算公式", "推导过程", "设计实例", "实测波形",
    "示波器", "万用表", "调试记录", "实验验证",
    "vout", "iout", "vin", "iin", "vref", "iref",
    "vds", "vgs", "rds", "fsw", "duty",
    "frequency response", "bode plot", "phase margin",
    "gain margin", "crossover frequency",
    "transfer function", "pole", "zero",
    "step response", "load transient",
    "thermal", "junction temperature", "heatsink",
    "derating", "soa", "safe operating area",
)

CAUSAL_REASONING_MARKERS = (
    "因为", "所以", "导致", "引起", "使得", "造成",
    "促进", "增强", "抑制", "减小", "抵消", "改善",
    "降低", "提升", "提高", "防止", "避免",
    "because", "therefore", "causes", "results in",
    "leads to", "prevents", "reduces", "increases",
)

QA_STRUCTURE_MARKERS = (
    "请问", "请教", "如何解决", "怎么处理", "为什么",
    "答", "回答", "回复", "解决方案", "解决方法",
    "how to", "solution", "answered", "resolved",
)

MAX_PAGES_PER_SOURCE = 50000  # Don't scan more than this per source

MIN_HIGH_QUALITY_SCORE = 3
MIN_CONTENT_LENGTH = 200
MAX_CHUNK_CHARS = 5000  # Hard limit: chunks longer than this get force-split
MIN_CHUNK_QUALITY_LENGTH = 500  # Post-chunk: discard chunks shorter than this without circuit keywords

CIRCUIT_KEYWORDS_IN_CHUNK = (
    "电路", "电压", "电流", "电阻", "电容", "MOS", "二极管", "三极管",
    "滤波", "反馈", "振荡", "稳压", "电源", "信号", "PWM", "ADC", "DAC",
    "运放", "比较器", "充电", "放电", "导通", "截止", "开关", "频率",
    "增益", "纹波", "尖峰", "噪声", "环路", "回路", "恒流", "恒压",
    "补偿", "采样", "限流", "保护", "驱动", "TL431", "LM358", "STM32",
    "BUCK", "BOOST", "flyback", "buck", "boost", "circuit", "voltage",
    "current", "resistor", "capacitor", "inductor", "op amp", "transistor",
    "diode", "filter", "feedback", "oscillator", "regulator", "power",
    "amplifier", "converter", "charger", "sensor", "relay", "transformer",
    "схема", "усилитель", "стабилизатор", "транзистор", "конденсатор",
    "резистор", "диод",
)

# ── Phase 1 quick-check terms (for filtering on title+url alone) ──
# Broadened to include English and Russian circuit-related keywords
# so that English/Russian sources pass the initial filter.

QUICK_CHECK_TERMS = (
    # Chinese circuit diagnosis markers (from CIRCUIT_DIAGNOSIS_CONTENT_MARKERS)
    "故障分析", "异常原因", "失效机制", "失效分析", "烧毁原因",
    "过热原因", "不工作原因", "维修方法", "维修步骤", "排查步骤",
    "故障诊断", "反馈环路", "补偿计算", "恒流原理", "恒压原理",
    "反激原理", "开关电源原理", "BUCK电路", "BOOST电路", "LLC谐振",
    "选型计算", "参数计算", "NTC选型", "浪涌计算", "缓启动设计",
    "纹波计算", "尖峰抑制", "EMI滤波", "电路原理", "电路分析",
    "电路设计", "波形分析", "电路故障",
    # English circuit terms (for English sources)
    "datasheet", "application note", "app note", "pin function",
    "typical application", "reference design", "evaluation board",
    "design guide", "schematic", "circuit diagram", "power supply",
    "amplifier", "oscillator", "voltage regulator", "led driver",
    "motor driver", "sensor module", "pwm controller", "mcu",
    "microcontroller", "converter", "inverter", "charger",
    "battery", "relay", "transformer", "diode", "transistor",
    "capacitor", "resistor", "inductor", "op amp", "opamp",
    "arduino", "stm32", "esp32", "raspberry",
    "buck", "boost", "flyback", "forward", "llc",
    "tl431", "lm358", "lm324", "lm393", "uc3842", "sg3525",
    "ir2110", "ir2153", "ne555", "lm78", "lm1117", "ams1117",
    "irf540", "irf", "ao340", "pc817", "el817", "4n25",
    "moc302", "bta16", "bt136", "2n2222", "2n3904",
    "lm2596", "lm2576", "xl4015", "mp1584", "mt3608",
    "pre-amplifier", "audio amplifier", "battery charger",
    "temperature sensor", "humidity sensor", "gas sensor",
    "co2 sensor", "motion sensor", "ir sensor",
    "reflow oven", "soldering", "pcb", "circuit",
    # Russian circuit terms (for radiokot.ru)
    "схема", "усилитель", "генератор", "питание", "стабилизатор",
    "транзистор", "микросхема", "конденсатор", "резистор",
    "диод", "реле", "трансформатор", "радио",
)

# ── Per-source URL filtering rules ──
# Only include pages whose URL path matches these allowed patterns for each domain

PER_SOURCE_ALLOWED_PATHS = {
    "elecfans.com": (
        "/article/", "/dianlutu/", "/analog/", "/power/",
        "/emc/", "/eda/", "/yuanqijian/", "/soft/",
        "/jiekou/",
    ),
    "www.elecfans.com": (
        "/article/", "/dianlutu/", "/analog/", "/power/",
        "/emc/", "/eda/", "/yuanqijian/", "/soft/",
        "/jiekou/",
    ),
    "bbs.eeworld.com.cn": (
        "/thread-", "/drycargo/",
    ),
    "www.eeworld.com.cn": (
        "/drycargo/",
    ),
    "eeworld.com.cn": (
        "/drycargo/",
    ),
    "www.eet-china.com": (
        "/blog/", "/ART_", "/news/",
    ),
    "mbb.eet-china.com": (
        "/blog/",
    ),
    "eet-china.com": (
        "/blog/", "/ART_",
    ),
    "radiokot.ru": (
        "/circuit/", "/articles/",
    ),
    "radiokot.ru:81": (
        "/circuit/", "/articles/",
    ),
    "circuitmaker.com": (
        "/Projects/Details/",
    ),
    "oshwlab.com": (
        "/",  # all project pages
    ),
    "www.hackster.io": (
        "/",  # all project pages
    ),
    "hackster.io": (
        "/",  # all project pages
    ),
    "www.electronicsforu.com": (
        "/electronics-projects/", "/electronicsforu/circuitarchives/",
        "/design-guides/", "/resources/",
    ),
    "electronicsforu.com": (
        "/electronics-projects/", "/electronicsforu/circuitarchives/",
        "/design-guides/", "/resources/",
    ),
}

PER_SOURCE_REJECT_PATH_PARTS = {
    "elecfans.com": ("/tag/", "/category/", "/company/",
                     "/download/", "/bbs/", "/forum/", "/user/"),
    "www.elecfans.com": ("/tag/", "/category/", "/company/",
                          "/download/", "/bbs/", "/forum/", "/user/"),
    "bbs.eeworld.com.cn": ("/forum/", "/user/", "/space/", "/group/"),
    "www.eet-china.com": ("/ace/", "/contest/", "/magazine/", "/subscribe/"),
    "eet-china.com": ("/ace/", "/contest/", "/magazine/", "/subscribe/"),
    "mbb.eet-china.com": ("/forum/", "/group/", "/activity/", "/contest/"),
    "circuitmaker.com": ("/Blog/", "/Documentation/", "/Forum/", "/Users/",
                         "/Components/", "/stream/", "/Hubs", "/About"),
    "www.hackster.io": ("/live/", "/contests/", "/channels/", "/products/"),
    "hackster.io": ("/live/", "/contests/", "/channels/", "/products/"),
    "oshwlab.com": ("/forum/", "/activity/"),
    "radiokot.ru": ("/forum/", "/search/", "/adv/"),
    "www.electronicsforu.com": ("/tag/", "/category/", "/author/"),
    "electronicsforu.com": ("/tag/", "/category/", "/author/"),
}

PER_SOURCE_REJECT_TITLES = {
    "elecfans.com": ("提示信息", "404", "电子发烧友网", "电子搜索"),
    "www.elecfans.com": ("提示信息", "404", "电子发烧友网", "电子搜索"),
    "bbs.eeworld.com.cn": ("提示信息", "安全验证", "404", "EEWORLD首页"),
    "www.eet-china.com": ("ACE Awards", "Subscription Corner", "404", "EETC首页"),
    "eet-china.com": ("ACE Awards", "Subscription Corner", "404", "EETC首页"),
    "mbb.eet-china.com": ("面包板首页", "广告", "404", "提示信息"),
    "circuitmaker.com": ("Loading project", "Sign in", "Error", "404"),
    "www.hackster.io": ("Hackster Hardware Meetup", "User Profile", "404", "Sign in"),
    "hackster.io": ("Hackster Hardware Meetup", "User Profile", "404", "Sign in"),
    "radiokot.ru": ("Главная", "РадиоКот :: Главная", "Поиск", "Форум"),
    "oshwlab.com": ("EasyEDA open source hardware lab", "OSHWLab", "Login"),
    "www.electronicsforu.com": ("404", "Subscribe", "Newsletter"),
    "electronicsforu.com": ("404", "Subscribe", "Newsletter"),
}

# Require .html extension for Chinese electronics sites with chaotic URL structures
PER_SOURCE_REQUIRE_HTML_EXT = ()  # No domains require .html extension anymore

# ── Per-source boilerplate patterns ──

# Universal boilerplate (all sources)
UNIVERSAL_BOILERPLATE_RE = [
    re.compile(r"^\s*(立即登录|注册|登录|Sign\s*Up|Sign\s*In|Login|Log\s*In)\s*$", re.IGNORECASE),
    re.compile(r"^\s*!\[.*?\]\(.*?\)\s*$"),
    re.compile(r"^\s*[*\s]*$"),
    re.compile(r"^\s*广告\s*$"),
    re.compile(r"(?:上一篇|下一篇|相关推荐|热门推荐|相关文章|Related\s*Articles|Related\s*Posts)\s*$"),
]

# elecfans.com specific
ELECFANS_BOILERPLATE_RE = [
    re.compile(r"发表于\s*\d{4}-\d{2}-\d{2}\s*[•·]\s*\d+次阅读"),
    re.compile(r"\d{4}-\d{2}-\d{2}\s*[•·]\s*\d+次阅读"),
    re.compile(r"\d+次阅读"),
    re.compile(r"次下载"),
    re.compile(r"下载该资料的人也在下载|下载该资料的人还在阅读"),
    re.compile(r"免费下载.*(?:PDF|pdf).*电子书"),
    re.compile(r"专栏\s+电子说\s+商业评论"),
    re.compile(r"电子发烧友网\s*>"),
    re.compile(r"skin\.elecfans\.com|skin-2012|file\.elecfans\.com"),
    re.compile(r"电子资料下载频道.*为电子工程师"),
    re.compile(r"电子技术应用频道.*为电子工程师"),
    re.compile(r"专业的电子元器件平台.*及时发布"),
    re.compile(r"电路图频道.*提供电子电路图"),
    re.compile(r"构建电子工程师交流的平台"),
    re.compile(r"电阻器\s+电容器\s+电感器"),
    re.compile(r"接口定义.*芯片引脚图.*元件代换.*光耦"),
    re.compile(r"您好，欢迎来.*电子发烧友网"),
    re.compile(r"请输入元器件型号"),
    re.compile(r"电子搜索"),
    re.compile(r"^\s*首页\s*最新更新\s*电子百科\s*电子问答"),
    # Category navigation listings (电阻器 电容器 ... 列表)
    re.compile(r"^\s*(?:首页|最新更新|电子百科|电子问答|网站导航)\s*$"),
]

# eeworld.com.cn specific — comprehensive forum noise removal
EEWORLD_BOILERPLATE_RE = [
    re.compile(r"设为首页|收藏本站|EEWORLD首页"),
    re.compile(r"社区导航|技术讨论创新帖|全部新帖|干货|资料区"),
    re.compile(r"快捷导航"),
    re.compile(r"单片机\s+物联网\s+汽车电子\s+嵌入式"),
    re.compile(r"^\s*(?:首页|技术|应用)\s*_\|_\s*$"),
    re.compile(r"热门技术文章|热门应用文章|今日技术|今日应用"),
    re.compile(r"^\s*###\s*技术分类\s*$"),
    re.compile(r"^\s*###\s*应用分类\s*$"),
    # Image references (static/ without leading slash, ./static/, http://static/)
    re.compile(r"!\[.*?\]\([\./]?static/"),
    re.compile(r"!\[.*?\]\(http[s]?://[^/]*\.eeworld[^/]*\.cn/static/"),
    # Forum interaction noise
    re.compile(r"复制链接"),
    re.compile(r"使用道具\s*举报"),
    re.compile(r"验证码|安全验证|请输入验证码"),
    re.compile(r"发帖|回帖|签到|打卡"),
    re.compile(r"高级搜索"),
    re.compile(r"\*\*推荐资源\*\*|推荐资源|相关帖子|相关推荐|###\s*推荐资源|###\s*相关帖子"),
    re.compile(r"您需要登录后才可以|登录\s*\|\s*立即注册"),
    re.compile(r"^\s*\|.*\|.*\|$"),  # markdown table rows (nav tables like | 搜索 | **** |)
    re.compile(r"下载次数|资源积分|资源大小|上传时间|上传者"),
    re.compile(r"^\s*!\[.*?\]\(http://download\.eeworld\.com\.cn/"),
    re.compile(r"评分|收藏|分享|举报"),
    re.compile(r"最新活动|TI直播|ADI之路|主题游|打卡"),
    re.compile(r"本帖最后由.*于.*编辑"),
    re.compile(r"_\s*回复\s*_|_\s*使用道具\s*_|_\s*举报\s*_"),
    re.compile(r"^\s*\d+\s*$"),  # standalone numbers (comment/like counts)
    re.compile(r"楼主|沙发|板凳|地板|地下室|下水道"),
    re.compile(r"^\s*<.*>\s*$"),  # HTML tag leftovers
    re.compile(r"小喇叭|提醒|消息"),
    # Forum user status lines (username + online/offline + rank)
    re.compile(r"当前离线|当前在线"),
    re.compile(r"一粒金砂(?:（(?:初级|中级|高级|入门)）)?|超级版主|版主|管理员"),
    re.compile(r"^\s*\S+\s+\*?\*\*\S+\*?\*\*\s*_?(?:当前离线|当前在线)_?\s*"),
    re.compile(r"^\s*>\s*\S+\s+发表于\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}"),
    # Promotional / resource download sections
    re.compile(r"本帖子中包含更多资源|感谢有你|愿一路同行|翻开《"),
    re.compile(r"!\[.*?\]\(http://download\.eeworld\.com\.cn/images/"),
    re.compile(r"有奖直播|任选下载有礼|TI 嵌入式研讨会集锦|ams投影"),
    re.compile(r"测评.*!\(.*?static/image/common/new\.gif"),
    re.compile(r"来源：EEWorld|转载请附上链接"),
    re.compile(r"查看:\s*\d+\s*\|\s*回复:\s*\d+"),
    re.compile(r"^\s*---\|---(?:\|---)*\s*$"),  # markdown table separator rows
    re.compile(r"^\s*\d+\s*$"),  # standalone numbers (already above but reinforce)
    re.compile(r"^\s*_\s*_\s*_\s*$"),  # standalone italic markers
]

# eet-china / 面包板 specific
EET_CHINA_BOILERPLATE_RE = [
    re.compile(r"电子技术社区|技术社区|EE社区|电子工程社区"),
    re.compile(r"电子工程专辑|国际电子商情|电子技术设计|CEO专栏"),
    re.compile(r"杂志免费订阅|EE\|Times\s*全球联播"),
    re.compile(r"在线研讨会|白皮书|小测验|资源中心|每月抽奖"),
    re.compile(r"最新发表\s*推荐阅读\s*明星博主"),
    re.compile(r"^\s*写博文\s*$"),
    re.compile(r"文章：\*\*\d+\*\*\s*阅读：\*\*\d+\*\*"),
    re.compile(r"^\s*\d+\s*$"),  # just a number (like/comment count)
    re.compile(r"好友\s+私信\s+个人主页"),
    re.compile(r"^\s*(?:FPGA|MCU|模拟|电源|测试|通信|PCB|汽车|消费|智能|物联网|软件|采购|供应链|职场|EDA|无人机|机器人|AI|医疗|工业)\s*"),
    re.compile(r"^\s*原创\s*\d+\s*$"),
    re.compile(r"抗击肺炎专题"),
    re.compile(r"最新帖子\s+问答\s+下载\s+更多"),
    # Comment section boilerplate
    re.compile(r"文章评论|条评论"),
    re.compile(r"您需要登录后才可以|登录\s*\|\s*立即注册"),
    re.compile(r"当前离线|离线|在线|\d+小时|\d+枚\(兑换\)"),
    re.compile(r"^\s*\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s*$"),  # date timestamps in comments
    re.compile(r"^\s*###\s*评论\s*$"),
    re.compile(r"复制链接|分享到|举报|收藏"),
    re.compile(r"^\s*\|.*\|.*\|$"),  # markdown table rows
    re.compile(r"评分|收藏|分享|举报"),
    re.compile(r"查看:\s*\d+\s*\|\s*回复:\s*\d+"),
    re.compile(r"_\s*回复\s*_|_\s*使用道具\s*_|_\s*举报\s*_"),
]

# radiokot.ru specific
RADIOKOT_BOILERPLATE_RE = [
    re.compile(r"РадиоКот\s*>|РадиоКот\s*::\s*Главная"),
    re.compile(r"Здравствуйте.*Разрешите представиться"),
    re.compile(r"Главная\s*Схемы\s*Лаборатория\s*Статьи\s*Обучалка"),
    re.compile(r"Ссылки\s*Справочник\s*КотАрт\s*О\s*проекте\s*Форум"),
    re.compile(r"datasheet\.png|find\.png"),
    re.compile(r"Мои\s+чокнутые\s+хозяева"),
    re.compile(r"А\s+здесь\s+Коту\s+наливали\s+пива"),
    re.compile(r"Технологии\s+BUGS"),
    re.compile(r"Радио-схемы.*Мануалы.*Справочники.*Форум"),
    re.compile(r"^\s*(?:Главная|Схемы|Лаборатория|Статьи|Обучалка|Ссылки|Справочник|КотАрт|Форум)\s*$"),
]

# circuitmaker.com specific
CIRCUITMAKER_BOILERPLATE_RE = [
    re.compile(r"Loading project.*Please wait"),
    re.compile(r"Blog\s+Documentation\s+Download\s+Service\s+Status"),
    re.compile(r"Home\s+Projects\s+Hubs\s+Components\s+Forum"),
    re.compile(r"Share\s+by\s+email"),
    re.compile(r"Your\s+message\s+was\s+succesfully\s+sent"),
    re.compile(r"Thumb\s+up\s+Edit\s+Delete"),
    re.compile(r"!\[.*?\]\(/Content/Images/(?:Social|ajax-loader|logo)"),
    re.compile(r"Overview\s+Design\s+Team\s+Components"),
    re.compile(r"From\s+To\s+Subject\s+Body\s+Send\s+Close"),
    re.compile(r"^\s*(?:Sign\s*in|Overview|Design|Team|Components)\s*$", re.IGNORECASE),
]

# hackster.io specific
HACKSTER_BOILERPLATE_RE = [
    re.compile(r"^\s*(?:Sign\s*in|Log\s*in|Register|Join)\s*$", re.IGNORECASE),
    re.compile(r"Explore\s+Projects\s+Contests\s+Events"),
    re.compile(r"!\[.*?\]\(https://hackster\.io/static/"),
]

# oshwlab.com specific
OSHWLAB_BOILERPLATE_RE = [
    re.compile(r"Home\s+Explore\s+Post\s+Forum"),
    re.compile(r"Share\s+Project"),
    re.compile(r"Login\s+\|\s+Register"),
    re.compile(r"Open\s+all\s+in\s+editor"),
    re.compile(r"Recent searches"),
    re.compile(r"^\s*(?:Home|Explore|Post|Forum|Share)\s*$"),
]

# electronicsforu.com specific
ELECTRONICSFORU_BOILERPLATE_RE = [
    re.compile(r"Subscribe\s+to\s+our\s+Newsletter"),
    re.compile(r"^\s*(?:Login|Register|Sign\s*in)\s*$", re.IGNORECASE),
    re.compile(r"^\s*\|.*\|.*\|$"),  # markdown table rows
    re.compile(r"^\s*---\|---(?:\|---)*\s*$"),
    re.compile(r"Share\s+this\s+article|Related\s+Projects|Popular\s+Posts"),
    re.compile(r"View\s+All|More\s+Projects|Recommended\s+For\s+You"),
]

# Map domain to its specific boilerplate patterns
DOMAIN_BOILERPLATE_MAP = {
    "elecfans.com": ELECFANS_BOILERPLATE_RE,
    "www.elecfans.com": ELECFANS_BOILERPLATE_RE,
    "bbs.eeworld.com.cn": EEWORLD_BOILERPLATE_RE,
    "www.eeworld.com.cn": EEWORLD_BOILERPLATE_RE,
    "eeworld.com.cn": EEWORLD_BOILERPLATE_RE,
    "www.eet-china.com": EET_CHINA_BOILERPLATE_RE,
    "eet-china.com": EET_CHINA_BOILERPLATE_RE,
    "mbb.eet-china.com": EET_CHINA_BOILERPLATE_RE,
    "radiokot.ru": RADIOKOT_BOILERPLATE_RE,
    "radiokot.ru:81": RADIOKOT_BOILERPLATE_RE,
    "circuitmaker.com": CIRCUITMAKER_BOILERPLATE_RE,
    "www.hackster.io": HACKSTER_BOILERPLATE_RE,
    "hackster.io": HACKSTER_BOILERPLATE_RE,
    "oshwlab.com": OSHWLAB_BOILERPLATE_RE,
    "www.electronicsforu.com": ELECTRONICSFORU_BOILERPLATE_RE,
    "electronicsforu.com": ELECTRONICSFORU_BOILERPLATE_RE,
}

# ── Multi-article splitting ──

# Expanded patterns for splitting multi-article pages (esp. elecfans)
ARTICLE_SPLIT_PATTERN = re.compile(
    r"\n*(?:"
    r"发表于\s*\d{4}-\d{2}-\d{2}"          # elecfans date marker
    r"|\n#{1,3}\s+(?:\S.{5,})"              # heading with meaningful title
    r"|\n---\s*\n"                           # horizontal rule separator
    r")\s*"
)


def _get_domain(url: str) -> str:
    """Extract hostname from URL."""
    parsed = urlparse(url)
    return parsed.hostname or ""


def _is_source_url_allowed(url: str, title: str) -> bool:
    """Check if a page URL and title are allowed based on per-source rules."""
    domain = _get_domain(url)
    parsed = urlparse(url)
    path_lower = parsed.path.lower()
    title_lower = (title or "").strip().lower()

    # Check domain-specific allowed paths (compare in lowercase)
    if domain in PER_SOURCE_ALLOWED_PATHS:
        allowed_paths = PER_SOURCE_ALLOWED_PATHS[domain]
        allowed_lower = [p.lower() for p in allowed_paths]
        if not any(path_lower.startswith(prefix) or prefix in path_lower for prefix in allowed_lower):
            return False

    # Check domain-specific rejected path parts (compare in lowercase)
    if domain in PER_SOURCE_REJECT_PATH_PARTS:
        reject_parts = PER_SOURCE_REJECT_PATH_PARTS[domain]
        reject_lower = [p.lower() for p in reject_parts]
        if any(part in path_lower for part in reject_lower):
            return False

    # Check domain-specific rejected titles
    if domain in PER_SOURCE_REJECT_TITLES:
        reject_titles = PER_SOURCE_REJECT_TITLES[domain]
        if any(marker.lower() in title_lower for marker in reject_titles):
            return False

    # Require .html extension for certain domains
    # Note: elecfans removed from this requirement since many valid URLs don't end with .html
    if domain in PER_SOURCE_REQUIRE_HTML_EXT:
        if not path_lower.endswith(".html"):
            return False

    # Hackster project pages have /username/project-slug format
    # User profile pages are just /username (only 1 segment)
    if domain in ("www.hackster.io", "hackster.io"):
        segments = path_lower.strip("/").split("/")
        if len(segments) <= 1:
            return False
        # Also reject known non-project paths
        if segments[0] in ("live", "contests", "channels", "products", "live"):
            return False

    # OshwLab: reject user profile pages (just /username, no /username/project)
    if domain in ("oshwlab.com"):
        # User profiles have URL like /username (no slash after)
        # Project pages have /username/projectname
        segments = path_lower.strip("/").split("/")
        if len(segments) <= 1:
            return False

    return True


def _strip_boilerplate(text: str, domain: str = "") -> str:
    """Remove boilerplate/navigation lines from scraped web content.

    Applies universal patterns + domain-specific patterns.
    """
    if not text:
        return ""

    # Collect applicable patterns
    patterns = list(UNIVERSAL_BOILERPLATE_RE)
    if domain in DOMAIN_BOILERPLATE_MAP:
        patterns.extend(DOMAIN_BOILERPLATE_MAP[domain])

    lines = text.split("\n")
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            clean_lines.append("")
            continue
        # Universal + domain-specific boilerplate removal
        if any(pat.search(stripped) for pat in patterns):
            continue
        # Skip very short lines that are likely navigation
        if len(stripped) < 10 and not re.search(r"[A-Za-z]{3,}|[一-鿿]{2,}", stripped):
            continue
        # Skip image-only lines
        if stripped.startswith("![") and ")" in stripped:
            continue
        clean_lines.append(stripped)

    result = "\n".join(clean_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def _split_multi_article_page(text: str, title: str = "") -> list[str]:
    """Split pages that contain multiple independent mini-articles.

    Many elecfans.com pages concatenate 10+ unrelated articles.
    Each sub-article starts with a date line or heading separator.
    """
    # Find all split points
    split_points = []
    for m in ARTICLE_SPLIT_PATTERN.finditer(text):
        split_points.append(m.start())

    if len(split_points) <= 1:
        return [text]

    # Split text at each point
    articles = []
    prev = 0
    for sp in split_points:
        if sp > prev:
            chunk = text[prev:sp].strip()
            if len(chunk) >= 200:
                articles.append(chunk)
        prev = sp
    # Last chunk
    last = text[prev:].strip()
    if len(last) >= 200:
        articles.append(last)

    if not articles:
        return [text]

    # For pages with title, prefer sub-articles whose content matches the title
    if title:
        title_lower = title.lower()
        title_keywords = re.findall(r"[A-Za-z]{3,}|[一-鿿]{2,}", title_lower)
        scored = []
        for article in articles:
            article_lower = article.lower()
            title_overlap = sum(1 for kw in title_keywords if kw in article_lower)
            scored.append((title_overlap, article))
        scored.sort(key=lambda x: x[0], reverse=True)
        # Return all sub-articles but prioritize ones matching the title
        return [a for _, a in scored]

    return articles


def _compute_page_quality_score(title: str, url: str, text: str, domain: str = "") -> tuple[int, dict[str, int]]:
    """Score a page's technical quality on multiple dimensions.

    Returns (total_score, dimension_scores).
    Applies domain-specific adjustments.
    """
    lowered = f"{title}\n{text}".lower()
    scores: dict[str, int] = {}

    # 1. Circuit diagnosis content markers (up to 5 points)
    diag_hits = sum(1 for m in CIRCUIT_DIAGNOSIS_CONTENT_MARKERS if m.lower() in lowered)
    scores["diagnosis_markers"] = min(5, diag_hits)

    # 2. Technical depth signals (up to 5 points)
    depth_hits = sum(1 for m in TECHNICAL_DEPTH_MARKERS if m.lower() in lowered)
    scores["technical_depth"] = min(5, depth_hits // 2)

    # 3. Component model numbers (up to 3 points)
    model_pattern = re.compile(
        r"\b(?:TL431|LM358|LM324|LM393|UC384[2-5]|SG3525|IR2110|IR2153|"
        r"NE555|LM78\d{2}|LM1117|AMS1117|AO340[0-9]|IRF[0-9]+|"
        r"STM32|ESP32|ATmega|PC817|EL817|4N25|MOC302[0-3]|"
        r"BTA[0-9]+|BT136|TYN[0-9]+|2N2222|2N3904|2N3906|"
        r"BC547|BC557|TIP[0-9]+|LM2596|LM2576|XL4015|MP1584|"
        r"MT3608|SX1308|TPS[0-9]+|CS[0-9]+|OB[0-9]+|CR[0-9]+|"
        r"MAX038|ICL8038|XR2206|AD8038|TDA7294)\b",
        re.IGNORECASE,
    )
    model_hits = len(set(model_pattern.findall(lowered)))
    scores["model_numbers"] = min(3, model_hits)

    # 4. Schematic/circuit references (up to 3 points)
    schematic_markers = (
        "原理图", "电路图", "schematic", "circuit diagram",
        "pcb layout", "布线", "layout", "r1", "r2", "c1", "c2",
        "q1", "q2", "d1", "d2", "u1", "u2",
        "схема",  # Russian: circuit/schematic
    )
    schematic_hits = sum(1 for m in schematic_markers if m in lowered)
    scores["schematic_refs"] = min(3, schematic_hits // 2)

    # 5. Content structure quality (up to 3 points)
    structure_score = 0
    if len(text) > 2000:
        structure_score += 1
    if text.count("##") >= 3:
        structure_score += 1
    if re.search(r"\d+\.\s+\w+", text):
        structure_score += 1
    scores["structure"] = min(3, structure_score)

    # 6. Causal reasoning density (up to 3 points)
    causal_hits = sum(1 for m in CAUSAL_REASONING_MARKERS if m.lower() in lowered)
    scores["causal_reasoning"] = min(3, causal_hits // 3)

    # 7. QA structure match (up to 3 points for forum content)
    qa_hits = sum(1 for m in QA_STRUCTURE_MARKERS if m.lower() in lowered)
    scores["qa_structure"] = min(3, qa_hits // 2)

    # 8. Penalties
    penalties = 0
    if is_low_value_source(url):
        penalties -= 5
    if is_boilerplate_text(f"{title}\n{text}"):
        penalties -= 3
    if len(text) < 200:
        penalties -= 3
    # Navigation/boilerplate residue penalty
    nav_terms = ("上一篇", "下一篇", "相关推荐", "热门推荐", "相关文章",
                 "related articles", "related posts", "you may also like")
    nav_hits = sum(1 for m in nav_terms if m in lowered)
    if nav_hits >= 1:
        penalties -= 2

    # Domain-specific adjustments
    if "elecfans.com" in domain:
        # elecfans: stricter penalty for title-content mismatch
        # If title has specific component names but content doesn't mention them
        title_models = set(model_pattern.findall(title.lower()))
        if title_models:
            content_lower = text.lower()
            missing_models = [m for m in title_models if m not in content_lower]
            if len(missing_models) >= len(title_models) // 2:
                penalties -= 3  # Title says "TL431" but content doesn't mention it
        # elecfans download shell penalty: pages with download metadata but no circuit content
        if ("下载次数" in lowered or "资源积分" in lowered or "资源大小" in lowered) and not any(kw in lowered for kw in ("电路图", "原理图", "schematic", "电路分析", "设计实例")):
            penalties -= 10
    if "circuitmaker.com" in domain:
        # circuitmaker: penalize pages with no technical content
        if "loading project" in lowered or "please wait" in lowered:
            penalties -= 10  # nearly empty page
    if "hackster.io" in domain or "hackster.com" in domain:
        # hackster: penalize user profile pages
        if " - hackster.io" in lowered or " - hackster.com" in lowered:
            penalties -= 10
    if "radiokot.ru" in domain:
        # radiokot: only /circuit/ and /articles/ paths are useful
        if "/circuit/" not in url.lower() and "/articles/" not in url.lower():
            penalties -= 5
        # Reward pages with breadcrumb (actual content pages)
        if "радиокот >" in lowered or "радиокот > схемы >" in lowered:
            scores["radiokot_breadcrumb"] = 2

    scores["penalties"] = penalties

    total = sum(scores.values())
    return total, scores


def build_high_quality_kb(
    source_dirs: list[Path],
    output_dir: Path,
    *,
    max_docs: int = 10000,
    max_file_bytes: int = DEFAULT_MAX_SOURCE_FILE_BYTES,
    progress_every: int = 500,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS,
) -> dict[str, Any]:
    """Build a high-quality KB using multi-dimensional quality scoring
    with per-source filtering rules."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    db_path = output_path / DB_FILENAME
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    stats: Counter[str] = Counter()
    seen_hashes: set[str] = set()
    seen_urls: set[str] = set()
    docs = 0
    chunks = 0

    try:
        _init_schema(conn)
        with conn:
            for source_dir in source_dirs:
                if not source_dir.exists():
                    stats["missing_source_dirs"] += 1
                    continue

                # Phase 1: Fast metadata + quality scoring
                scored_candidates: list[tuple[int, Path, dict[str, Any]]] = []
                stats["phase1_scanned"] = 0
                # Use os.listdir for fast scanning (much faster than glob on large dirs)
                try:
                    all_files = os.listdir(source_dir)
                except OSError:
                    stats["missing_source_dirs"] += 1
                    continue
                page_files = [f for f in all_files if f.startswith("page_") and f.endswith(".md")]
                stats["phase1_total_files"] = len(page_files)

                for filename in page_files[:MAX_PAGES_PER_SOURCE]:
                    path = source_dir / filename
                    stats["phase1_scanned"] += 1
                    if progress_every > 0 and stats["phase1_scanned"] % 5000 == 0:
                        print(
                            f"phase1: scanned={stats['phase1_scanned']}/{stats['phase1_total_files']} "
                            f"candidates={len(scored_candidates)} "
                            f"kept={docs}",
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

                    title = metadata["title"].strip()
                    url = metadata["url"].strip()
                    domain = _get_domain(url)

                    # Per-source URL filtering
                    if not _is_source_url_allowed(url, title):
                        stats["phase1_filtered_url"] += 1
                        continue

                    if is_low_value_source(url):
                        stats["phase1_filtered_low_value"] += 1
                        continue

                    # Quick quality check on metadata alone (using broadened terms)
                    quick_score = sum(
                        1 for m in QUICK_CHECK_TERMS
                        if m.lower() in f"{title}\n{url}".lower()
                    )
                    if quick_score == 0:
                        stats["phase1_filtered_no_diag_terms"] += 1
                        continue

                    scored_candidates.append((quick_score, path, metadata))

                # Sort by quick score, process best first (up to max_docs * 2)
                scored_candidates.sort(key=lambda x: x[0], reverse=True)
                target_candidates = min(len(scored_candidates), max_docs * 2)

                if progress_every > 0:
                    print(
                        f"phase1 done for {source_dir.name}: "
                        f"scanned={stats['phase1_scanned']} "
                        f"quick_candidates={len(scored_candidates)} "
                        f"will_deep_check={target_candidates}",
                        flush=True,
                    )

                # Phase 2: Deep content quality check
                stats["phase2_scanned"] = 0
                for idx, (_, path, metadata) in enumerate(scored_candidates[:target_candidates]):
                    if docs >= max_docs:
                        break

                    stats["phase2_scanned"] += 1
                    if progress_every > 0 and stats["phase2_scanned"] % 500 == 0:
                        print(
                            f"phase2: processed={stats['phase2_scanned']}/{target_candidates} "
                            f"kept={docs}",
                            flush=True,
                        )

                    try:
                        page = parse_markdown_page(path)
                    except OSError:
                        stats["read_errors"] += 1
                        continue

                    url = page.url or metadata["url"]
                    domain = _get_domain(url)

                    # Per-source title rejection (catch pages like "提示信息")
                    if domain in PER_SOURCE_REJECT_TITLES:
                        title_lower = (page.title or "").strip().lower()
                        if any(marker.lower() in title_lower for marker in PER_SOURCE_REJECT_TITLES[domain]):
                            stats["phase2_filtered_title"] += 1
                            continue

                    # Clean page text — strip boilerplate (with domain-specific rules)
                    cleaned_page = _clean_public_tech_page(page)
                    cleaned_text = _strip_boilerplate(cleaned_page.text, domain)

                    # Reject pages that are mostly empty after cleaning
                    if len(cleaned_text) < MIN_CONTENT_LENGTH:
                        stats["phase2_filtered_empty_after_clean"] += 1
                        continue

                    # For elecfans-style multi-article pages, split and score each sub-article
                    sub_articles = _split_multi_article_page(cleaned_text, page.title)
                    best_score = -999
                    best_text = cleaned_text
                    best_dim_scores = {}

                    for sub_text in sub_articles:
                        if len(sub_text) < MIN_CONTENT_LENGTH:
                            continue
                        score, dim_scores = _compute_page_quality_score(
                            cleaned_page.title, cleaned_page.url, sub_text, domain
                        )
                        if score > best_score:
                            best_score = score
                            best_text = sub_text
                            best_dim_scores = dim_scores

                    if best_score < MIN_HIGH_QUALITY_SCORE:
                        stats["filtered_low_quality"] += 1
                        continue

                    # Use the best sub-article text
                    text = best_text

                    # Chunk the page
                    page_chunks = _chunk_text(
                        text,
                        chunk_chars=chunk_chars,
                        chunk_overlap=chunk_overlap,
                        min_chunk_chars=min_chunk_chars,
                    )

                    # Post-chunk quality filter: discard short chunks without circuit keywords
                    quality_chunks = []
                    for chunk_text in page_chunks:
                        if len(chunk_text) < MIN_CHUNK_QUALITY_LENGTH:
                            # Short chunk: only keep if it contains circuit keywords
                            has_circuit_kw = any(kw.lower() in chunk_text.lower() for kw in CIRCUIT_KEYWORDS_IN_CHUNK)
                            if not has_circuit_kw:
                                stats["filtered_low_quality_chunk"] += 1
                                continue
                        quality_chunks.append(chunk_text)

                    if not quality_chunks:
                        stats["filtered_all_chunks_low_quality"] += 1
                        continue

                    url_key = cleaned_page.url or cleaned_page.path
                    if url_key in seen_urls:
                        stats["duplicate_url"] += 1
                        continue

                    seen_urls.add(url_key)
                    docs += 1

                    for chunk_idx, chunk_text in enumerate(quality_chunks):
                        # Prepend contextual retrieval prefix
                        contextual_text = f"[文档: {cleaned_page.title[:100]}] [来源: {domain}] {chunk_text}"
                        chunk_hash = compact_text(contextual_text, 80)
                        if chunk_hash in seen_hashes:
                            continue
                        seen_hashes.add(chunk_hash)

                        conn.execute(
                            """
                            INSERT INTO chunks
                                (page_id, chunk_index, title, url, path,
                                 text, char_start, char_end)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                docs,
                                chunk_idx,
                                cleaned_page.title[:500],
                                url_key[:2048],
                                str(path)[:2048],
                                contextual_text,
                                0,
                                len(contextual_text),
                            ),
                        )
                        chunks += 1

                _build_fts_index(conn)

    finally:
        conn.close()

    meta = {
        "source_dirs": [str(d) for d in source_dirs],
        "output_dir": str(output_path),
        "max_docs": max_docs,
        "chunk_chars": chunk_chars,
        "chunk_overlap": chunk_overlap,
        "min_chunk_chars": min_chunk_chars,
        "documents": docs,
        "chunks": chunks,
        "quality_threshold": MIN_HIGH_QUALITY_SCORE,
        "stats": dict(stats),
    }
    (output_path / "build_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return meta


def _chunk_text(
    text: str,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS,
    max_chunk_chars: int = MAX_CHUNK_CHARS,
) -> list[str]:
    """Split text into overlapping chunks at section boundaries.

    Enforces a hard maximum chunk size — any chunk exceeding max_chunk_chars
    is force-split by paragraphs and then by sentences.
    """
    if len(text) <= chunk_chars:
        return [text] if len(text) >= min_chunk_chars else []

    chunks: list[str] = []
    sections = re.split(r"\n(?=#{1,4}\s)", text)

    current = ""
    for section in sections:
        if len(current) + len(section) <= chunk_chars:
            current += ("\n" if current else "") + section
        else:
            if len(current) >= min_chunk_chars:
                chunks.append(current)
            # If section is too long, split by paragraphs
            if len(section) > chunk_chars:
                paragraphs = re.split(r"\n\n+", section)
                current = ""
                for para in paragraphs:
                    if len(current) + len(para) <= chunk_chars:
                        current += ("\n\n" if current else "") + para
                    else:
                        if len(current) >= min_chunk_chars:
                            chunks.append(current)
                        # If even a single paragraph is too long, force-split by sentences
                        if len(para) > max_chunk_chars:
                            sentence_chunks = _force_split_long_text(para, max_chunk_chars, min_chunk_chars)
                            chunks.extend(sentence_chunks)
                            current = ""
                        else:
                            current = para
            else:
                current = section

    if len(current) >= min_chunk_chars:
        chunks.append(current)

    # Final pass: force-split any remaining chunks that exceed max_chunk_chars
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_chunk_chars:
            final_chunks.extend(_force_split_long_text(chunk, max_chunk_chars, min_chunk_chars))
        else:
            final_chunks.append(chunk)

    return final_chunks


def _force_split_long_text(text: str, max_chars: int = MAX_CHUNK_CHARS, min_chars: int = DEFAULT_MIN_CHUNK_CHARS) -> list[str]:
    """Force-split text that exceeds max_chars by splitting on sentence boundaries."""
    if len(text) <= max_chars:
        return [text] if len(text) >= min_chars else []

    # Split on sentence boundaries (Chinese/English periods, question marks, etc.)
    sentences = re.split(r"(?<=[。！？.!?；;])\s*", text)
    chunks = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        if len(current) + len(sentence) <= max_chars:
            current += sentence
        else:
            if len(current) >= min_chars:
                chunks.append(current)
            current = sentence
    if len(current) >= min_chars:
        chunks.append(current)

    # If we still have chunks > max_chars, do a hard character-level split
    result = []
    for chunk in chunks:
        if len(chunk) > max_chars:
            for i in range(0, len(chunk), max_chars):
                sub = chunk[i:i + max_chars]
                if len(sub) >= min_chars:
                    result.append(sub)
        else:
            result.append(chunk)
    return result


def _build_fts_index(conn: sqlite3.Connection) -> None:
    """Build FTS5 index on chunks."""
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            title, url, path, text, content=chunks, content_rowid=chunk_id
        )
    """)
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a high-quality circuit diagnosis KB from WARC sources."
    )
    parser.add_argument("--warc-root", default="/media/work/1ECC291B3E106A4A/xinyang/circuit/warc_output")
    parser.add_argument("--source-id", action="append", default=None)
    parser.add_argument("--output-dir", default="knowledge_base/circuit_diagnosis_fts_hq")
    parser.add_argument("--max-docs", type=int, default=10000)
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()

    warc_root = Path(args.warc_root)
    source_ids = args.source_id or [
        # circuitmaker and hackster removed: pages are mostly empty shells / user profiles
        "98f81c1f-cfda-46a6-91c0-921a7894d20b",  # electronicsforu.com
        "b8ee701e-9911-43a2-9cc0-3a1a1bda8782",  # bbs.eeworld.com.cn
        "c9df5712-eb33-40f1-9d8b-dc5baf39f21d",  # oshwlab.com
        "c9feda27-eb86-43aa-bc04-d39d69344a8d",  # elecfans.com (with strict filtering)
        "cf80eebf-5d42-4556-b2c3-a4de6966df52",  # mbb.eet-china.com
        "df5771a0-8fad-4d88-8a57-248fd4ef5f6e",  # radiokot.ru
        "efedf473-6a15-42fd-ae8c-321460f852f1",  # www.eet-china.com
    ]
    source_dirs = [warc_root / sid for sid in source_ids]

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    meta = build_high_quality_kb(
        source_dirs,
        output_dir,
        max_docs=args.max_docs,
        progress_every=args.progress_every,
    )
    print(f"Built high-quality KB at {output_dir}")
    print(f"documents={meta['documents']} chunks={meta['chunks']}")
    print(f"stats={meta['stats']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
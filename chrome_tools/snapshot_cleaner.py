"""Clean Chrome DevTools MCP accessibility snapshots, adapted from NightHeron's
fetchers/snapshot_cleaner.py (copied, then left largely unchanged)."""

from __future__ import annotations

import logging
import re


logger = logging.getLogger("chrome_tools.snapshot_cleaner")

TRUNCATION_SUFFIX = "[... truncated]"

NODE_RE = re.compile(r"^(?P<indent>\s*)uid=\S+\s+(?P<role>\S+)")
LINK_RE = re.compile(r"^\s*uid=\S+\s+link\b")
LINK_TITLE_RE = re.compile(r'link(?: "(?P<title>.*?)")? url=')
LINK_URL_RE = re.compile(r'url="(?P<url>[^"]*)"')

CONTAINER_ROLES_TO_DROP = {
    "banner",
    "combobox",
    "form",
    "Iframe",
    "listbox",
    "navigation",
    "tablist",
    "toolbar",
}

LEAF_ROLES_TO_DROP = {
    "button",
    "checkbox",
    "image",
    "application",
    "alert",
    "LineBreak",
    "menuitem",
    "option",
    "slider",
    "tab",
    "textbox",
}

CONTAINER_TEXT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\badvertisement\b",
        r"\bmarket data\b",
        r"\bnotifications\b",
        r"\btop securities\b",
        r"\bvideo player\b",
        r"\bcurrent price\b",
        r"\bcurrent percentage change\b",
    )
)

LINK_URL_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^javascript:",
        r"doubleclick|googleads",
        r"/client/silent-login|signin|login",
        r"/subscriptions|subscribe",
        r"/professional/",
        r"/account/newsletters",
        r"/market-data/quotes",
        r"/quote/",
        r"realstock/company",
        r"vip\.stock\.finance\.sina\.com\.cn",
        r"suggest/lookup",
        r"fund_center",
        r"desktopapp|/mobile/|comfinanceweb",
        r"toujiao\.sina|tousu\.sina|zhongce\.sina",
        r"/games/",
        r"#comments",
        r"/zt_d/subject-",
        r"/stock/jyts",
        r"/stock/cpbd",
        r"/stock/sl/",
        r"/futures/quotes",
        r"/option/quotes",
        r"/fund/web/fundGoods",
        r"/focus/zqbjh",
        r"/data/#",
        r"/stock/reportch/",
        r"/realstock/income_statement",
        r"i\.finance\.sina\.com\.cn/zixuan",
        r"guba\.sina\.com\.cn",
        r"stock\.finance\.sina\.com\.cn",
    )
)

NON_NEWS_TITLES = {
    "APP",
    "Barron's",
    "Bloomberg",
    "Bloomberg Opinion",
    "Businessweek",
    "DAX",
    "ESG",
    "ETF期权",
    "Euro",
    "FTSE 100",
    "Finance",
    "Gold",
    "IBD",
    "Industries",
    "Live TV",
    "MarketWatch",
    "Markets",
    "More",
    "Nasdaq",
    "Opinion",
    "Politics",
    "Pound/Dollar",
    "Podcasts",
    "Podcasts View All",
    "S&P 500",
    "Search",
    "Subscribe",
    "Tech",
    "Video",
    "WSJ",
    "WSJ | Buy Side",
    "View All",
    "买基金",
    "保险",
    "博客",
    "财经客户端PC版",
    "财报查询",
    "成交额",
    "创业板指",
    "跌幅榜",
    "港股",
    "个股诊断",
    "股价预测",
    "股民维权",
    "股票",
    "股票开户",
    "行情中心",
    "换手率",
    "黄金",
    "贵金属",
    "基金",
    "基金净值",
    "基金排名",
    "基金超市",
    "机构荐股",
    "交易提示",
    "金麒麟研报",
    "酒业内参",
    "科创综指",
    "会议",
    "理财",
    "美股",
    "免费试用",
    "我的自选",
    "企业MSCI ESG评级数据",
    "期货",
    "期指",
    "千股千评",
    "上证综指",
    "手机版",
    "深证成指",
    "沪港通",
    "环球股指",
    "环球市场>>",
    "数据",
    "数据中心",
    "商品",
    "投教基地",
    "投研中心",
    "外汇",
    "新浪财经",
    "新浪财经APP",
    "新浪财经芝麻AI",
    "新浪科技",
    "新浪首页",
    "信托",
    "新股",
    "学投资",
    "医药",
    "银行",
    "原油",
    "债券",
    "涨幅榜",
    "直播",
    "专栏",
    "证券报",
    "资金流向",
    "自选股",
    "全球实时播报",
    "区块链",
    "区块链|",
    "区块链| ",
    "吧",
}

LINK_TITLE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^\d+$",
        r"^\d{1,2}:\d{2}$",
        r"^\d+\s+(minutes?|hours?|mins?) ago\b",
        r"\bcurrent percentage change\b",
        r"\b(Reuters|Getty Images|Shutterstock|AFP)\b",
        r"\bfor WSJ\b",
        r"^(B500|CAC 40|Crude Oil|FTSE|IBEX|KOSPI|SX5P|US 10 Yr)\b",
        r"(客户端|APP|下载|登录|注册|订阅|搜索|广告|Advertisement)$",
    )
)

UTILITY_TEXT_RE = re.compile(r'\bStaticText "(登录|注册|订阅|搜索|Advertisement|广告|More)"', re.IGNORECASE)
STATIC_NOISE_RE = re.compile(
    r'\b(?:StaticText|generic) "('
    r'[+\-–]?\d[\d,.]*[%％]?|'
    r'%|'
    r'\d{1,2}:\d{2}|'
    r'KRW|'
    r'Current Time\s*|'
    r'Latest Updates|'
    r'Related|'
    r'Live|'
    r'Watch|'
    r'以下为热门股票|'
    r'\s+'
    r')"',
    re.IGNORECASE,
)


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _role(line: str) -> str:
    match = NODE_RE.match(line)
    return match.group("role") if match else ""


def _link_title(line: str) -> str:
    match = LINK_TITLE_RE.search(line)
    if not match:
        return ""
    return match.group("title") or ""


def _link_url(line: str) -> str:
    match = LINK_URL_RE.search(line)
    if not match:
        return ""
    return match.group("url")


def _drop_line_with_subtree(line: str) -> bool:
    role = _role(line)
    if role in CONTAINER_ROLES_TO_DROP or role in LEAF_ROLES_TO_DROP:
        return True
    if any(pattern.search(line) for pattern in CONTAINER_TEXT_PATTERNS):
        return True
    if LINK_RE.search(line):
        title = _link_title(line).strip()
        url = _link_url(line).strip()
        if not title:
            return True
        if title in NON_NEWS_TITLES:
            return True
        if any(pattern.search(title) for pattern in LINK_TITLE_PATTERNS):
            return True
        if any(pattern.search(url) for pattern in LINK_URL_PATTERNS):
            return True
    return False


def _drop_line_only(line: str) -> bool:
    return bool(UTILITY_TEXT_RE.search(line) or STATIC_NOISE_RE.search(line))


def clean_snapshot(snapshot_text: str, max_length: int | None = None) -> str:
    """Remove obvious non-news nodes and optionally limit the cleaned snapshot."""
    if max_length is not None and max_length < 0:
        raise ValueError("max_length must be non-negative")

    kept_lines: list[str] = []
    seen_link_titles: set[str] = set()
    drop_children_until_indent: int | None = None

    for line in snapshot_text.splitlines():
        line_indent = _indent(line)
        if drop_children_until_indent is not None:
            if line_indent > drop_children_until_indent:
                continue
            drop_children_until_indent = None

        if _drop_line_with_subtree(line):
            drop_children_until_indent = line_indent
            continue
        if _drop_line_only(line):
            continue
        if LINK_RE.search(line):
            title = _link_title(line).strip()
            if title:
                if title in seen_link_titles:
                    drop_children_until_indent = line_indent
                    continue
                seen_link_titles.add(title)
        kept_lines.append(line)

    cleaned_snapshot = "\n".join(kept_lines) + ("\n" if snapshot_text.endswith("\n") else "")
    if max_length is not None and len(cleaned_snapshot) > max_length:
        logger.warning(
            "cleaned Chrome MCP snapshot exceeds max_length and will be truncated: chars=%d max_length=%d",
            len(cleaned_snapshot),
            max_length,
        )
        return cleaned_snapshot[:max_length] + TRUNCATION_SUFFIX
    return cleaned_snapshot

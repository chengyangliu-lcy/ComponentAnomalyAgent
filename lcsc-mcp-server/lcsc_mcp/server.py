from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .client import (
    LCSCClient,
    RateLimitError,
    format_detail,
    format_jlcpcb_results,
    format_search_results,
)

mcp = FastMCP(
    "lcsc-component-library",
    instructions=(
        "这是一个嘉立创(LCSC/JLCPCB)电子元器件查询工具。"
        "可以帮助用户搜索元器件、查看技术规格、获取数据手册链接。"
        "当用户询问电子元器件相关问题时使用这些工具。\n\n"
        "重要：嘉立创 API 有频率限制。每次只调用一个工具，"
        "不要并行调用多个搜索请求。如果收到限流错误，请等待后重试。"
    ),
)

client = LCSCClient()


@mcp.tool()
def lcsc_search(keyword: str, page: int = 1, page_size: int = 10) -> str:
    """搜索 LCSC 嘉立创元器件库。

    输入关键词（元器件型号、参数描述等），返回匹配的元器件列表。
    每条结果包含：LCSC C编号、名称、型号、厂商、封装、库存、阶梯价格。

    Args:
        keyword: 搜索关键词，如 "STM32F103"、"100nF 0402"、"LM358"
        page: 页码，从1开始
        page_size: 每页数量，1-50
    """
    page_size = max(1, min(50, page_size))
    page = max(1, page)
    try:
        data = client.search(keyword, page=page, page_size=page_size)
        return format_search_results(data)
    except RateLimitError as e:
        return (
            f"嘉立创 API 请求过于频繁，已被限流。"
            f"请等待 {e.retry_after:.0f} 秒后重试，或减少并发查询次数。"
        )
    except Exception as e:
        return f"搜索失败: {e}"


@mcp.tool()
def lcsc_detail(product_code: str) -> str:
    """根据 LCSC C 编号获取元器件详细信息。

    返回完整的技术参数、阶梯价格、库存、数据手册链接、RoHS状态等。
    C编号格式为 "C" + 数字，如 "C123456"。

    Args:
        product_code: LCSC C编号，如 "C123456"
    """
    product_code = product_code.strip().upper()
    if not product_code.startswith("C"):
        product_code = "C" + product_code
    try:
        data = client.get_detail(product_code)
        return format_detail(data)
    except RateLimitError as e:
        return (
            f"嘉立创 API 请求过于频繁，已被限流。"
            f"请等待 {e.retry_after:.0f} 秒后重试。"
        )
    except Exception as e:
        return f"查询失败: {e}"


@mcp.tool()
def lcsc_datasheet(product_code: str) -> str:
    """根据 LCSC C 编号获取元器件数据手册下载链接。

    返回官方数据手册(PDF)的URL，用户可直接下载查看完整的电气规格、
    引脚定义、应用电路等信息。

    Args:
        product_code: LCSC C编号，如 "C123456"
    """
    product_code = product_code.strip().upper()
    if not product_code.startswith("C"):
        product_code = "C" + product_code
    try:
        url = client.get_datasheet_url(product_code)
        if url:
            return f"数据手册下载链接: {url}"
        return "该元器件暂无数据手册链接。"
    except RateLimitError as e:
        return (
            f"嘉立创 API 请求过于频繁，已被限流。"
            f"请等待 {e.retry_after:.0f} 秒后重试。"
        )
    except Exception as e:
        return f"查询失败: {e}"


@mcp.tool()
def jlcpcb_parts_search(keyword: str, page: int = 1, page_size: int = 10) -> str:
    """搜索 JLCPCB SMT 贴片零件库。

    JLCPCB 的 SMT 贴片加工服务使用的零件库。返回零件的 C 编号、型号、
    是否 Basic（免费贴片）或 Extended（额外收费）、库存等信息。
    适用于查询 JLCPCB 贴片加工可用的零件。

    Args:
        keyword: 搜索关键词，如 "STM32F103"、"0402 100nF"
        page: 页码，从1开始
        page_size: 每页数量，1-50
    """
    page_size = max(1, min(50, page_size))
    page = max(1, page)
    try:
        data = client.search(keyword, page=page, page_size=page_size)
        return format_jlcpcb_results(data)
    except RateLimitError as e:
        return (
            f"嘉立创 API 请求过于频繁，已被限流。"
            f"请等待 {e.retry_after:.0f} 秒后重试，或减少并发查询次数。"
        )
    except Exception as e:
        return f"搜索失败: {e}"


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

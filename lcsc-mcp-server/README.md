# LCSC MCP Server

嘉立创(LCSC/JLCPCB)电子元器件查询 MCP Server。让 AI 助手能够搜索元器件、查看技术规格、获取数据手册。

兼容 Claude Code、OpenAI Codex CLI、OpenCode 等支持 MCP 协议的 AI 工具。

## 功能

| 工具 | 说明 |
|------|------|
| `lcsc_search` | 按关键词搜索 LCSC 元器件库 |
| `lcsc_detail` | 按 C 编号获取元器件详细规格 |
| `lcsc_datasheet` | 获取元器件数据手册 PDF 链接 |
| `jlcpcb_parts_search` | 搜索 JLCPCB SMT 贴片零件库 |

## 前置条件

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (推荐) 或 pip

## 快速安装

### 方法 1: 使用 uv (推荐)

```bash
# 进入 lcsc-mcp-server 目录
cd lcsc-mcp-server

# 安装到当前环境
uv pip install -e .

# 添加到 Claude Code
claude mcp add lcsc -- uv run lcsc-mcp
```

### 方法 2: 使用 pip

```bash
cd lcsc-mcp-server
pip install -e .
claude mcp add lcsc -- lcsc-mcp
```

### 方法 3: 运行安装脚本

```bash
# Linux/Mac
bash install_claude.sh
# Windows
install_claude.bat
```

## 跨工具配置

安装包后，配置各 AI 工具：

### Claude Code
```bash
claude mcp add lcsc -- uv run lcsc-mcp
```

### Codex CLI
编辑 `~/.codex/config.json`：
```json
{
  "mcp_servers": {
    "lcsc": {
      "command": "uv",
      "args": ["run", "lcsc-mcp"]
    }
  }
}
```

### OpenCode
编辑 `~/.opencode/config.json`：
```json
{
  "mcpServers": {
    "lcsc": {
      "command": "uv",
      "args": ["run", "lcsc-mcp"]
    }
  }
}
```

## 使用示例

安装后，在 AI 助手中直接用自然语言询问：

- "帮我查一下 STM32F103C8T6 在嘉立创的价格和库存"
- "C123456 的详细参数是什么？"
- "给我 LM358 的 datasheet 链接"
- "JLCPCB 有哪些 0402 100nF 电容可以贴片？"

## 注意事项

- 嘉立创 API 有频率限制，工具内置了限速和重试机制
- 请勿并行调用多个搜索请求，逐个调用更稳妥
- 价格和库存实时变化，以官网为准

## 项目结构

```
lcsc-mcp-server/
  pyproject.toml        # 包配置和依赖
  README.md             # 本文件
  install_claude.sh     # Claude Code 安装脚本 (Linux/Mac)
  install_claude.bat    # Claude Code 安装脚本 (Windows)
  install_codex.sh      # Codex CLI 安装脚本
  install_opencode.sh   # OpenCode 安装脚本
  lcsc_mcp/             # Python 包
    __init__.py
    client.py           # LCSC/JLCPCB API 客户端
    server.py           # MCP Server 定义
```

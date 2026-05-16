@echo off
REM Install LCSC MCP Server for Claude Code (Windows)
REM Prerequisites: uv installed (https://docs.astral.sh/uv/)
REM Usage: install_claude.bat

echo Installing LCSC MCP Server for Claude Code...

REM Install the package into current environment
uv pip install -e "%~dp0"

REM Add MCP server to Claude Code
claude mcp add lcsc -- uv run lcsc-mcp

echo.
echo Done! Restart Claude Code to use the LCSC component tools.
echo Tools: lcsc_search, lcsc_detail, lcsc_datasheet, jlcpcb_parts_search
pause

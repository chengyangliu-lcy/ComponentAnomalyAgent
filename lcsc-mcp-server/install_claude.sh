#!/bin/bash
# Install LCSC MCP Server for Claude Code
# Prerequisites: uv installed (https://docs.astral.sh/uv/)
# Usage: bash install_claude.sh

echo "Installing LCSC MCP Server for Claude Code..."

# Install the package into current environment
uv pip install -e "$(dirname "$0")"

# Add MCP server to Claude Code
claude mcp add lcsc -- uv run lcsc-mcp

echo ""
echo "Done! Restart Claude Code to use the LCSC component tools."
echo "Tools: lcsc_search, lcsc_detail, lcsc_datasheet, jlcpcb_parts_search"

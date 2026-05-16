#!/bin/bash
# Install LCSC MCP Server for Codex CLI
# Prerequisites: uv installed (https://docs.astral.sh/uv/)
# Usage: bash install_codex.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="$HOME/.codex"
CONFIG_FILE="$CONFIG_DIR/config.json"

echo "Installing LCSC MCP Server for Codex CLI..."

# Install the package
uv pip install -e "$SCRIPT_DIR"

mkdir -p "$CONFIG_DIR"

if [ -f "$CONFIG_FILE" ]; then
    python3 -c "
import json
with open('$CONFIG_FILE', 'r') as f:
    config = json.load(f)
if 'mcp_servers' not in config:
    config['mcp_servers'] = {}
config['mcp_servers']['lcsc'] = {
    'command': 'uv',
    'args': ['run', 'lcsc-mcp']
}
with open('$CONFIG_FILE', 'w') as f:
    json.dump(config, f, indent=2)
print('Updated config.json')
"
else
    cat > "$CONFIG_FILE" << EOF
{
  "mcp_servers": {
    "lcsc": {
      "command": "uv",
      "args": ["run", "lcsc-mcp"]
    }
  }
}
EOF
    echo "Created config.json"
fi

echo "Done! Restart Codex CLI to use the LCSC component tools."

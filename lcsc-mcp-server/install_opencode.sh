#!/bin/bash
# Install LCSC MCP Server for OpenCode
# Prerequisites: uv installed (https://docs.astral.sh/uv/)
# Usage: bash install_opencode.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="$HOME/.opencode"
CONFIG_FILE="$CONFIG_DIR/config.json"

echo "Installing LCSC MCP Server for OpenCode..."

# Install the package
uv pip install -e "$SCRIPT_DIR"

mkdir -p "$CONFIG_DIR"

if [ -f "$CONFIG_FILE" ]; then
    python3 -c "
import json
with open('$CONFIG_FILE', 'r') as f:
    config = json.load(f)
if 'mcpServers' not in config:
    config['mcpServers'] = {}
config['mcpServers']['lcsc'] = {
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
  "mcpServers": {
    "lcsc": {
      "command": "uv",
      "args": ["run", "lcsc-mcp"]
    }
  }
}
EOF
    echo "Created config.json"
fi

echo "Done! Restart OpenCode to use the LCSC component tools."

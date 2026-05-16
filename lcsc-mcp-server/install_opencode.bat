@echo off
REM Install LCSC MCP Server for OpenCode (Windows)
REM Usage: install_opencode.bat

set SCRIPT_DIR=%~dp0
set SERVER_PATH=%SCRIPT_DIR%server.py
set CONFIG_DIR=%USERPROFILE%\.opencode
set CONFIG_FILE=%CONFIG_DIR%\config.json

echo Installing LCSC MCP Server for OpenCode...
echo Server path: %SERVER_PATH%

if not exist "%CONFIG_DIR%" mkdir "%CONFIG_DIR%"

if exist "%CONFIG_FILE%" (
    echo Existing config found. Merging...
    python -c "import json; f=open(r'%CONFIG_FILE%');c=json.load(f);f.close();c.setdefault('mcpServers',{})['lcsc']={'command':'python','args':[r'%SERVER_PATH%']};f=open(r'%CONFIG_FILE%','w');json.dump(c,f,indent=2);f.close();print('Updated config.json')"
) else (
    echo Creating new config...
    python -c "import json;json.dump({'mcpServers':{'lcsc':{'command':'python','args':[r'%SERVER_PATH%']}}},open(r'%CONFIG_FILE%','w'),indent=2);print('Created config.json')"
)

echo.
echo Done! Restart OpenCode to use the LCSC component tools.
pause

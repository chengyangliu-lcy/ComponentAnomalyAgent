@echo off
chcp 65001 >nul 2>&1
setlocal

set SCRIPT_DIR=%~dp0
set CONFIG_DIR=%USERPROFILE%\.codex
set CONFIG_FILE=%CONFIG_DIR%\config.json

echo === LCSC MCP Server - Codex CLI 全局安装 ===

where node >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到 node，请先安装 Node.js ^>= 18
    pause
    exit /b 1
)

echo 正在全局安装 lcsc-mcp-server...
cd /d "%SCRIPT_DIR%" && call npm install -g .

if not exist "%CONFIG_DIR%" mkdir "%CONFIG_DIR%"

if exist "%CONFIG_FILE%" (
    python -c "import json;p=r'%CONFIG_FILE%';c=json.load(open(p));c.setdefault('mcp_servers',{})['lcsc']={'command':'lcsc-mcp'};json.dump(c,open(p,'w'),indent=2);print('已更新')"
) else (
    python -c "import json;json.dump({'mcp_servers':{'lcsc':{'command':'lcsc-mcp'}}},open(r'%CONFIG_FILE%','w'),indent=2);print('已创建')"
)

echo.
echo 安装完成！重启 Codex CLI 后所有项目均可使用。
pause

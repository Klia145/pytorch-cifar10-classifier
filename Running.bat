@echo off
REM --- 第0步：解决中文乱码问题 ---
chcp 65001 > nul

:: --- 第1步：定义我们需要的Conda环境名称 ---
SET ENV_NAME=cnn_env

echo --- 欢迎使用 屿的 "猜猜看" 小游戏启动器 ---
echo.
echo --- 正在激活 Conda 环境: %ENV_NAME% ---

:: --- 第2步：激活正确的Conda环境 ---
call conda activate %ENV_NAME%

:: --- 这是一个安全检查，确保环境已经被正确安装 ---
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo !!! 错误: 无法激活 Conda 环境 '%ENV_NAME%'。 !!!
    echo !!! 请先运行 'setup_environment.bat' 来配置环境。 !!!
    echo.
    pause
    exit /b 1
)

echo --- Conda 环境已激活 ---
echo.

:: --- 第3步：切换到脚本所在的目录 (确保能找到 app.py) ---
cd /d %~dp0

echo --- 正在启动 "猜猜看" Web 应用... ---
echo 当前工作目录是: %cd%
echo.

:: --- 第4步：执行你的 app.py 脚本 ---
python app.py

echo.
echo --- Web 应用已关闭。按任意键退出此终端。 ---
pause
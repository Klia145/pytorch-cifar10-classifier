@echo off
chcp 65001 > nul

:: --- 1. 定义我们需要的环境名称 ---
SET ENV_NAME=cnn_env

echo --- 欢迎使用屿的CNN项目环境部署工具 ---
echo.
echo --- 正在检查是否存在 Conda 环境: %ENV_NAME% ---


conda env list | findstr /C:"%ENV_NAME%" > nul

IF %ERRORLEVEL% NEQ 0 (
    echo --- 环境 '%ENV_NAME%' 不存在, 正在为您创建... ---
    conda create --name %ENV_NAME% python=3.9 -y
    IF %ERRORLEVEL% NEQ 0 (
        echo.
        echo !!! 错误: 创建 Conda 环境失败。!!!
        echo !!! 请确保您已正确安装 Anaconda/Miniconda。!!!
        goto :eof
    )
    echo --- 环境 '%ENV_NAME%' 创建成功! ---
) ELSE (
    echo --- 环境 '%ENV_NAME%' 已存在, 无需创建。---
)

echo.
echo --- 3. 激活 Conda 环境 ---
call conda activate %ENV_NAME%

-
cd /d %~dp0

echo.
echo --- 4. 正在根据 requirements.txt 安装/更新依赖库... ---
pip install -r requirements.txt

IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo !!! 错误: 依赖库安装失败。请检查您的网络连接或 requirements.txt 文件。!!!
    goto :eof
)

echo.
echo ===================================================
echo  环境配置成功!
echo.
echo  现在您可以运行 '训练CNN.bat' 或 '评估MiniVGG.bat' 等脚本了。
echo ===================================================
echo.

:eof
pause
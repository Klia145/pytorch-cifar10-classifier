@echo off
cd /d %~dp0..
chcp 65001 > nul

REM 
echo --- 正在激活 Conda 环境 ---
call conda activate cnn_env

REM 
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo !!! 错误: 无法激活 Conda 环境 'neuopt'。 !!!
    echo !!! 请确保你已经正确安装 Anaconda/Miniconda 并且环境存在。!!!
    echo.
    pause
    exit /b 1
)

echo --- 正在启动 MiniVGG 评估任务---
echo 当前工作目录是: %cd%
REM 
python ./CNN.PY --model MiniVGG  --mode eval

echo.
echo --- 任务完成，按任意键退出 ---
pause
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

call conda activate neuopt
echo --- 正在启动 MiniVGG 训练任务 (20个周期) ---
echo 当前工作目录是: %cd%
REM 
python ./CNN.PY --model MiniVGG --epochs 10 --mode train

echo.
echo --- 任务完成，按任意键退出 ---
pause
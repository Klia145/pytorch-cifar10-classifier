@echo off
chcp 65001 > nul

:: --- 1. 定义项目所需的变量 ---
SET ENV_NAME=cnn_env
SET REQUIREMENTS_FILE=requirements.txt
SET DATA_DIR=data
SET DATA_FILENAME=cifar-10-python.tar.gz
SET DATA_URL=https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
SET DATA_FULL_PATH=%DATA_DIR%\%DATA_FILENAME%

echo --- =================================================== ---
echo ---  欢迎使用 屿的 CNN 项目一键部署工具
echo --- =================================================== ---
echo.

:: --- 2. 检查并创建 Conda 环境 ---
echo --- [步骤 1/3] 正在检查并配置 Conda 环境: %ENV_NAME% ---
conda env list | findstr /C:"%ENV_NAME%" > nul
IF %ERRORLEVEL% NEQ 0 (
    echo   环境 '%ENV_NAME%' 不存在, 正在为您创建...
    conda create --name %ENV_NAME% python=3.9 -y
    IF %ERRORLEVEL% NEQ 0 (
        echo. & echo !!! 错误: 创建 Conda 环境失败。请确保您已正确安装 Anaconda/Miniconda。!!! & goto :error
    )
    echo   环境 '%ENV_NAME%' 创建成功!
) ELSE (
    echo   环境 '%ENV_NAME%' 已存在, 无需创建。
)
echo.

:: --- 3. 激活环境并安装依赖 ---
call conda activate %ENV_NAME%
echo --- [步骤 2/3] 正在根据 %REQUIREMENTS_FILE% 安装/更新依赖库... ---
pip install -r %REQUIREMENTS_FILE%
IF %ERRORLEVEL% NEQ 0 (
    echo. & echo !!! 错误: 依赖库安装失败。请检查您的网络连接或 %REQUIREMENTS_FILE% 文件。!!! & goto :error
)
echo   依赖库已是最新状态。
echo.

:: --- 4. 检查并下载数据集 ---
echo --- [步骤 3/3] 正在检查并下载数据集... ---
cd /d %~dp0..  :: 切换到项目根目录
IF NOT EXIST "%DATA_DIR%" (
    echo   文件夹 '%DATA_DIR%' 不存在, 正在创建...
    mkdir "%DATA_DIR%"
)
IF EXIST "%DATA_FULL_PATH%" (
    echo   数据集 '%DATA_FILENAME%' 已存在。无需下载。
) ELSE (
    echo   数据集不存在, 准备开始下载...
    echo   从: %DATA_URL%
    echo   保存至: %DATA_FULL_PATH%
    echo   下载过程可能需要几分钟，请耐心等待...
    echo.
    curl -o "%DATA_FULL_PATH%" "%DATA_URL%"
    IF %ERRORLEVEL% EQU 0 (
        echo. & echo   --- 下载成功! ---
    ) ELSE (
        echo. & echo !!! 错误: 下载失败。请检查您的网络连接。!!! & goto :error
    )
)
echo.

:: --- 成功结束 ---
echo ===================================================
echo  所有环境和数据已准备就绪!
echo.
echo  您现在可以双击运行 "训练..." 或 "评估..." 等 .bat 脚本了。
echo ===================================================
echo.
goto :eof

:error
echo.
echo !!! 部署过程中发生错误，请检查上面的提示信息。!!!
echo.

:eof
pause
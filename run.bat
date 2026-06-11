@echo off
chcp 65001 >nul
title Pose Analysis System

echo ========================================
echo   POSE ANALYSIS SYSTEM v2.0
echo   YOLOv8-Pose Engine
echo ========================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 未找到 Python，请先安装 Python 3.10+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [*] 检查依赖...
python -c "import fastapi, uvicorn, ultralytics, cv2, torch, numpy" >nul 2>&1
if errorlevel 1 (
    echo [!] 缺少依赖，正在安装...
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
    if errorlevel 1 (
        echo [!] 镜像源安装失败，尝试默认源...
        pip install -r requirements.txt
    )
)

echo [*] 启动服务...
echo.
echo 浏览器打开: http://localhost:8080
echo 按 Ctrl+C 停止服务
echo.

python api.py

pause

@echo off
SETLOCAL EnableExtensions

REM Check if venv directory exists
IF NOT EXIST venv (
    echo [INFO] Virtual environment not found. Creating one...
    python -m venv venv
    IF ERRORLEVEL 1 (
        echo [ERROR] Failed to create virtual environment. Please ensure Python is installed and added to PATH.
        pause
        exit /b 1
    )
    
    echo [INFO] Activating virtual environment...
    call venv\Scripts\activate.bat
    
    echo [INFO] Installing required dependencies...
    pip install -r requirements.txt --no-cache-dir
    IF ERRORLEVEL 1 (
        echo [ERROR] Dependency installation failed.
        pause
        exit /b 1
    )
) ELSE (
    echo [INFO] Virtual environment found. Activating...
    call venv\Scripts\activate.bat
)

echo [INFO] Starting YOLO Streamlit App...
streamlit run app.py
if ERRORLEVEL 1 (
    echo [ERROR] Failed to start Streamlit app.
    pause
    exit /b 1
)

ENDLOCAL

@echo off
setlocal
call .venv\Scripts\activate.bat
python run_pipeline.py
if errorlevel 1 pause & exit /b 1
python -m uvicorn backend.main:app --reload
endlocal

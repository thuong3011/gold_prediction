@echo off
call .venv\Scripts\activate.bat
python -m uvicorn backend.main:app --reload

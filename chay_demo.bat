@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Demo - Phat hien bai bao sai tac gia
set "CONDA="
for %%P in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\Anaconda3" "%USERPROFILE%\miniconda3" "C:\ProgramData\anaconda3" "C:\ProgramData\Anaconda3" "%LOCALAPPDATA%\anaconda3" "C:\anaconda3") do (
  if exist "%%~P\Scripts\activate.bat" if not defined CONDA set "CONDA=%%~P"
)
if defined CONDA (
  echo Dung Anaconda tai: %CONDA%
  call "%CONDA%\Scripts\activate.bat" "%CONDA%"
)
python -m streamlit --version >nul 2>&1
if errorlevel 1 (
  echo Chua co streamlit, dang cai dat...
  python -m pip install streamlit
)
echo Dang mo demo tai http://localhost:8501 ...
python -m streamlit run app_demo.py --server.port 8501
pause

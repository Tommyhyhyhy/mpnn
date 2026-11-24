@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Day code len GitHub
echo Dang day code len https://github.com/Tommyhyhyhy/mpnn ...
echo.
git add -A
git commit -m "Đồ án Mô phỏng ngẫu nhiên: Phát hiện bài báo bị ghi sai tác giả bằng NLP" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01UyPjjYus7vEjanxWbhnuFK"
git branch -M main
git push -u origin main
echo.
if errorlevel 1 (
  echo [LOI] Chua day len duoc. Chup man hinh cua so nay gui cho Claude.
) else (
  echo [XONG] Mo https://github.com/Tommyhyhyhy/mpnn de kiem tra.
)
pause

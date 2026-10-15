@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python 3.10+ не найден. Установите с python.org и повторите.
  pause
  exit /b 1
)
if not exist .venv\Scripts\activate.bat python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if not exist assets\icon.ico python tools_make_icon.py
pyinstaller --noconfirm --clean --onefile --windowed --name VocalStudio --icon assets\icon.ico ^
  --exclude-module tkinter --exclude-module matplotlib --exclude-module IPython --exclude-module pytest ^
  --exclude-module PyQt6.QtWebEngineCore --exclude-module PyQt6.QtQml --exclude-module PyQt6.QtQuick ^
  --exclude-module PyQt6.QtNetwork --exclude-module PyQt6.QtSql --exclude-module PyQt6.QtMultimedia ^
  --exclude-module PyQt6.QtBluetooth --exclude-module PyQt6.QtPdf --exclude-module PyQt6.QtSvg ^
  run.py
echo.
echo Готово: dist\VocalStudio.exe
pause

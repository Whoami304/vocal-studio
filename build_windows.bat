@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo Python 3.10+ не найден. Установите с python.org ^(галочка "Add python.exe to PATH"^) и повторите.
  pause
  exit /b 1
)

if not exist .venv\Scripts\activate.bat python -m venv .venv
call .venv\Scripts\activate.bat

python -m pip install --upgrade pip
if errorlevel 1 goto fail
python -m pip install -r requirements.txt
if errorlevel 1 goto fail

if exist VocalStudio.spec del VocalStudio.spec
set ICON=
if not exist assets\icon.ico python tools_make_icon.py
if exist assets\icon.ico set ICON=--icon assets\icon.ico

python -m PyInstaller --noconfirm --clean --onefile --windowed --name VocalStudio %ICON% ^
  --exclude-module tkinter --exclude-module matplotlib --exclude-module IPython --exclude-module pytest ^
  --exclude-module PyQt6.QtWebEngineCore --exclude-module PyQt6.QtQml --exclude-module PyQt6.QtQuick ^
  --exclude-module PyQt6.QtNetwork --exclude-module PyQt6.QtSql --exclude-module PyQt6.QtMultimedia ^
  --exclude-module PyQt6.QtBluetooth --exclude-module PyQt6.QtPdf --exclude-module PyQt6.QtSvg ^
  run.py
if errorlevel 1 goto fail

echo.
echo Готово: dist\VocalStudio.exe
pause
exit /b 0

:fail
echo.
echo Сборка не удалась. Если ошибка при установке пакетов — поставьте Python 3.12 и повторите.
pause
exit /b 1

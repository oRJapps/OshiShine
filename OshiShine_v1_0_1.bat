@echo off
rem バッチファイルがあるフォルダ（D:\OshiShineApp\）に移動
cd /d "%~dp0"

rem 仮想環境(venv)を起動
call venv\Scripts\activate.bat

rem 【裏技】pythonw を使うと、黒い窓を残さずにアプリだけ起動できるぜ！
start pythonw Oshi_Shine.py

rem バッチファイル自体はここで終了（黒い窓が閉じる）
exit
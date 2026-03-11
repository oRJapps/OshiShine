@echo off
title OshiShine GOD Edition - Setup ^& Launch
cd /d "%~dp0"

echo ==========================================
echo  OshiShine GOD Edition
echo  自動セットアップ＆起動ツール
echo ==========================================
echo.

:: 1. Pythonがインストールされているかチェック
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [エラー] Pythonが見つかりません
    echo PCに Python (3.10推奨) をインストールして、
    echo インストール時の「Add Python to PATH」にチェックを入れてから出直してくれ。
    echo.
    pause
    exit /b
)

:: 2. 仮想環境(venv)が存在しない場合は自動作成
if not exist venv\Scripts\activate.bat (
    echo [情報] 仮想環境(venv)が見つからないな。新しく構築するぜ！
    echo 初回は少し時間がかかるから、コーヒーでも飲んで待っててくれ。
    python -m venv venv
)

:: 3. 仮想環境を起動
call venv\Scripts\activate.bat

:: 4. 必要なライブラリのインストール (requirements.txt から)
echo [情報] ライブラリをチェック・インストール中だ...
python -m pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt

:: 5. AIモデルの存在チェック（超重要！）
if not exist model\dpt_beit_large_384.pt (
    echo.
    echo ==========================================
    echo [警告] 待ってくれ！AIモデルファイルが見つかりません
    echo.
    echo 'model' フォルダの中に 'dpt_beit_large_384.pt' を入れてくれ。
    echo ダウンロード先や手順は README.md を確認してくれよな。
    echo ==========================================
    echo.
    pause
    exit /b
)

:: 6. プログラムの実行
echo.
echo [情報] 準備完了。OshiShineAppを起動します。
:: 黒い窓を残さずに起動
start pythonw Oshi_Shine.py

exit
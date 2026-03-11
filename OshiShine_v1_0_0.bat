:: バッチファイルがあるフォルダ（D:\OshiShineApp\）に移動
cd /d "%~dp0"

:: 仮想環境(venv)を起動
call venv\Scripts\activate.bat

:: 【裏技】pythonw を使うと、黒い窓を残さずにアプリだけ起動できるぜ！
start pythonw Oshi_Shine.py

:: バッチファイル自体はここで終了（黒い窓が閉じる）
exit
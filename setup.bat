@echo off
echo ピクセルアート生成システム セットアップ
echo =========================================

cd /d "%~dp0"

if not exist "..\\.venv\\Scripts\\pip.exe" (
    echo 仮想環境が見つかりません。プロジェクトルートで setup.sh を実行してください。
    pause
    exit /b 1
)

echo 必要パッケージをインストール中...
"..\\.venv\\Scripts\\pip.exe" install -r requirements.txt

echo.
echo セットアップ完了！
echo.
echo 使い方:
echo   ..\\.venv\\Scripts\\python.exe generate_pixel_art.py "a cute cat" --name cat
echo   ..\\.venv\\Scripts\\python.exe generate_pixel_art.py --skip-generate --input path\to\image.png --name my_sprite
echo.
echo GEMINI_API_KEY 環境変数を設定してから実行してください。
pause

@echo off
echo.
echo ============================================
echo   Tomato Disease Detection - Web App
echo ============================================
echo.

REM Check if model exists
if not exist "models\best_predictor.pkl" (
    echo [ERROR] Trained model not found!
    echo.
    echo Please train the models first by running:
    echo   python main.py --data_dir data/raw --max_per_class 300
    echo.
    pause
    exit /b 1
)

echo [OK] Model found.
echo [OK] Starting web app...
echo.
echo Open your browser and go to:
echo   http://127.0.0.1:5000
echo.
echo Press Ctrl+C to stop the server.
echo.

python app.py

pause

@echo off
echo ==============================================
echo Barings Bank (1995) Fraud Detection Platform
echo ==============================================
echo Installing requirements...
python -m pip install -r requirements.txt

echo Bootstrapping the simulation and initial models...
python -m app.api.bootstrap

echo Starting FastAPI Backend (Port 8000)...
start "Barings API" cmd /c "uvicorn app.api.main:app --host 0.0.0.0 --port 8000"

echo Wait for API to start...
timeout /t 5 /nobreak >nul

echo Starting Streamlit Dashboard (Port 8501)...
start "Barings Dashboard" cmd /c "streamlit run app/dashboard/app.py"

echo Platform is running!
echo Access the Dashboard at: http://localhost:8501
echo Access the API Docs at: http://localhost:8000/docs

$env:FOREST_API_KEY = "forest"
.\.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000

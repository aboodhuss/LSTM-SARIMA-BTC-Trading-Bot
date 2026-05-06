# Run Guide

## Local Run

Backend:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Frontend:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## What Must Be Running

The React site needs the FastAPI backend. If the backend is not running, the dashboard will load but live candles, predictions, portfolio updates, and network tests will not work.

## For GitHub

Do not commit local generated folders:

- `.venv`
- `frontend/node_modules`
- `frontend/dist`
- `backend/__pycache__`

The root `.gitignore` already excludes them.

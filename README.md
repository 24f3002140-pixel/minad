# Minimal Adaptation and Repair a PEFT Run

## Run locally
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Endpoint
POST `/adapt`

Content-Type: application/json

## Deploy
For Render:
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

Enter the public service base URL only, for example:
`https://your-service.onrender.com`

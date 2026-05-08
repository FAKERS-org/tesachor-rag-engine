# micro service controller

# run docker compose
docker compose up -d postgres pgadmin redis

# run the micro services
wt `
"new-tab --title `"llm`" bash -lc 'cd services/llm && .venv/Scripts/activate && uv run uvicorn app:app --host 0.0.0.0 --port 8080 --reload'" `
"; split-pane -H --title `"embedding`" bash -lc 'cd services/embedding && .venv/Scripts/activate && uv run uvicorn app:app --host 0.0.0.0 --port 8081 --reload'" `
"; split-pane -V --title `"api`" bash -lc 'cd services/api && .venv/Scripts/activate && uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload'" `
"; focus-pane -t 1" `
"; split-pane -V --title `"ingestion`" bash -lc 'cd services/ingestion && .venv/Scripts/activate && uv run celery -A worker worker --loglevel=info --pool=solo'"
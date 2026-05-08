#!/bin/bash

# micro service controller

# run docker compose
docker compose up -d postgres pgadmin redis

# create tmux session
tmux new-session -d -s tesachor-dev

# =========================
# llm service
# =========================
tmux send-keys -t tesachor-dev "
cd services/llm &&
source .venv/bin/activate &&
uv run uvicorn app:app --host 0.0.0.0 --port 8002 --reload
" C-m

# =========================
# embedding service
# =========================
tmux split-window -h -t tesachor-dev

tmux send-keys -t tesachor-dev "
cd services/embedding &&
source .venv/bin/activate &&
uv run uvicorn app:app --host 0.0.0.0 --port 8080 --reload
" C-m

# =========================
# api service
# =========================
tmux select-pane -t 0
tmux split-window -v

tmux send-keys -t tesachor-dev "
cd services/api &&
source .venv/bin/activate &&
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
" C-m

# =========================
# ingestion worker
# =========================
tmux select-pane -t 1
tmux split-window -v

tmux send-keys -t tesachor-dev "
cd services/ingestion &&
source .venv/bin/activate &&
uv run celery -A worker worker --loglevel=info --pool=solo
" C-m

# organize layout
tmux select-layout tiled

# attach session
tmux attach-session -t tesachor-dev
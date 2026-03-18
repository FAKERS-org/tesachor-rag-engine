from fastapi import FastAPI
from src.api.v1.router import router as v1_router

app = FastAPI()

# This "mounts" your new file's routes into the main app
app.include_router(v1_router, prefix="/api/v1")
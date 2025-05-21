from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from forecast import router as forecast_router
from auth import router as auth_router
from history_logs import register_history_log_routes
from configurations import register_configuration_routes
from storage import router as storage_router
from fastapi.middleware.cors import CORSMiddleware
from model import Base
from db import engine, get_db

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)

# Mount static files directory
app.mount("/static", StaticFiles(directory="forecasts"), name="static")

app.include_router(forecast_router)
app.include_router(auth_router)
app.include_router(storage_router)
register_history_log_routes(app)
register_configuration_routes(app, get_db)

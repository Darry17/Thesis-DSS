from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from forecast import router as forecast_router
from auth import router as auth_router
import uuid
from history_logs import register_history_log_routes
from configurations import register_configuration_routes
from storage import router as storage_router
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware  # Updated import
from model import Base
from db import engine, get_db
import os
import shutil

# Define the middleware class
class TempCleanupMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, temp_folder: str):
        super().__init__(app)
        self.temp_folder = temp_folder
        os.makedirs(self.temp_folder, exist_ok=True)
        self.active_sessions = set()

    async def dispatch(self, request: Request, call_next):
        # Check if this is a protected path
        protected_paths = [
            "/upload",
            "/select-type",
            "/model-selection",
            "/configure-single",
            "/configure-hybrid",
            "/view-graph",
            "/result"
        ]
        
        # Generate or get session ID
        session_id = request.cookies.get("session_id") or str(uuid.uuid4())
        
        response = await call_next(request)
        
        # Only clear if not in protected paths and session is ending
        if request.url.path not in protected_paths:
            if session_id in self.active_sessions:
                self.active_sessions.remove(session_id)
                self.clear_temp_folder()
            else:
                self.active_sessions.add(session_id)
        
        # Set session cookie if not already set
        if "session_id" not in request.cookies:
            response.set_cookie(key="session_id", value=session_id)
            
        return response

    def clear_temp_folder(self):
        try:
            for filename in os.listdir(self.temp_folder):
                file_path = os.path.join(self.temp_folder, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
        except Exception as e:
            print(f'Failed to clear temp folder. Reason: {e}')

app = FastAPI()

# Add middlewares (order matters - add cleanup after CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TempCleanupMiddleware,
    temp_folder="temp"  # or your actual temp folder path
)

# Database setup
Base.metadata.create_all(bind=engine)

# Mount static files directory
app.mount("/static", StaticFiles(directory="forecasts"), name="static")

# Include routers
app.include_router(forecast_router)
app.include_router(auth_router)
app.include_router(storage_router)
register_history_log_routes(app)
register_configuration_routes(app, get_db)
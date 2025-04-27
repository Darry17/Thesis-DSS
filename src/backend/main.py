from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
import os
from dotenv import load_dotenv
import logging
import uvicorn

# Import components
from components.auth import register_auth_routes, Base as AuthBase, get_current_user, get_admin_user
from components.storage import register_storage_routes, Base as StorageBase
from components.forecast import register_forecast_routes, Base as ForecastBase, Forecast
from components.configurations import register_configuration_routes
from components.history_logs import register_history_log_routes, Base as HistoryBase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Enable CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
DATABASE_URL = f"mysql+mysqlconnector://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create all tables in the database
AuthBase.metadata.create_all(bind=engine)
StorageBase.metadata.create_all(bind=engine)
ForecastBase.metadata.create_all(bind=engine)
HistoryBase.metadata.create_all(bind=engine)

# Database session dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Register routes from components
register_auth_routes(app, get_db)
register_storage_routes(app, get_db)
register_forecast_routes(app, get_db)
register_configuration_routes(app, get_db, Forecast)
register_history_log_routes(app, get_db, get_current_user, get_admin_user, Forecast)

# Root route
@app.get("/")
async def root():
    return {"message": "API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
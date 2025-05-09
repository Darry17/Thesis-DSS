import os
from fastapi import FastAPI, HTTPException, Depends, Request
from sqlalchemy import Column, Integer, String, DateTime, text, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Optional
import logging
import pandas as pd
import numpy as np
from .models.dhr_solar_forecast import generate_forecast, fourier_transform, load_and_prepare_data
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# Create SQLAlchemy engine using the MySQL connection string
DATABASE_URL = f"mysql+mysqlconnector://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
engine = create_engine(DATABASE_URL, pool_recycle=3600)  # pool_recycle ensures long-lived connections are refreshed

# Create a sessionmaker bound to the engine
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a base class for declarative models
Base = declarative_base()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Storage configuration
BASE_STORAGE_PATH = os.getenv('STORAGE_PATH', os.path.join(os.path.dirname(__file__), '..', '..', '..', 'storage'))
os.makedirs(BASE_STORAGE_PATH, exist_ok=True)

# SQLAlchemy Models
Base = declarative_base()

# Dependency that provides a database session for each request
def get_db():
    db = SessionLocal()  # Create a new session
    try:
        yield db  # Yield the session for the route to use
    finally:
        db.close()  # Close the session when done

class Forecast(Base):
    __tablename__ = "forecasts"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=True)
    forecast_model = Column(String(50), nullable=False)
    steps = Column(String(50), nullable=False)
    granularity = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.now)

# Pydantic Models
class ForecastCreate(BaseModel):
    filename: str
    original_filename: Optional[str] = None
    forecast_model: str
    steps: str
    granularity: str
    model_config = {'from_attributes': True}

class ForecastRequest(BaseModel):
    forecast_id: int
    granularity: str

class HourlyData:
    def __init__(self, forecast_id, file_name):
        self.forecast_id = forecast_id
        self.file_name = file_name

# FastAPI setup
app = FastAPI()

# Routes
def register_forecast_routes(app: FastAPI, get_db):
    @app.post("/api/forecasts")
    async def create_forecast(
        forecast: ForecastCreate, 
        db: Session = Depends(get_db)
    ):
        try:
            logger.info(f"Creating forecast: {forecast}")
            
            db_forecast = Forecast(
                filename=forecast.filename,
                original_filename=forecast.original_filename,
                forecast_model=forecast.forecast_model,
                steps=forecast.steps,
                granularity=forecast.granularity
            )
            
            db.add(db_forecast)
            db.commit()
            db.refresh(db_forecast)

            logger.info(f"Created forecast with ID: {db_forecast.id}")

            return {
                "id": db_forecast.id,
                "filename": db_forecast.filename,
                "original_filename": db_forecast.original_filename,
                "forecast_model": db_forecast.forecast_model,
                "steps": db_forecast.steps,
                "granularity": db_forecast.granularity,
                "created_at": db_forecast.created_at
            }

        except Exception as e:
            logger.error(f"Error creating forecast: {str(e)}")
            db.rollback()
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/forecasts/{forecast_id}")
    async def get_forecast(forecast_id: int, db: Session = Depends(get_db)):
        try:
            forecast = db.query(Forecast).filter(Forecast.id == forecast_id).first()
            
            if not forecast:
                raise HTTPException(status_code=404, detail=f"Forecast {forecast_id} not found")
            
            return {
                "id": forecast.id,
                "filename": forecast.filename,
                "original_filename": forecast.original_filename,
                "model": forecast.forecast_model,
                "steps": forecast.steps,
                "granularity": forecast.granularity,
                "created_at": forecast.created_at
            }
        except Exception as e:
            logger.error(f"Error fetching forecast {forecast_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        
    @app.post("/api/forecasts/dhr")
    async def compute_dhr_forecast(request: ForecastRequest, db: Session = Depends(get_db)):
        try:
            # Query the hourly table for the filename
            file_query = text("SELECT filename FROM forecasts WHERE id = :forecast_id")
            result = db.execute(file_query, {"forecast_id": request.forecast_id}).fetchone()
            if not result:
                raise HTTPException(status_code=404, detail="No dataset found for forecast_id")

            filename = result.filename
            data_folder = "hourly" if request.granularity.lower() == "hourly" else None
            if not data_folder:
                raise HTTPException(status_code=400, detail="Invalid granularity")

            file_path = os.path.join(BASE_STORAGE_PATH, data_folder, filename)
            logger.info(f"Looking for file at: {file_path}")
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail=f"File {filename} not found in {data_folder} folder")

            data = pd.read_json(file_path)

            # Check for required columns
            required_columns = ["solar_power", "ghi", "dni", "dhi", "solar_zenith_angle"]
            for col in required_columns:
                if col not in data.columns:
                    raise HTTPException(status_code=400, detail=f"Missing column: {col}")
                if data[col].isnull().all():
                    raise HTTPException(status_code=400, detail=f"All values missing in column: {col}")
                logger.info(f"{col} length: {len(data[col])}, sample: {data[col].head(3).tolist()}")

            # Get configuration
            config_query = text("SELECT * FROM dhr_configurations WHERE forecast_id = :forecast_id")
            config_result = db.execute(config_query, {"forecast_id": request.forecast_id}).fetchone()
            if not config_result:
                raise HTTPException(status_code=404, detail="Configuration not found")

            window = int(float(config_result.window_length))
            max_window = len(data)
            if window > max_window:
                window = max_window if max_window % 2 == 1 else max_window - 1

            try:
                period_list = [24, 168]
                if not period_list:
                    raise ValueError("Seasonality periods list is empty.")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid seasonality periods: {e}")

            params = {
                "fourier_terms": int(float(config_result.fourier_order)),
                "reg_strength": config_result.regularization_dhr,
                "ar_order": int(float(config_result.trend_components)),
                "window": window,
                "polyorder": int(float(config_result.polyorder)),
                "periods": period_list,
            }

            logger.info(f"Using parameters: {params}")

            # Prepare data
            try:
                prepared_data = load_and_prepare_data(data)
                logger.info(f"Prepared data shape: {prepared_data.shape}")
            except Exception as e:
                logger.error(f"Error in load_and_prepare_data: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to prepare data: {str(e)}")

            if prepared_data is None or prepared_data.empty:
                raise HTTPException(status_code=400, detail="Prepared data is empty or invalid.")

            forecast_steps = 168
            data_length = len(prepared_data)
            forecast_steps = min(forecast_steps, data_length - 1)

            try:
                fourier_extended = fourier_transform(
                    t=range(data_length + forecast_steps),
                    n_harmonics=params["fourier_terms"],
                    periods=params["periods"]
                )
                if fourier_extended is None or len(fourier_extended) == 0:
                    raise ValueError("Fourier transform returned empty data.")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Fourier transform failed: {e}")

            forecast = generate_forecast(
                model=None,
                start_values=prepared_data["solar_power"].values[:data_length],
                fourier_data=fourier_extended,
                steps=forecast_steps,
                params=params,
                ghi=prepared_data["ghi"].values[:data_length],
                dni=prepared_data["dni"].values[:data_length],
                dhi=prepared_data["dhi"].values[:data_length],
                sza=prepared_data["solar_zenith_angle"].values[:data_length],
            )

            # Save forecast to database
            save_query = text("INSERT INTO forecasts (forecast_id, forecast_values) VALUES (:id, :values)")
            db.execute(save_query, {"id": request.forecast_id, "values": forecast.tolist()})
            db.commit()

            return {"forecast_id": request.forecast_id, "forecast": forecast.tolist()}

        except Exception as e:
            logger.error(f"Error in compute_dhr_forecast: {str(e)}")
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Failed to compute forecast: {str(e)}")

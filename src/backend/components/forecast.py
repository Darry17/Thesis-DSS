import os
from fastapi import FastAPI, HTTPException, Depends, Request
from sqlalchemy import Column, Integer, String, DateTime, text
from sqlalchemy.orm import Session
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Optional
import logging
import pandas as pd
import numpy as np
from .models.dhr_solar_forecast import generate_forecast, fourier_transform, repeat_last_week, load_and_prepare_data, create_features
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Storage configuration
BASE_STORAGE_PATH = os.getenv('STORAGE_PATH', os.path.join(os.path.dirname(__file__), '..', '..', '..', 'storage'))
os.makedirs(BASE_STORAGE_PATH, exist_ok=True)

# SQLAlchemy Models
Base = declarative_base()

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
            # Query the hourly table for the filename using text()
            file_query = text("SELECT file_name FROM hourly_data WHERE forecast_id = :forecast_id")
            result = db.execute(file_query, {"forecast_id": request.forecast_id}).fetchone()
            if not result:
                raise HTTPException(status_code=404, detail="No dataset found for forecast_id")

            file_name = result.file_name
            data_folder = "hourly" if request.granularity.lower() == "hourly" else None
            if not data_folder:
                raise HTTPException(status_code=400, detail="Invalid granularity")

            # Load dataset from storage folder
            file_path = os.path.join(BASE_STORAGE_PATH, data_folder, file_name)
            logger.info(f"Looking for file at: {file_path}")
            
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail=f"File {file_name} not found in {data_folder} folder")

            data = pd.read_csv(file_path)  # Adjust based on file format

            # Load configuration from database using text()
            config_query = text("SELECT * FROM dhr_configurations WHERE forecast_id = :forecast_id")
            config_result = db.execute(config_query, {"forecast_id": request.forecast_id}).fetchone()
            if not config_result:
                raise HTTPException(status_code=404, detail="Configuration not found")

            logger.info(f"Retrieved configuration: {config_result}")

            params = {
                "fourier_terms": config_result.fourier_order,
                "reg_strength": config_result.regularization_dhr,
                "ar_order": config_result.trend_components,
                "window": config_result.window_length,
                "polyorder": config_result.polyorder,
                "periods": [int(p) for p in config_result.seasonality_periods.split(",")]
            }

            logger.info(f"Using parameters: {params}")

            # Prepare data and compute forecast
            prepared_data = load_and_prepare_data(data)
            fourier_extended = fourier_transform(
                t=range(len(prepared_data) + 168),  # Example forecast horizon
                n_harmonics=params["fourier_terms"],
                periods=params["periods"]
            )
            forecast = generate_forecast(
                model=None,  # Replace with actual model if needed
                target_values=prepared_data.values,
                fourier_extended=fourier_extended,
                forecast_steps=168,  # Example
                params=params,
                ghi_ext=None, dni_ext=None, dhi_ext=None, sza_ext=None  # Adjust as needed
            )

            # Save forecast to database using text()
            save_query = text("INSERT INTO forecasts (forecast_id, forecast_values) VALUES (:id, :values)")
            db.execute(save_query, {"id": request.forecast_id, "values": forecast.tolist()})
            db.commit()

            return {"forecast_id": request.forecast_id, "forecast": forecast.tolist()}
        except Exception as e:
            logger.error(f"Error in compute_dhr_forecast: {str(e)}")
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Failed to compute forecast: {str(e)}")
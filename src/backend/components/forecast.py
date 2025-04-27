from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import Session
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)
# SQLAlchemy Models
Base = declarative_base()
# SQLAlchemy Models
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
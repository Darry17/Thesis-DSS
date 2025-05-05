from fastapi import FastAPI, HTTPException, Depends, Request
from sqlalchemy import Column, Integer, String, DateTime
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
    async def generate_dhr_forecast_route(request: Request, db: Session = Depends(get_db)):
        try:
            body = await request.json()
            data = body.get("data")
            steps = body.get("steps")
            granularity = body.get("granularity")
            model_name = body.get("model")

            if not all([data, steps, granularity, model_name]):
                raise HTTPException(status_code=400, detail="Missing input fields")

            if granularity != "Hourly":
                raise HTTPException(status_code=400, detail="Only Hourly granularity is supported")
            if model_name != "DHR":
                raise HTTPException(status_code=400, detail="Only DHR model is supported")

            step_mapping = {
                "1-hour": 1,
                "24-hour": 24,
                "168-hour": 168
            }
            if steps not in step_mapping:
                raise HTTPException(status_code=400, detail="Invalid steps value")
            forecast_steps = step_mapping[steps]

            # Validate input data
            df = pd.DataFrame(data)
            required_columns = ['time', 'solar_power', 'GHI', 'DNI', 'DHI', 'Solar Zenith Angle']
            if not all(col in df.columns for col in required_columns):
                raise HTTPException(status_code=400, detail="Data must contain 'time', 'solar_power', 'GHI', 'DNI', 'DHI', and 'Solar Zenith Angle' columns")

            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time').sort_index()
            df = df.interpolate()

            # Model parameters
            params = {
                'fourier_terms': 3,
                'reg_strength': 0.0001000100524,
                'ar_order': 3,
                'window': 23,
                'polyorder': 3
            }

            # Create features and train model
            X, y = create_features(df, 'solar_power', params['fourier_terms'], params['ar_order'], params['window'], params['polyorder'])
            if len(X) == 0 or len(y) == 0:
                raise HTTPException(status_code=400, detail="Insufficient data for training")

            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)
            model = Ridge(alpha=params['reg_strength'])
            model.fit(X_train, y_train)

            # Prepare exogenous variables
            ghi = df['GHI'].values
            dni = df['DNI'].values
            dhi = df['DHI'].values
            sza = df['Solar Zenith Angle'].values
            ghi_ext = repeat_last_week(ghi, forecast_steps)
            dni_ext = repeat_last_week(dni, forecast_steps)
            dhi_ext = repeat_last_week(dhi, forecast_steps)
            sza_ext = repeat_last_week(sza, forecast_steps)

            # Create extended Fourier terms
            extended_df_length = len(df) + forecast_steps
            t_extended = np.arange(extended_df_length)
            fourier_extended = fourier_transform(t_extended, n_harmonics=params['fourier_terms'], periods=[24, 168])

            # Generate forecast
            target_values = df['solar_power'].values
            forecast_values = generate_forecast(
                model, target_values, fourier_extended, forecast_steps, params,
                ghi_ext, dni_ext, dhi_ext, sza_ext
            )

            # Prepare response
            last_time = df.index[-1]
            future_index = pd.date_range(start=last_time + timedelta(hours=1), periods=forecast_steps, freq='H')
            forecast_data = [{"time": time.strftime("%Y-%m-%d %H:%M:%S"), "solar_power": float(value)} for time, value in zip(future_index, forecast_values)]

            actual_tail = df['solar_power'].iloc[-336:].reset_index()
            actual_data = [{"time": row['time'].strftime("%Y-%m-%d %H:%M:%S"), "solar_power": float(row['solar_power'])} for _, row in actual_tail.iterrows()]

            return {
                "actual": actual_data,
                "forecast": forecast_data,
                "steps": forecast_steps,
                "granularity": granularity,
                "model": model_name
            }

        except Exception as e:
            logger.error(f"Error in DHR forecast route: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
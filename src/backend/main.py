from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, desc, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
from datetime import datetime
import os
import re
import json
import pandas as pd
import logging
from dotenv import load_dotenv
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI()

# Enable CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection
DATABASE_URL = f"mysql+mysqlconnector://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# SQLAlchemy models
class BaseDataModel(Base):
    __abstract__ = True
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    upload_date = Column(DateTime, default=datetime.now)

class JsonData(BaseDataModel):
    __tablename__ = "json_data"

class HourlyData(BaseDataModel):
    __tablename__ = "hourly_data"

class DailyData(BaseDataModel):
    __tablename__ = "daily_data"

class WeeklyData(BaseDataModel):
    __tablename__ = "weekly_data"

class Forecast(Base):
    __tablename__ = "forecasts"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    forecast_model = Column(String(50), nullable=False)
    steps = Column(String(50), nullable=False)
    granularity = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.now)

class DHRConfiguration(Base):
    __tablename__ = "dhr_configurations"
    
    id = Column(Integer, primary_key=True, index=True)
    forecast_id = Column(Integer, nullable=False)
    fourier_order = Column(Integer, nullable=False)
    window_length = Column(Integer, nullable=False)
    seasonality_periods = Column(String(50), nullable=False)
    polyorder = Column(Float, nullable=False)
    regularization_dhr = Column(Float, nullable=False)
    trend_components = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

class ESNConfiguration(Base):
    __tablename__ = "esn_configurations"
    
    id = Column(Integer, primary_key=True, index=True)
    forecast_id = Column(Integer, nullable=False)
    reservoir_size = Column(Integer, nullable=False)
    spectral_radius = Column(Float, nullable=False)
    sparsity = Column(Float, nullable=False)
    input_scaling = Column(Float, nullable=False)
    dropout = Column(Float, nullable=False)
    lags = Column(Integer, nullable=False)
    regularization_esn = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

class HybridConfiguration(Base):
    __tablename__ = "hybrid_configurations"
    
    id = Column(Integer, primary_key=True, index=True)
    forecast_id = Column(Integer, nullable=False)
    fourier_order = Column(Integer, nullable=False)
    window_length = Column(Integer, nullable=False)
    seasonality_periods = Column(String(50), nullable=False)
    polyorder = Column(Float, nullable=False)
    regularization_dhr = Column(Float, nullable=False)
    trend_components = Column(Integer, nullable=False)
    reservoir_size = Column(Integer, nullable=False)
    spectral_radius = Column(Float, nullable=False)
    sparsity = Column(Float, nullable=False)
    input_scaling = Column(Float, nullable=False)
    dropout = Column(Float, nullable=False)
    lags = Column(Integer, nullable=False)
    regularization_esn = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

Base.metadata.create_all(bind=engine)

# Database session dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Response models
class FileResponse(BaseModel):
    status: str
    file_path: str

class LatestFileResponse(BaseModel):
    id: int
    filename: str

# Storage configuration
BASE_STORAGE_PATH = os.getenv('STORAGE_PATH', "../../storage")
os.makedirs(BASE_STORAGE_PATH, exist_ok=True)

FOLDERS = ["hourly", "daily", "weekly", "others", "json"]
DATA_MODELS = {
    "hourly": HourlyData,
    "daily": DailyData,
    "weekly": WeeklyData,
    "json": JsonData
}

def get_file_path(filename: str) -> str:
    for folder in FOLDERS:
        file_path = os.path.join(BASE_STORAGE_PATH, folder, filename)
        if os.path.exists(file_path):
            return file_path
    return None

@app.get("/storage/read/{filename}")
async def read_json_file(filename: str):
    try:
        logger.info(f"Reading file: {filename}")
        folders = ['hourly', 'daily', 'weekly', 'json', 'others']
        file_path = None
        for folder in folders:
            temp_path = os.path.join(BASE_STORAGE_PATH, folder, filename)
            if os.path.exists(temp_path):
                file_path = temp_path
                break
        if not file_path:
            logger.error(f"File not found: {filename}")
            raise HTTPException(status_code=404, detail="File not found")

        with open(file_path, 'r') as f:
            content = f.read()
            logger.info(f"Raw content: {content[:100]}...")
            data = json.loads(content)
        logger.info(f"Parsed data: {data[:2] if isinstance(data, list) else data}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Invalid JSON: {str(e)}")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

@app.post("/storage/process_model_data/")
async def process_model_data(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        filename = file.filename
        logger.info(f"Processing file: {filename}")
        
        # Determine subfolder and data model
        if "hourly" in filename:
            subfolder = "hourly"
            data_model = HourlyData
            logger.info("Matched subfolder: hourly")
        elif "daily" in filename:
            subfolder = "daily"
            data_model = DailyData
            logger.info("Matched subfolder: daily")
        elif "weekly" in filename:
            subfolder = "weekly"
            data_model = WeeklyData
            logger.info("Matched subfolder: weekly")
        else:
            logger.error(f"Invalid filename format: {filename}")
            raise HTTPException(status_code=400, detail="Invalid filename format")
        
        # Save file to storage
        target_folder = os.path.join(BASE_STORAGE_PATH, subfolder)
        os.makedirs(target_folder, exist_ok=True)
        file_path = os.path.join(target_folder, filename)
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        # Save to database
        db_entry = data_model(filename=filename)
        db.add(db_entry)
        db.commit()
        db.refresh(db_entry)

        logger.info(f"File saved to: {file_path} and database table: {data_model.__tablename__}")
        return {
            "status": "File processed", 
            "file_path": file_path,
            "filename": filename,
            "table": data_model.__tablename__,
            "upload_date": db_entry.upload_date
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/storage/upload/", response_model=FileResponse)
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        filename = file.filename
        print(f"Received file: {filename}")

        # Determine data type from filename
        pattern = r"^(hourly|daily|weekly)_(solar|wind)_data_\d{4}_\d{2}_\d{2}T\d{2}_\d{2}_\d{2}_\d{3}Z\.json$"
        match = re.match(pattern, filename)
        
        subfolder = "others"
        data_model = None
        
        if match:
            frequency = match.group(1)
            subfolder = frequency
            data_model = DATA_MODELS[frequency]

        # Save file
        target_folder = os.path.join(BASE_STORAGE_PATH, subfolder)
        os.makedirs(target_folder, exist_ok=True)
        file_location = os.path.join(target_folder, filename)
        
        content = await file.read()
        with open(file_location, "wb") as f:
            f.write(content)

        # Create DB entry if valid model type
        if data_model:
            db_entry = data_model(filename=filename)
            db.add(db_entry)
            db.commit()

        return {"status": "File uploaded successfully", "file_path": file_location}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add this new model
class UploadResponse(BaseModel):
    status: str
    file_path: str
    filename: str

# Modify the upload_csv endpoint
@app.post("/storage/upload_csv/", response_model=UploadResponse)
async def upload_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        filename = file.filename
        file_location = os.path.join(BASE_STORAGE_PATH, "json", filename)
        os.makedirs(os.path.dirname(file_location), exist_ok=True)
        
        content = await file.read()
        with open(file_location, "wb") as f:
            f.write(content)

        db_entry = JsonData(filename=filename)
        db.add(db_entry)
        db.commit()

        return {
            "status": "File uploaded successfully", 
            "file_path": file_location,
            "filename": filename
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/storage/latest-file/", response_model=LatestFileResponse)
async def get_latest_file(data_type: str, db: Session = Depends(get_db)):
    try:
        data_model = DATA_MODELS.get(data_type)
        if not data_model:
            raise HTTPException(status_code=400, detail="Invalid data_type. Valid options are 'hourly', 'daily', 'weekly'.")

        latest_file = db.query(data_model).order_by(desc(data_model.id)).first()
        if not latest_file:
            raise HTTPException(status_code=404, detail=f"No files found in the {data_model.__tablename__} table.")

        return {"id": latest_file.id, "filename": latest_file.filename}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/storage/get-latest-by-pattern/{filename}", response_model=LatestFileResponse)
async def get_latest_by_pattern(filename: str, db: Session = Depends(get_db)):
    try:
        # Determine data type from filename
        data_type = None
        if "weekly" in filename:
            data_type = "weekly"
        elif "daily" in filename:
            data_type = "daily"
        elif "hourly" in filename:
            data_type = "hourly"
        
        if not data_type:
            raise HTTPException(status_code=400, detail="Invalid filename pattern")
            
        data_model = DATA_MODELS.get(data_type)
        latest_file = db.query(data_model).order_by(desc(data_model.id)).first()
        
        if not latest_file:
            raise HTTPException(status_code=404, detail=f"No files found in the {data_model.__tablename__} table.")

        return {
            "id": latest_file.id, 
            "filename": latest_file.filename,
            "upload_date": latest_file.upload_date
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add this to your Pydantic models section
class ForecastCreate(BaseModel):
    filename: str
    forecast_model: str
    steps: str
    granularity: str

    model_config = {
        'from_attributes': True
    }

# Add this new endpoint
@app.post("/api/forecasts")
async def create_forecast(forecast: ForecastCreate, db: Session = Depends(get_db)):
    try:
        logger.info(f"Creating forecast: {forecast}")
        
        db_forecast = Forecast(
            filename=forecast.filename,
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
            "forecast_model": db_forecast.forecast_model,
            "steps": db_forecast.steps,
            "granularity": db_forecast.granularity,
            "created_at": db_forecast.created_at
        }

    except Exception as e:
        logger.error(f"Error creating forecast: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# Add these Pydantic models (after your existing ForecastCreate model)
class DHRConfigurationCreate(BaseModel):
    forecast_id: int
    fourier_order: int
    window_length: int
    seasonality_periods: str
    polyorder: float
    regularization_dhr: float
    trend_components: int

    model_config = {
        'from_attributes': True
    }

class ESNConfigurationCreate(BaseModel):
    forecast_id: int
    reservoir_size: int
    spectral_radius: float
    sparsity: float
    input_scaling: float
    dropout: float
    lags: int
    regularization_esn: float

    model_config = {
        'from_attributes': True
    }

# Add these new endpoints (after your existing endpoints)
@app.post("/api/dhr-configurations")
async def create_dhr_configuration(config: DHRConfigurationCreate, db: Session = Depends(get_db)):
    try:
        logger.info(f"Creating DHR configuration: {config}")
        
        # Check if forecast exists
        forecast = db.query(Forecast).filter(Forecast.id == config.forecast_id).first()
        if not forecast:
            raise HTTPException(status_code=404, detail="Forecast not found")

        db_config = DHRConfiguration(
            forecast_id=config.forecast_id,
            fourier_order=config.fourier_order,
            window_length=config.window_length,
            seasonality_periods=config.seasonality_periods,
            polyorder=config.polyorder,
            regularization_dhr=config.regularization_dhr,
            trend_components=config.trend_components
        )
        
        db.add(db_config)
        db.commit()
        db.refresh(db_config)

        logger.info(f"Created DHR configuration with ID: {db_config.id}")

        return {
            "id": db_config.id,
            "message": "DHR configuration created successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating DHR configuration: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/esn-configurations")
async def create_esn_configuration(config: ESNConfigurationCreate, db: Session = Depends(get_db)):
    try:
        logger.info(f"Creating ESN configuration: {config}")
        
        # Check if forecast exists
        forecast = db.query(Forecast).filter(Forecast.id == config.forecast_id).first()
        if not forecast:
            raise HTTPException(status_code=404, detail="Forecast not found")

        db_config = ESNConfiguration(
            forecast_id=config.forecast_id,
            reservoir_size=config.reservoir_size,
            spectral_radius=config.spectral_radius,
            sparsity=config.sparsity,
            input_scaling=config.input_scaling,
            dropout=config.dropout,
            lags=config.lags,
            regularization_esn=config.regularization_esn
        )
        
        db.add(db_config)
        db.commit()
        db.refresh(db_config)

        logger.info(f"Created ESN configuration with ID: {db_config.id}")

        return {
            "id": db_config.id,
            "message": "ESN configuration created successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating ESN configuration: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# Optional: Add these GET endpoints
@app.get("/api/dhr-configurations/{forecast_id}")
async def get_dhr_configuration(forecast_id: int, db: Session = Depends(get_db)):
    try:
        config = db.query(DHRConfiguration).filter(DHRConfiguration.forecast_id == forecast_id).first()
        if not config:
            raise HTTPException(status_code=404, detail="DHR configuration not found")
        return config
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/esn-configurations/{forecast_id}")
async def get_esn_configuration(forecast_id: int, db: Session = Depends(get_db)):
    try:
        config = db.query(ESNConfiguration).filter(ESNConfiguration.forecast_id == forecast_id).first()
        if not config:
            raise HTTPException(status_code=404, detail="ESN configuration not found")
        return config
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Pydantic model for request validation
class HybridConfigurationCreate(BaseModel):
    forecast_id: int
    fourier_order: int
    window_length: int
    seasonality_periods: str
    polyorder: float
    regularization_dhr: float
    trend_components: int
    reservoir_size: int
    spectral_radius: float
    sparsity: float
    input_scaling: float
    dropout: float
    lags: int
    regularization_esn: float

    model_config = {
        'from_attributes': True
    }

@app.post("/api/hybrid-configurations")
async def create_hybrid_configuration(config: HybridConfigurationCreate, db: Session = Depends(get_db)):
    print("Received configuration:", config)  # Debug log
    try:
        db_config = HybridConfiguration(
            forecast_id=config.forecast_id,
            fourier_order=config.fourier_order,
            window_length=config.window_length,
            seasonality_periods=config.seasonality_periods,
            polyorder=config.polyorder,
            regularization_dhr=config.regularization_dhr,
            trend_components=config.trend_components,
            reservoir_size=config.reservoir_size,
            spectral_radius=config.spectral_radius,
            sparsity=config.sparsity,
            input_scaling=config.input_scaling,
            dropout=config.dropout,
            lags=config.lags,
            regularization_esn=config.regularization_esn,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

        db.add(db_config)
        db.commit()
        db.refresh(db_config)

        return {
            "id": db_config.id,
            "message": "Hybrid configuration created successfully"
        }

    except Exception as e:
        db.rollback()
        print(f"Error creating hybrid configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add GET endpoint for retrieving the configuration
@app.get("/api/hybrid-configurations/{forecast_id}")
async def get_hybrid_configuration(forecast_id: int, db: Session = Depends(get_db)):
    try:
        config = db.query(HybridConfiguration).filter(
            HybridConfiguration.forecast_id == forecast_id
        ).first()
        
        if not config:
            raise HTTPException(
                status_code=404,
                detail=f"Hybrid configuration not found for forecast_id: {forecast_id}"
            )
            
        return {
            "id": config.id,
            "forecast_id": config.forecast_id,
            "fourier_order": config.fourier_order,
            "window_length": config.window_length,
            "seasonality_periods": config.seasonality_periods,
            "polyorder": config.polyorder,
            "regularization_dhr": config.regularization_dhr,
            "trend_components": config.trend_components,
            "reservoir_size": config.reservoir_size,
            "spectral_radius": config.spectral_radius,
            "sparsity": config.sparsity,
            "input_scaling": config.input_scaling,
            "dropout": config.dropout,
            "lags": config.lags,
            "regularization_esn": config.regularization_esn,
            "created_at": config.created_at,
            "updated_at": config.updated_at
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/forecasts/{forecast_id}")
async def get_forecast(forecast_id: int, db: Session = Depends(get_db)):
    try:
        # Query forecast using regular SQLAlchemy session
        forecast = db.query(Forecast).filter(Forecast.id == forecast_id).first()
        
        if not forecast:
            raise HTTPException(status_code=404, detail=f"Forecast {forecast_id} not found")
        
        # Convert SQLAlchemy model to dictionary
        return {
            "id": forecast.id,
            "filename": forecast.filename,
            "model": forecast.forecast_model,
            "steps": forecast.steps,
            "granularity": forecast.granularity,
            "created_at": forecast.created_at
        }
    except Exception as e:
        logger.error(f"Error fetching forecast {forecast_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Test endpoint to verify API is working
@app.get("/")
async def root():
    return {"message": "API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0000", port=8000)
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from sqlalchemy import Column, Integer, String, DateTime, desc
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
import os
import re
import json
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Base for SQLAlchemy models
Base = declarative_base()

# Storage configuration
BASE_STORAGE_PATH = os.getenv('STORAGE_PATH', "../../storage")
os.makedirs(BASE_STORAGE_PATH, exist_ok=True)

FOLDERS = ["hourly", "daily", "weekly", "others", "json"]

# SQLAlchemy Models
class BaseDataModel(Base):
    __abstract__ = True
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    upload_date = Column(DateTime, default=datetime.now)
    original_filename = Column(String(255), nullable=True)

class JsonData(BaseDataModel):
    __tablename__ = "json_data"

class HourlyData(BaseDataModel):
    __tablename__ = "hourly_data"

class DailyData(BaseDataModel):
    __tablename__ = "daily_data"

class WeeklyData(BaseDataModel):
    __tablename__ = "weekly_data"

# Pydantic Models
class JsonDataResponse(BaseModel):
    id: int
    filename: str
    original_filename: Optional[str] = None
    upload_date: datetime
    model_config = {'from_attributes': True}

class FileResponse(BaseModel):
    status: str
    file_path: str
    original_filename: Optional[str] = None

class UploadResponse(BaseModel):
    status: str
    file_path: str
    filename: str
    original_filename: Optional[str] = None

class LatestFileResponse(BaseModel):
    id: int
    filename: str
    original_filename: Optional[str] = None
    upload_date: Optional[datetime] = None

# Data model mapping
DATA_MODELS = {
    "hourly": HourlyData,
    "daily": DailyData,
    "weekly": WeeklyData,
    "json": JsonData
}

# Helper functions
def get_file_path(filename: str) -> str:
    for folder in FOLDERS:
        file_path = os.path.join(BASE_STORAGE_PATH, folder, filename)
        if os.path.exists(file_path):
            return file_path
    return None

# Routes
def register_storage_routes(app: FastAPI, get_db):
    @app.get("/storage/read/{filename}")
    async def read_json_file(filename: str):
        try:
            logger.info(f"Reading file: {filename}")
            file_path = get_file_path(filename)
            if not file_path:
                logger.error(f"File not found: {filename}")
                raise HTTPException(status_code=404, detail="File not found")
            with open(file_path, 'r') as f:
                content = f.read()
                data = json.loads(content)
            return data
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Invalid JSON: {str(e)}")
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

    @app.post("/storage/process_model_data/")
    async def process_model_data(
        file: UploadFile = File(...), 
        original_filename: str = Form(None),
        db: Session = Depends(get_db)
    ):
        try:
            filename = file.filename
            logger.info(f"Processing file: {filename}, Original file: {original_filename}")
            
            if "hourly" in filename:
                subfolder = "hourly"
                data_model = HourlyData
            elif "daily" in filename:
                subfolder = "daily"
                data_model = DailyData
            elif "weekly" in filename:
                subfolder = "weekly"
                data_model = WeeklyData
            else:
                logger.error(f"Invalid filename format: {filename}")
                raise HTTPException(status_code=400, detail="Invalid filename format")
            
            target_folder = os.path.join(BASE_STORAGE_PATH, subfolder)
            os.makedirs(target_folder, exist_ok=True)
            file_path = os.path.join(target_folder, filename)
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)

            db_entry = data_model(
                filename=filename,
                original_filename=original_filename or "Unknown"
            )
            db.add(db_entry)
            db.commit()
            db.refresh(db_entry)
            
            return {
                "status": "File processed", 
                "file_path": file_path,
                "filename": filename,
                "original_filename": original_filename,
                "table": data_model.__tablename__,
                "upload_date": db_entry.upload_date
            }
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/storage/upload/", response_model=FileResponse)
    async def upload_file(
        file: UploadFile = File(...), 
        original_filename: str = Form(None),
        db: Session = Depends(get_db)
    ):
        try:
            filename = file.filename
            logger.info(f"Received file: {filename}, Original file: {original_filename}")

            pattern = r"^(hourly|daily|weekly)_(solar|wind)_data_\d{4}_\d{2}_\d{2}T\d{2}_\d{2}_\d{2}_\d{3}Z\.json$"
            match = re.match(pattern, filename)
            
            subfolder = "others"
            data_model = None
            
            if match:
                frequency = match.group(1)
                subfolder = frequency
                data_model = DATA_MODELS[frequency]

            target_folder = os.path.join(BASE_STORAGE_PATH, subfolder)
            os.makedirs(target_folder, exist_ok=True)
            file_location = os.path.join(target_folder, filename)
            
            content = await file.read()
            with open(file_location, "wb") as f:
                f.write(content)

            if data_model:
                db_entry = data_model(
                    filename=filename,
                    original_filename=original_filename or "Unknown"
                )
                db.add(db_entry)
                db.commit()

            return {
                "status": "File uploaded successfully", 
                "file_path": file_location,
                "original_filename": original_filename
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/storage/upload_csv/", response_model=UploadResponse)
    async def upload_csv(
        file: UploadFile = File(...), 
        original_filename: str = Form(...),
        db: Session = Depends(get_db)
    ):
        try:
            filename = file.filename
            file_location = os.path.join(BASE_STORAGE_PATH, "json", filename)
            os.makedirs(os.path.dirname(file_location), exist_ok=True)
            
            content = await file.read()
            with open(file_location, "wb") as f:
                f.write(content)

            db_entry = JsonData(
                filename=filename, 
                original_filename=original_filename
            )
            db.add(db_entry)
            db.commit()
            db.refresh(db_entry)

            logger.info(f"Uploaded JSON file: {filename}, Original CSV: {original_filename}")

            return {
                "status": "File uploaded successfully", 
                "file_path": file_location,
                "filename": filename,
                "original_filename": original_filename
            }

        except Exception as e:
            logger.error(f"Upload error: {str(e)}")
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

            return {
                "id": latest_file.id, 
                "filename": latest_file.filename, 
                "original_filename": latest_file.original_filename,
                "upload_date": latest_file.upload_date
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/storage/get-latest-by-pattern/{filename}", response_model=LatestFileResponse)
    async def get_latest_by_pattern(filename: str, db: Session = Depends(get_db)):
        try:
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
                "original_filename": latest_file.original_filename,
                "upload_date": latest_file.upload_date
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/storage/json-data/{json_id}", response_model=JsonDataResponse)
    async def get_json_data(json_id: int, db: Session = Depends(get_db)):
        try:
            json_data = db.query(JsonData).filter(JsonData.id == json_id).first()
            if not json_data:
                raise HTTPException(status_code=404, detail=f"JSON data with ID {json_id} not found")
            
            logger.info(f"Retrieved JSON data: {json_data.filename}, Original file: {json_data.original_filename}")
            return json_data
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error fetching JSON data: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import uuid
import os
import logging
from model import Temp
from db import SessionLocal

router = APIRouter()
logger = logging.getLogger(__name__)

router = APIRouter()

FORECAST_FOLDER = "forecasts"

@router.post("/temp-upload")
async def upload_temp_file(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    temp_name = f"temp_{file_id}.csv"
    temp_path = os.path.join("temp", temp_name)
    db = SessionLocal()

    try:
        contents = await file.read()
        with open(temp_path, "wb") as f:
            f.write(contents)

        # Insert into the database
        temp_entry = Temp(filename=temp_name)
        db.add(temp_entry)
        db.commit()
        db.refresh(temp_entry)

        return {
            "temp_filename": temp_name,
            "temp_id": temp_entry.id
        }

    except Exception as e:
        db.rollback()
        logger.error(f"Temp upload failed: {e}")
        raise HTTPException(status_code=500, detail="Upload failed")

    finally:
        db.close()

@router.get("/api/forecast-file/{filename}")
async def get_forecast_file(filename: str):
    filepath = os.path.join(FORECAST_FOLDER, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(filepath, media_type='text/csv')




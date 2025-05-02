from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy import Column, Integer, String, DateTime, desc
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
import logging
from sqlalchemy import or_

# Configure logging
logger = logging.getLogger(__name__)

# Define the Base class
Base = declarative_base()

# SQLAlchemy Models
class HistoryLog(Base):
    __tablename__ = "history_logs"
    id = Column(Integer, primary_key=True, index=True)
    forecast_id = Column(Integer, nullable=True)
    file_name = Column(String(255), nullable=False)
    model = Column(String(50), nullable=False)
    date = Column(DateTime, default=datetime.now)

class DeletedForecast(Base):
    __tablename__ = "deleted_forecasts"
    id = Column(Integer, primary_key=True, index=True)
    forecast_id = Column(Integer, nullable=True)
    file_name = Column(String(255), nullable=False)
    model = Column(String(50), nullable=False)
    date = Column(DateTime, default=datetime.now)
    deleted_by = Column(String(255), nullable=False)

# Pydantic Models
class HistoryLogCreate(BaseModel):
    file_name: str
    model: str
    action: Optional[str] = "Saved Forecast"
    forecast_id: Optional[int] = None
    model_config = {'from_attributes': True}

class HistoryLogResponse(BaseModel):
    id: int
    forecast_id: Optional[int] = None
    file_name: str
    model: str
    date: datetime
    model_config = {'from_attributes': True}

class PaginatedHistoryLogResponse(BaseModel):
    logs: List[HistoryLogResponse]
    total_pages: int
    model_config = {'from_attributes': True}

class DeletedForecastResponse(BaseModel):
    id: int
    forecast_id: Optional[int]
    file_name: str
    model: str
    date: datetime
    deleted_by: str
    model_config = {'from_attributes': True}

class PaginatedDeletedForecastResponse(BaseModel):
    logs: List[DeletedForecastResponse]
    total_pages: int
    model_config = {'from_attributes': True}

# Routes
def register_history_log_routes(app: FastAPI, get_db, get_current_user, get_admin_user, Forecast):
    @app.post("/api/history-logs")
    async def create_history_log(
        log: HistoryLogCreate, 
        db: Session = Depends(get_db)
    ):
        try:
            logger.info(f"Creating history log with data: {log.dict()}")
            
            forecast_id = None
            if log.forecast_id:
                try:
                    forecast_id = int(log.forecast_id)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid forecast_id format: {log.forecast_id}, setting to None")
                    forecast_id = None
            
            db_log = HistoryLog(
                file_name=log.file_name,
                model=log.model,
                forecast_id=forecast_id
            )
            
            db.add(db_log)
            
            try:
                db.commit()
                db.refresh(db_log)
                logger.info(f"Successfully created history log with ID: {db_log.id}")
            except Exception as commit_error:
                db.rollback()
                logger.error(f"Database commit error: {str(commit_error)}")
                raise commit_error

            return {
                "id": db_log.id,
                "forecast_id": db_log.forecast_id,
                "file_name": db_log.file_name,
                "model": db_log.model,
                "date": db_log.date
            }

        except Exception as e:
            logger.error(f"Error creating history log: {str(e)}")
            db.rollback()
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to create history log: {str(e)}"
            )

    @app.get("/api/history-logs/{log_id}", response_model=HistoryLogResponse)
    async def get_history_log(log_id: int, db: Session = Depends(get_db)):
        try:
            log = db.query(HistoryLog).filter(HistoryLog.id == log_id).first()
            
            if not log:
                raise HTTPException(status_code=404, detail=f"History log with ID {log_id} not found")
            
            return {
                "id": log.id,
                "forecast_id": log.forecast_id,
                "file_name": log.file_name,
                "model": log.model,
                "date": log.date
            }
        except Exception as e:
            logger.error(f"Error fetching history log {log_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/history-logs", response_model=PaginatedHistoryLogResponse)
    async def get_all_history_logs(
        db: Session = Depends(get_db),
        page: int = 1,
        limit: int = 10,
        search: Optional[str] = ""
    ):
        try:
            # Build the query
            query = db.query(HistoryLog)

            # Apply search filter if provided
            if search:
                search_term = f"%{search}%"
                query = query.filter(
                    or_(
                        HistoryLog.file_name.ilike(search_term),
                        HistoryLog.model.ilike(search_term)
                    )
                )

            # Calculate pagination
            total_logs = query.count()
            total_pages = (total_logs + limit - 1) // limit

            # Apply pagination
            logs = (
                query.order_by(desc(HistoryLog.date))
                .offset((page - 1) * limit)
                .limit(limit)
                .all()
            )

            # Serialize logs to HistoryLogResponse
            serialized_logs = [HistoryLogResponse.model_validate(log) for log in logs]

            return {
                "logs": serialized_logs,
                "total_pages": total_pages
            }
        except Exception as e:
            logger.error(f"Error fetching history logs: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/forecasts/{forecast_id}")
    async def delete_forecast(
        forecast_id: int, 
        db: Session = Depends(get_db), 
        current_user: dict = Depends(get_current_user)
    ):
        try:
            # Fetch the forecast to ensure it exists
            forecast = db.query(Forecast).filter(Forecast.id == forecast_id).first()
            if not forecast:
                raise HTTPException(status_code=404, detail="Forecast not found")

            # Fetch the associated history log to get the username
            history_log = db.query(HistoryLog).filter(HistoryLog.forecast_id == forecast_id).first()
            if not history_log:
                raise HTTPException(status_code=404, detail="History log not found for this forecast")

            # Log the deletion in deleted_forecasts
            deleted_forecast = DeletedForecast(
                forecast_id=forecast_id,
                file_name=history_log.file_name,
                model=history_log.model,
                deleted_by=current_user["username"]
            )
            db.add(deleted_forecast)

            # Delete the history log entries
            db.query(HistoryLog).filter(HistoryLog.forecast_id == forecast_id).delete()

            # Commit the transaction
            db.commit()

            return {"message": "Forecast deleted successfully"}

        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Error deleting forecast {forecast_id}: {str(e)}")
            db.rollback()
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/recover-forecast/{forecast_id}")
    async def recover_forecast(
        forecast_id: int,
        db: Session = Depends(get_db),
        current_user: dict = Depends(get_admin_user)
    ):
        try:
            # Fetch the deleted forecast record
            deleted_forecast = db.query(DeletedForecast).filter(DeletedForecast.forecast_id == forecast_id).first()
            if not deleted_forecast:
                raise HTTPException(status_code=404, detail="Deleted forecast not found")

            # Create a new history log entry
            recovered_log = HistoryLog(
                forecast_id=deleted_forecast.forecast_id,
                file_name=deleted_forecast.file_name,
                model=deleted_forecast.model,
                date=datetime.now()
            )
            db.add(recovered_log)

            # Delete the record from deleted_forecasts
            db.query(DeletedForecast).filter(DeletedForecast.forecast_id == forecast_id).delete()

            # Commit the transaction
            db.commit()

            return {"message": f"Forecast {forecast_id} recovered successfully"}

        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Error recovering forecast {forecast_id}: {str(e)}")
            db.rollback()
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/deleted-forecasts", response_model=PaginatedDeletedForecastResponse)
    async def get_deleted_forecasts(
        db: Session = Depends(get_db), 
        current_user: dict = Depends(get_current_user),
        page: int = 1,
        limit: int = 10,
        search: Optional[str] = ""
    ):
        try:
            # Build the query
            query = db.query(DeletedForecast)

            # Apply search filter if provided
            if search:
                search_term = f"%{search}%"
                query = query.filter(
                    or_(
                        DeletedForecast.file_name.ilike(search_term),
                        DeletedForecast.model.ilike(search_term)
                    )
                )

            # Calculate pagination
            total_logs = query.count()
            total_pages = (total_logs + limit - 1) // limit

            # Apply pagination
            logs = (
                query.order_by(desc(DeletedForecast.date))
                .offset((page - 1) * limit)
                .limit(limit)
                .all()
            )

            # Serialize logs to DeletedForecastResponse
            serialized_logs = [DeletedForecastResponse.model_validate(log) for log in logs]

            return {
                "logs": serialized_logs,
                "total_pages": total_pages
            }
        except Exception as e:
            logger.error(f"Error fetching deleted forecasts: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/deleted-forecasts/id/{id}")
    async def delete_deleted_forecast(
        id: int,
        db: Session = Depends(get_db),
        current_user: dict = Depends(get_admin_user)
    ):
        try:
            # Fetch the deleted forecast record by id
            deleted_forecast = db.query(DeletedForecast).filter(DeletedForecast.id == id).first()
            if not deleted_forecast:
                raise HTTPException(status_code=404, detail="Deleted forecast not found")

            # Delete the record from deleted_forecasts
            db.query(DeletedForecast).filter(DeletedForecast.id == id).delete()

            # Commit the transaction
            db.commit()

            return {"message": f"Deleted forecast with ID {id} permanently removed"}

        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Error deleting deleted forecast with ID {id}: {str(e)}")
            db.rollback()
            raise HTTPException(status_code=500, detail=str(e))
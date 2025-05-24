from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy import desc, or_
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional, List
import logging
from model import HistoryLog, DeletedForecast, Forecast, HistoryLogCreate, HistoryLogResponse, PaginatedHistoryLogResponse, DeletedForecastResponse, PaginatedDeletedForecastResponse
from db import get_db
from auth import get_current_user, get_admin_user

# Configure logging
logger = logging.getLogger(__name__)

def register_history_log_routes(app: FastAPI):
    @app.post("/api/history-logs", response_model=HistoryLogResponse)
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
                    forecast = db.query(Forecast).filter(Forecast.id == forecast_id).first()
                    if not forecast:
                        raise HTTPException(status_code=404, detail=f"Forecast with ID {forecast_id} not found")
                except (ValueError, TypeError):
                    logger.warning(f"Invalid forecast_id format: {log.forecast_id}")
                    raise HTTPException(status_code=400, detail="Invalid forecast_id format")
            
            db_log = HistoryLog(
                file_name=log.file_name,
                forecast_type=log.forecast_type,
                granularity=log.granularity,
                steps=log.steps,
                model=log.model,
                forecast_id=forecast_id
            )
            db.add(db_log)
            db.commit()
            db.refresh(db_log)
            logger.info(f"Successfully created history log with ID: {db_log.id}")
            return HistoryLogResponse.model_validate(db_log)
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Error creating history log: {str(e)}")
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Failed to create history log: {str(e)}")

    @app.get("/api/history-logs/{log_id}", response_model=HistoryLogResponse)
    async def get_history_log(
        log_id: int,
        db: Session = Depends(get_db)
    ):
        try:
            log = db.query(HistoryLog).filter(HistoryLog.id == log_id).first()
            if not log:
                raise HTTPException(status_code=404, detail=f"History log with ID {log_id} not found")
            return HistoryLogResponse.model_validate(log)
        except HTTPException as e:
            raise e
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
            query = db.query(HistoryLog)
            if search:
                search_term = f"%{search}%"
                query = query.filter(
                    or_(
                        HistoryLog.file_name.ilike(search_term),
                        HistoryLog.model.ilike(search_term)
                    )
                )
            total_logs = query.count()
            total_pages = (total_logs + limit - 1) // limit
            logs = query.order_by(desc(HistoryLog.date)).offset((page - 1) * limit).limit(limit).all()
            return PaginatedHistoryLogResponse(
                logs=[HistoryLogResponse.model_validate(log) for log in logs],
                total_pages=total_pages
            )
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
            forecast = db.query(Forecast).filter(Forecast.id == forecast_id).first()
            if not forecast:
                raise HTTPException(status_code=404, detail="Forecast not found")
            history_log = db.query(HistoryLog).filter(HistoryLog.forecast_id == forecast_id).first()
            if not history_log:
                raise HTTPException(status_code=404, detail="History log not found for this forecast")
            deleted_forecast = DeletedForecast(
                forecast_id=forecast_id,
                file_name=history_log.file_name,
                forecast_type=history_log.forecast_type,
                granularity=history_log.granularity,
                steps=history_log.steps,
                model=history_log.model,
                deleted_by=current_user["username"],
                date=datetime.now()
            )
            db.add(deleted_forecast)
            db.query(HistoryLog).filter(HistoryLog.forecast_id == forecast_id).delete()
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
            deleted_forecast = db.query(DeletedForecast).filter(DeletedForecast.forecast_id == forecast_id).first()
            if not deleted_forecast:
                raise HTTPException(status_code=404, detail="Deleted forecast not found")
            recovered_log = HistoryLog(
                forecast_id=deleted_forecast.forecast_id,
                file_name=deleted_forecast.file_name,
                forecast_type=deleted_forecast.forecast_type,
                granularity=deleted_forecast.granularity,
                steps=deleted_forecast.steps,
                model=deleted_forecast.model,
                date=datetime.now()
            )
            db.add(recovered_log)
            db.query(DeletedForecast).filter(DeletedForecast.forecast_id == forecast_id).delete()
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
            query = db.query(DeletedForecast)
            if search:
                search_term = f"%{search}%"
                query = query.filter(
                    or_(
                        DeletedForecast.file_name.ilike(search_term),
                        DeletedForecast.model.ilike(search_term)
                    )
                )
            total_logs = query.count()
            total_pages = (total_logs + limit - 1) // limit
            logs = query.order_by(desc(DeletedForecast.date)).offset((page - 1) * limit).limit(limit).all()
            return PaginatedDeletedForecastResponse(
                logs=[DeletedForecastResponse.model_validate(log) for log in logs],
                total_pages=total_pages
            )
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
            deleted_forecast = db.query(DeletedForecast).filter(DeletedForecast.id == id).first()
            if not deleted_forecast:
                raise HTTPException(status_code=404, detail="Deleted forecast not found")
            db.query(DeletedForecast).filter(DeletedForecast.id == id).delete()
            db.commit()
            return {"message": f"Deleted forecast with ID {id} permanently removed"}
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Error deleting deleted forecast with ID {id}: {str(e)}")
            db.rollback()
            raise HTTPException(status_code=500, detail=str(e))
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os
import uuid
import logging
from db import get_db
from datetime import datetime
from models.solar_forecast_dhr_hourly import run_forecast as run_dhr_forecast_solar_hourly
from models.solar_forecast_dhr_daily import run_forecast as run_dhr_forecast_solar_daily
from models.solar_forecast_esn_hourly import run_forecast as run_esn_forecast_solar_hourly
from models.solar_forecast_esn_daily import run_forecast as run_esn_forecast_solar_daily
from models.solar_forecast_hybrid_hourly import run_forecast as run_hybrid_forecast_solar_hourly
from models.solar_forecast_hybrid_daily import run_forecast as run_hybrid_forecast_solar_daily
from models.wind_forecast_dhr_hourly import run_forecast as run_dhr_forecast_wind_hourly
from models.wind_forecast_dhr_daily import run_forecast as run_dhr_forecast_wind_daily
from models.wind_forecast_esn_hourly import run_forecast as run_esn_forecast_wind_hourly
from models.wind_forecast_esn_daily import run_forecast as run_esn_forecast_wind_daily
from models.wind_forecast_hybrid_hourly import run_forecast as run_hybrid_forecast_wind_hourly
from models.wind_forecast_hybrid_daily import run_forecast as run_hybrid_forecast_wind_daily
from db import SessionLocal
from model import Forecast, DHRForecast, ESNForecast, HybridForecast , HistoryLog
from sqlalchemy.orm import Session

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

router = APIRouter()

# Mount static files directory for serving generated files
router.mount("/static", StaticFiles(directory="forecasts"), name="static")

@router.post("/upload/dhr/hourly")
async def upload_file_dhr(
    original_filename: str = Form(...),
    tempFilename: str = Form(...),
    forecast_type: str = Form(...),
    granularity: str = Form(...),
    steps: int = Form(...),
    model: str = Form(...),
    max_capacity: float = Form(...),
    energy_demand: float = Form(...),
    fourier_terms: int = Form(...),
    reg_strength: float = Form(...),
    ar_order: int = Form(...),
    window: int = Form(...),
    polyorder: int = Form(...),
    temp_id: int = Form(...),
):
    temp_path = f"temp/{tempFilename}"
    db = SessionLocal()

    try:
        if not os.path.exists(temp_path):
            raise HTTPException(status_code=404, detail="Temp file not found")

        # Common parameters for both solar and wind forecasts
        params = {
            "fourier_terms": fourier_terms,
            "reg_strength": reg_strength,
            "ar_order": ar_order,
            "window": window,
            "polyorder": polyorder
        }
        
        # Run appropriate forecast based on type
        if forecast_type == "solar":
            output_files, meta = run_dhr_forecast_solar_hourly(
                temp_path,
                forecast_type=forecast_type,
                steps=steps,
                params=params
            )
        elif forecast_type == "wind":
            output_files, meta = run_dhr_forecast_wind_hourly(
                temp_path,
                forecast_type=forecast_type,
                steps=steps,
                params=params
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported forecast type: {forecast_type}")
        
        # Log the generated files
        logger.info(f"Generated files: {output_files}")

        # Save forecast entry
        forecast_entry = Forecast(
            original_filename=original_filename,
            filename=os.path.basename(output_files[0]),
            forecast_type=forecast_type,
            granularity=granularity,
            steps=steps,
            model=model,
            energy_demand=energy_demand,
            max_capacity=max_capacity,
            temp_id=temp_id,
        )
        db.add(forecast_entry)
        db.commit()
        db.refresh(forecast_entry)

        # Save DHR configuration
        dhr_entry = DHRForecast(
            forecast_id=forecast_entry.id,
            fourier_terms=fourier_terms,
            reg_strength=reg_strength,
            ar_order=ar_order,
            window=window,
            polyorder=polyorder
        )
        db.add(dhr_entry)
        db.commit()

        # Add to history log
        history_log_entry = HistoryLog(
            forecast_id=forecast_entry.id,
            file_name=original_filename,
            forecast_type=forecast_type,
            granularity=granularity,
            steps=steps,
            model="DHR"
        )
        db.add(history_log_entry)
        db.commit()

        # Fix download URL construction
        download_urls = [
            f"http://localhost:8000/download/{os.path.basename(file)}"
            for file in output_files
        ]

        return {
            "message": "Files processed successfully",
            "download_urls": download_urls,
            "forecast_id": forecast_entry.id
        }

    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        db.close()

@router.post("/upload/edit-dhr/hourly")
async def upload_file_dhr(
    tempFilename: str = Form(...),
    forecast_type: str = Form(...),
    steps: int = Form(...),
    fourier_terms: int = Form(...),
    reg_strength: float = Form(...),
    ar_order: int = Form(...),
    window: int = Form(...),
    polyorder: int = Form(...),
    forecast_id: int = Form(...),
):
    temp_path = f"temp/{tempFilename}"
    db = SessionLocal()

    try:
        if not os.path.exists(temp_path):
            raise HTTPException(status_code=404, detail="Temp file not found")

        # Common parameters for both solar and wind forecasts
        params = {
            "fourier_terms": fourier_terms,
            "reg_strength": reg_strength,
            "ar_order": ar_order,
            "window": window,
            "polyorder": polyorder
        }
        
        # Run appropriate forecast based on type
        if forecast_type == "solar":
            output_files, meta = run_dhr_forecast_solar_hourly(
                temp_path,
                forecast_type=forecast_type,
                steps=steps,
                params=params
            )
        elif forecast_type == "wind":
            output_files, meta = run_dhr_forecast_wind_hourly(
                temp_path,
                forecast_type=forecast_type,
                steps=steps,
                params=params
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported forecast type: {forecast_type}")

        # Log the generated files
        logger.info(f"Generated files: {output_files}")

        # Retrieve the existing forecast entry
        forecast_entry = db.query(Forecast).filter(Forecast.id == forecast_id).first()

        # Update the filename
        forecast_entry.filename = os.path.basename(output_files[0])
        db.commit()
        db.refresh(forecast_entry)  # Optional but good for safety

        # Fix download URL construction - match ESN format
        download_urls = [
            f"http://localhost:8000/download/{os.path.basename(file)}"
            for file in output_files
        ]

        return {
            "message": "Files processed successfully",
            "download_urls": download_urls,
        }

    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        db.close()

@router.post("/upload/dhr/daily")
async def upload_file_dhr(
    original_filename: str = Form(...),
    tempFilename: str = Form(...),
    forecast_type: str = Form(...),
    granularity: str = Form(...),
    steps: int = Form(...),
    model: str = Form(...),
    max_capacity: float = Form(...),
    energy_demand: float = Form(...),
    fourier_terms: int = Form(...),
    reg_strength: float = Form(...),
    ar_order: int = Form(...),
    window: int = Form(...),
    polyorder: int = Form(...),
    temp_id: int = Form(...),
):
    temp_path = f"temp/{tempFilename}"
    db = SessionLocal()

    try:
        if not os.path.exists(temp_path):
            raise HTTPException(status_code=404, detail="Temp file not found")

        if forecast_type == "solar":
            output_files, meta = run_dhr_forecast_solar_daily(
                temp_path,
                forecast_type=forecast_type,
                steps=steps,
                params={
                    "fourier_terms": fourier_terms,
                    "reg_strength": reg_strength,
                    "ar_order": ar_order,
                    "window": window,
                    "polyorder": polyorder
                }
            )
        elif forecast_type == "wind":
            output_files, meta = run_dhr_forecast_wind_daily(
                temp_path,
                forecast_type=forecast_type,
                steps=steps,
                params={
                    "fourier_terms": fourier_terms,
                    "reg_strength": reg_strength,
                    "ar_order": ar_order,
                    "window": window,
                    "polyorder": polyorder
                }
            )
            
        # Log the generated files
        logger.info(f"Generated files: {output_files}")

        # Save forecast entry
        forecast_entry = Forecast(
            original_filename=original_filename,
            filename=os.path.basename(output_files[0]),
            forecast_type=forecast_type,
            granularity=granularity,
            steps=steps,
            model=model,
            energy_demand=energy_demand,
            max_capacity=max_capacity,
            temp_id = temp_id,
        )
        db.add(forecast_entry)
        db.commit()
        db.refresh(forecast_entry)

        # Save DHR configuration
        dhr_entry = DHRForecast(
            forecast_id=forecast_entry.id,
            fourier_terms=fourier_terms,
            reg_strength=reg_strength,
            ar_order=ar_order,
            window=window,
            polyorder=polyorder
        )
        db.add(dhr_entry)
        db.commit()

        # Add to history log
        history_log_entry = HistoryLog(
            forecast_id=forecast_entry.id,
            file_name=original_filename,
            forecast_type=forecast_type,
            granularity=granularity,
            steps=steps,
            model="DHR"
        )
        db.add(history_log_entry)
        db.commit()

        # Fix download URL construction - match ESN format
        download_urls = [
            f"http://localhost:8000/download/{os.path.basename(file)}"
            for file in output_files
        ]

        return {
            "message": "Files processed successfully",
            "download_urls": download_urls,
            "forecast_id": forecast_entry.id
        }

    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        db.close()

@router.post("/upload/edit-dhr/daily")
async def upload_file_dhr(
    tempFilename: str = Form(...),
    forecast_type: str = Form(...),
    steps: int = Form(...),
    fourier_terms: int = Form(...),
    reg_strength: float = Form(...),
    ar_order: int = Form(...),
    window: int = Form(...),
    polyorder: int = Form(...),
    forecast_id: int = Form(...),
):
    temp_path = f"temp/{tempFilename}"
    db = SessionLocal()

    try:
        if not os.path.exists(temp_path):
            raise HTTPException(status_code=404, detail="Temp file not found")

        params = {
            "fourier_terms": fourier_terms,
            "reg_strength": reg_strength,
            "ar_order": ar_order,
            "window": window,
            "polyorder": polyorder
        }
        
        if forecast_type == "solar":
            output_files, meta = run_dhr_forecast_solar_daily(
                temp_path,
                forecast_type=forecast_type,
                steps=steps,
                params=params
            )
        elif forecast_type == "wind":
            output_files, meta = run_dhr_forecast_wind_daily(
                temp_path,
                forecast_type=forecast_type,
                steps=steps,
                params=params
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported forecast type: {forecast_type}")

        # Log the generated files
        logger.info(f"Generated files: {output_files}")

        # Retrieve the existing forecast entry
        forecast_entry = db.query(Forecast).filter(Forecast.id == forecast_id).first()
        if not forecast_entry:
            raise HTTPException(status_code=404, detail="Forecast not found")

        # Update the filename (CSV)
        forecast_entry.filename = os.path.basename(output_files[0])
        db.commit()
        db.refresh(forecast_entry)

        # Update DHR configuration
        dhr_entry = db.query(DHRForecast).filter(DHRForecast.forecast_id == forecast_id).first()
        if dhr_entry:
            dhr_entry.fourier_terms = fourier_terms
            dhr_entry.reg_strength = reg_strength
            dhr_entry.ar_order = ar_order
            dhr_entry.window = window
            dhr_entry.polyorder = polyorder
            db.commit()
            db.refresh(dhr_entry)

        download_urls = [
            f"http://localhost:8000/download/{os.path.basename(file)}"
            for file in output_files
        ]

        return {
            "message": "Files processed successfully",
            "download_urls": download_urls,
        }

    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        db.close()
        
@router.post("/upload/esn/hourly")
async def upload_file_esn(
    original_filename: str = Form(...),
    tempFilename: str = Form(...),
    forecast_type: str = Form(...),
    granularity: str = Form(...),
    steps: int = Form(...),
    model: str = Form(...),
    max_capacity: float = Form(...),
    energy_demand: float = Form(...),
    lags: int = Form(...),
    N_res: int = Form(...),
    rho: float = Form(...),
    alpha: float = Form(...),
    sparsity: float = Form(...),
    lambda_reg: float = Form(...),
    temp_id: int = Form(...),
):
    temp_path = f"temp/{tempFilename}"
    db = SessionLocal()

    try:
        if not os.path.exists(temp_path):
            raise HTTPException(status_code=404, detail="Temp file not found")

        if forecast_type == "solar":
            output_files = run_esn_forecast_solar_hourly(
                csv_path=temp_path,
                forecast_type=forecast_type,
                steps=steps,
                params={
                    "lags": lags,
                    "N_res": N_res,
                    "rho": rho,
                    "alpha": alpha,
                    "sparsity": sparsity,
                    "lambda_reg": lambda_reg,
                }
            )
        elif forecast_type == "wind":
            output_files = run_esn_forecast_wind_hourly(
                csv_path=temp_path,
                forecast_type=forecast_type,
                steps=steps,
                params={
                    "lags": lags,
                    "N_res": N_res,
                    "rho": rho,
                    "alpha": alpha,
                    "sparsity": sparsity,
                    "lambda_reg": lambda_reg,
                }
            )
        
        logger.info(f"Generated files: {output_files}")

        forecast_entry = Forecast(
            original_filename=original_filename,
            filename=os.path.basename(output_files[0]),
            forecast_type=forecast_type,
            granularity=granularity,
            steps=steps,
            model=model,
            energy_demand=energy_demand,
            max_capacity=max_capacity,
            temp_id = temp_id,
        )
        db.add(forecast_entry)
        db.commit()
        db.refresh(forecast_entry)

        esn_entry = ESNForecast(
            forecast_id=forecast_entry.id,
            lags=lags,
            N_res=N_res,
            rho=rho,
            alpha=alpha,
            sparsity=sparsity,
            lambda_reg=lambda_reg,
        )
        db.add(esn_entry)
        db.commit()
        db.refresh(esn_entry)

        history_log_entry = HistoryLog(
            forecast_id=forecast_entry.id,
            file_name=original_filename,
            forecast_type=forecast_type,
            granularity=granularity,
            steps=steps,
            model=model
        )
        db.add(history_log_entry)
        db.commit()
        db.refresh(history_log_entry)

        download_urls = [
            f"http://localhost:8000/download/{os.path.basename(file)}"
            for file in output_files
        ]

        return {"download_urls": download_urls, "forecast_id": forecast_entry.id}

    except Exception as e:
        db.rollback()
        return {"message": str(e)}

    finally:
        db.close()

@router.post("/upload/edit-esn/hourly")
async def upload_file_esn(
    tempFilename: str = Form(...),
    forecast_type: str = Form(...),
    steps: int = Form(...),
    lags: int = Form(...),
    N_res: int = Form(...),
    rho: float = Form(...),
    alpha: float = Form(...),
    sparsity: float = Form(...),
    lambda_reg: float = Form(...),
    forecast_id: int = Form(...),
):
    temp_path = f"temp/{tempFilename}"
    db = SessionLocal()

    try:
        if not os.path.exists(temp_path):
            raise HTTPException(status_code=404, detail="Temp file not found")

        if forecast_type == "solar":
            output_files = run_esn_forecast_solar_hourly(
                csv_path=temp_path,
                forecast_type=forecast_type,
                steps=steps,
                params={
                    "lags": lags,
                    "N_res": N_res,
                    "rho": rho,
                    "alpha": alpha,
                    "sparsity": sparsity,
                    "lambda_reg": lambda_reg,
                }
            )
        elif forecast_type == "wind":
            output_files = run_esn_forecast_wind_hourly(
                csv_path=temp_path,
                forecast_type=forecast_type,
                steps=steps,
                params={
                    "lags": lags,
                    "N_res": N_res,
                    "rho": rho,
                    "alpha": alpha,
                    "sparsity": sparsity,
                    "lambda_reg": lambda_reg,
                }
            )
        
        logger.info(f"Generated files: {output_files}")

        # Retrieve the existing forecast entry
        forecast_entry = db.query(Forecast).filter(Forecast.id == forecast_id).first()

        # Update the filename
        forecast_entry.filename = os.path.basename(output_files[0])
        db.commit()
        db.refresh(forecast_entry)  # Optional but good for safety

        download_urls = [
            f"http://localhost:8000/download/{os.path.basename(file)}"
            for file in output_files
        ]

        return {"download_urls": download_urls}

    except Exception as e:
        db.rollback()
        return {"message": str(e)}

    finally:
        db.close()

@router.post("/upload/esn/daily")
async def upload_file_esn(
    original_filename: str = Form(...),
    tempFilename: str = Form(...),
    forecast_type: str = Form(...),
    granularity: str = Form(...),
    steps: int = Form(...),
    model: str = Form(...),
    max_capacity: float = Form(...),
    energy_demand: float = Form(...),
    lags: int = Form(...),
    N_res: int = Form(...),
    rho: float = Form(...),
    alpha: float = Form(...),
    sparsity: float = Form(...),
    lambda_reg: float = Form(...),
    temp_id: int = Form(...),
):
    temp_path = f"temp/{tempFilename}"
    db = SessionLocal()

    try:
        if not os.path.exists(temp_path):
            raise HTTPException(status_code=404, detail="Temp file not found")

        if forecast_type == "solar":
            output_files = run_esn_forecast_solar_daily(
                csv_path=temp_path,
                forecast_type=forecast_type,
                steps=steps,
                params={
                    "lags": lags,
                    "N_res": N_res,
                    "rho": rho,
                    "alpha": alpha,
                    "sparsity": sparsity,
                    "lambda_reg": lambda_reg,
                }
            )
        elif forecast_type == "wind":
            output_files = run_esn_forecast_wind_daily(
                csv_path=temp_path,
                forecast_type=forecast_type,
                steps=steps,
                params={
                    "lags": lags,
                    "N_res": N_res,
                    "rho": rho,
                    "alpha": alpha,
                    "sparsity": sparsity,
                    "lambda_reg": lambda_reg,
                }
            )
        
        logger.info(f"Generated files: {output_files}")

        forecast_entry = Forecast(
            original_filename=original_filename,
            filename=os.path.basename(output_files[0]),
            forecast_type=forecast_type,
            granularity=granularity,
            steps=steps,
            model=model,
            energy_demand=energy_demand,
            max_capacity=max_capacity,
            temp_id = temp_id,
        )
        db.add(forecast_entry)
        db.commit()
        db.refresh(forecast_entry)

        esn_entry = ESNForecast(
            forecast_id=forecast_entry.id,
            lags=lags,
            N_res=N_res,
            rho=rho,
            alpha=alpha,
            sparsity=sparsity,
            lambda_reg=lambda_reg,
        )
        db.add(esn_entry)
        db.commit()
        db.refresh(esn_entry)

        history_log_entry = HistoryLog(
            forecast_id=forecast_entry.id,
            file_name=original_filename,
            forecast_type=forecast_type,
            granularity=granularity,
            steps=steps,
            model=model
        )
        db.add(history_log_entry)
        db.commit()
        db.refresh(history_log_entry)

        download_urls = [
            f"http://localhost:8000/download/{os.path.basename(file)}"
            for file in output_files
        ]

        return {"download_urls": download_urls, "forecast_id": forecast_entry.id}

    except Exception as e:
        db.rollback()
        return {"message": str(e)}

    finally:
        db.close()

@router.post("/upload/edit-esn/daily")
async def upload_file_esn(
    tempFilename: str = Form(...),
    forecast_type: str = Form(...),
    steps: int = Form(...),
    lags: int = Form(...),
    N_res: int = Form(...),
    rho: float = Form(...),
    alpha: float = Form(...),
    sparsity: float = Form(...),
    lambda_reg: float = Form(...),
    forecast_id: int = Form(...),
):
    temp_path = f"temp/{tempFilename}"
    db = SessionLocal()

    try:
        if not os.path.exists(temp_path):
            raise HTTPException(status_code=404, detail="Temp file not found")

        if forecast_type == "solar":
            output_files = run_esn_forecast_solar_daily(
                csv_path=temp_path,
                forecast_type=forecast_type,
                steps=steps,
                params={
                    "lags": lags,
                    "N_res": N_res,
                    "rho": rho,
                    "alpha": alpha,
                    "sparsity": sparsity,
                    "lambda_reg": lambda_reg,
                }
            )
        elif forecast_type == "wind":
            output_files = run_esn_forecast_wind_daily(
                csv_path=temp_path,
                forecast_type=forecast_type,
                steps=steps,
                params={
                    "lags": lags,
                    "N_res": N_res,
                    "rho": rho,
                    "alpha": alpha,
                    "sparsity": sparsity,
                    "lambda_reg": lambda_reg,
                }
            )
        
        logger.info(f"Generated files: {output_files}")

        # Retrieve the existing forecast entry
        forecast_entry = db.query(Forecast).filter(Forecast.id == forecast_id).first()

        # Update the filename
        forecast_entry.filename = os.path.basename(output_files[0])
        db.commit()
        db.refresh(forecast_entry)  # Optional but good for safety

        download_urls = [
            f"http://localhost:8000/download/{os.path.basename(file)}"
            for file in output_files
        ]

        return {"download_urls": download_urls}

    except Exception as e:
        db.rollback()
        return {"message": str(e)}

    finally:
        db.close()

@router.post("/upload/hybrid/hourly")
async def upload_file_hybrid(
    original_filename: str = Form(...),
    tempFilename: str = Form(...),
    forecast_type: str = Form(...),
    granularity: str = Form(...),
    steps: int = Form(...),
    model: str = Form(...),
    max_capacity: float = Form(...),
    energy_demand: float = Form(...),
    fourier_terms: int = Form(...),
    reg_strength: float = Form(...),
    ar_order: int = Form(...),
    window: int = Form(...),
    polyorder: int = Form(...),
    N_res: int = Form(...),
    rho: float = Form(...),
    sparsity: float = Form(...),
    alpha: float = Form(...),
    lambda_reg: float = Form(...),
    lags: int = Form(...),
    n_features: int = Form(...),  
    temp_id: int = Form(...)
):

    temp_path = f"temp/{tempFilename}"
    db: Session = SessionLocal()

    try:
        if not os.path.exists(temp_path):
            raise HTTPException(status_code=404, detail="Temp file not found")

        # Call the hybrid forecast function using the structured configs
        if forecast_type == "solar":
            output_files = run_hybrid_forecast_solar_hourly(
                csv_path=temp_path,
                forecast_type=forecast_type,
                steps=steps,
                params={
                    "fourier_terms": fourier_terms,
                    "reg_strength": reg_strength,
                    "ar_order": ar_order,
                    "window": window,
                    "polyorder": polyorder,
                    "lags": lags,
                    "N_res": N_res,
                    "rho": rho,
                    "alpha": alpha,
                    "sparsity": sparsity,
                    "lambda_reg": lambda_reg,
                    "n_features": n_features,
                }
            )
        elif forecast_type == "wind":
            output_files = run_hybrid_forecast_wind_hourly(
                csv_path=temp_path,
                forecast_type=forecast_type,
                steps=steps,
                params={
                    "fourier_terms": fourier_terms,
                    "reg_strength": reg_strength,
                    "ar_order": ar_order,
                    "window": window,
                    "polyorder": polyorder,
                    "lags": lags,
                    "N_res": N_res,
                    "rho": rho,
                    "alpha": alpha,
                    "sparsity": sparsity,
                    "lambda_reg": lambda_reg,
                    "n_features": n_features,
                }
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid forecast type")
        
        logger.info(f"Generated files: {output_files}")

        # Save forecast metadata to database
        forecast_entry = Forecast(
            original_filename=original_filename,
            filename=os.path.basename(output_files[0]),
            forecast_type=forecast_type,
            granularity=granularity,
            steps=steps,
            model=model,
            energy_demand=energy_demand,
            max_capacity=max_capacity,
            temp_id=temp_id,
        )
        db.add(forecast_entry)
        db.commit()
        db.refresh(forecast_entry)

        hybrid_entry = HybridForecast(
            forecast_id=forecast_entry.id,
            fourier_terms=fourier_terms,
            reg_strength=reg_strength,
            ar_order=ar_order,
            window=window,
            polyorder=polyorder,
            N_res=N_res,
            rho=rho,
            sparsity=sparsity,
            alpha=alpha,
            lambda_reg=lambda_reg,
            lags=lags,
            n_features=n_features
        )
        db.add(hybrid_entry)
        db.commit()
        db.refresh(hybrid_entry)

        history_log_entry = HistoryLog(
            forecast_id=forecast_entry.id,
            file_name=original_filename,
            forecast_type=forecast_type,
            granularity=granularity,
            steps=steps,
            model=model
        )
        db.add(history_log_entry)
        db.commit()
        db.refresh(history_log_entry)

        download_urls = [
            f"http://localhost:8000/download/{os.path.basename(file)}"
            for file in output_files
        ]

        return {
                "message": "Files processed successfully",
                "download_urls": download_urls,
                "forecast_id": forecast_entry.id
            }

    except Exception as e:
        db.rollback()
        return {"message": str(e)}

    finally:
        db.close()

@router.post("/upload/edit-hybrid/hourly")
async def upload_file_hybrid(
    tempFilename: str = Form(...),
    forecast_type: str = Form(...),
    forecast_id: int = Form(...),  # Added forecast_id parameter
    steps: int = Form(...),
    fourier_terms: int = Form(...),
    reg_strength: float = Form(...),
    ar_order: int = Form(...),
    window: int = Form(...),
    polyorder: int = Form(...),
    N_res: int = Form(...),
    rho: float = Form(...),
    sparsity: float = Form(...),
    alpha: float = Form(...),
    lambda_reg: float = Form(...),
    lags: int = Form(...),
    n_features: int = Form(...),
):

    temp_path = f"temp/{tempFilename}"
    db: Session = SessionLocal()

    try:
        if not os.path.exists(temp_path):
            raise HTTPException(status_code=404, detail="Temp file not found")

        # Call the hybrid forecast function using the structured configs
        if forecast_type == "solar":
            output_files = run_hybrid_forecast_solar_hourly(
                csv_path=temp_path,
                forecast_type=forecast_type,
                steps=steps,
                params={
                    "fourier_terms": fourier_terms,
                    "reg_strength": reg_strength,
                    "ar_order": ar_order,
                    "window": window,
                    "polyorder": polyorder,
                    "lags": lags,
                    "N_res": N_res,
                    "rho": rho,
                    "alpha": alpha,
                    "sparsity": sparsity,
                    "lambda_reg": lambda_reg,
                    "n_features": n_features,
                }
            )
        elif forecast_type == "wind":
            output_files = run_hybrid_forecast_wind_hourly(
                csv_path=temp_path,
                forecast_type=forecast_type,
                steps=steps,
                params={
                    "fourier_terms": fourier_terms,
                    "reg_strength": reg_strength,
                    "ar_order": ar_order,
                    "window": window,
                    "polyorder": polyorder,
                    "lags": lags,
                    "N_res": N_res,
                    "rho": rho,
                    "alpha": alpha,
                    "sparsity": sparsity,
                    "lambda_reg": lambda_reg,
                    "n_features": n_features,
                }
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid forecast type")
        
        logger.info(f"Generated files: {output_files}")

        # Retrieve the existing forecast entry
        forecast_entry = db.query(Forecast).filter(Forecast.id == forecast_id).first()

        # Update the filename
        forecast_entry.filename = os.path.basename(output_files[0])
        db.commit()
        db.refresh(forecast_entry)  # Optional but good for safety

        download_urls = [
            f"http://localhost:8000/download/{os.path.basename(file)}"
            for file in output_files
        ]

        return {
                "message": "Files processed successfully",
                "download_urls": download_urls,
            }

    except Exception as e:
        db.rollback()
        return {"message": str(e)}

    finally:
        db.close()

@router.post("/upload/hybrid/daily")
async def upload_file_hybrid(
    original_filename: str = Form(...),
    tempFilename: str = Form(...),
    forecast_type: str = Form(...),
    granularity: str = Form(...),
    steps: int = Form(...),
    model: str = Form(...),
    max_capacity: float = Form(...),
    energy_demand: float = Form(...),
    fourier_terms: int = Form(...),
    reg_strength: float = Form(...),
    ar_order: int = Form(...),
    window: int = Form(...),
    polyorder: int = Form(...),
    N_res: int = Form(...),
    rho: float = Form(...),
    sparsity: float = Form(...),
    alpha: float = Form(...),
    lambda_reg: float = Form(...),
    lags: int = Form(...),
    n_features: int = Form(...),
    temp_id: int = Form(...)
):

    temp_path = f"temp/{tempFilename}"
    db: Session = SessionLocal()

    try:
        if not os.path.exists(temp_path):
            raise HTTPException(status_code=404, detail="Temp file not found")

        # Call the hybrid forecast function using the structured configs
        if forecast_type == "solar":
            output_files = run_hybrid_forecast_solar_daily(
                csv_path=temp_path,
                forecast_type=forecast_type,
                steps=steps,
                params={
                    "fourier_terms": fourier_terms,
                    "reg_strength": reg_strength,
                    "ar_order": ar_order,
                    "window": window,
                    "polyorder": polyorder,
                    "lags": lags,
                    "N_res": N_res,
                    "rho": rho,
                    "alpha": alpha,
                    "sparsity": sparsity,
                    "lambda_reg": lambda_reg,
                    "n_features": n_features,
                }
            )
        elif forecast_type == "wind":
            output_files = run_hybrid_forecast_wind_daily(
                csv_path=temp_path,
                forecast_type=forecast_type,
                steps=steps,
                params={
                    "fourier_terms": fourier_terms,
                    "reg_strength": reg_strength,
                    "ar_order": ar_order,
                    "window": window,
                    "polyorder": polyorder,
                    "lags": lags,
                    "N_res": N_res,
                    "rho": rho,
                    "alpha": alpha,
                    "sparsity": sparsity,
                    "lambda_reg": lambda_reg,
                    "n_features": n_features,
                }
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid forecast type")
        
        logger.info(f"Generated files: {output_files}")

        # Save forecast metadata to database
        forecast_entry = Forecast(
            original_filename=original_filename,
            filename=os.path.basename(output_files[0]),
            forecast_type=forecast_type,
            granularity=granularity,
            steps=steps,
            model=model,
            energy_demand=energy_demand,
            max_capacity=max_capacity,
            temp_id=temp_id,
        )
        db.add(forecast_entry)
        db.commit()
        db.refresh(forecast_entry)

        hybrid_entry = HybridForecast(
            forecast_id=forecast_entry.id,
            fourier_terms=fourier_terms,
            reg_strength=reg_strength,
            ar_order=ar_order,
            window=window,
            polyorder=polyorder,
            N_res=N_res,
            rho=rho,
            sparsity=sparsity,
            alpha=alpha,
            lambda_reg=lambda_reg,
            lags=lags,
            n_features=n_features
        )
        db.add(hybrid_entry)
        db.commit()
        db.refresh(hybrid_entry)

        history_log_entry = HistoryLog(
            forecast_id=forecast_entry.id,
            file_name=original_filename,
            forecast_type=forecast_type,
            granularity=granularity,
            steps=steps,
            model=model
        )
        db.add(history_log_entry)
        db.commit()
        db.refresh(history_log_entry)

        download_urls = [
            f"http://localhost:8000/download/{os.path.basename(file)}"
            for file in output_files
        ]

        return {
                "message": "Files processed successfully",
                "download_urls": download_urls,
                "forecast_id": forecast_entry.id
            }

    except Exception as e:
        db.rollback()
        return {"message": str(e)}

    finally:
        db.close()

@router.post("/upload/edit-hybrid/daily")
async def upload_file_hybrid(
    tempFilename: str = Form(...),
    forecast_type: str = Form(...),
    forecast_id: int = Form(...),  # Added forecast_id parameter
    steps: int = Form(...),
    fourier_terms: int = Form(...),
    reg_strength: float = Form(...),
    ar_order: int = Form(...),
    window: int = Form(...),
    polyorder: int = Form(...),
    N_res: int = Form(...),
    rho: float = Form(...),
    sparsity: float = Form(...),
    alpha: float = Form(...),
    lambda_reg: float = Form(...),
    lags: int = Form(...),
    n_features: int = Form(...),
):

    temp_path = f"temp/{tempFilename}"
    db: Session = SessionLocal()

    try:
        if not os.path.exists(temp_path):
            raise HTTPException(status_code=404, detail="Temp file not found")

        # Call the hybrid forecast function using the structured configs
        if forecast_type == "solar":
            output_files = run_hybrid_forecast_solar_daily(
                csv_path=temp_path,
                forecast_type=forecast_type,
                steps=steps,
                params={
                    "fourier_terms": fourier_terms,
                    "reg_strength": reg_strength,
                    "ar_order": ar_order,
                    "window": window,
                    "polyorder": polyorder,
                    "lags": lags,
                    "N_res": N_res,
                    "rho": rho,
                    "alpha": alpha,
                    "sparsity": sparsity,
                    "lambda_reg": lambda_reg,
                    "n_features": n_features,
                }
            )
        elif forecast_type == "wind":
            output_files = run_hybrid_forecast_wind_daily(
                csv_path=temp_path,
                forecast_type=forecast_type,
                steps=steps,
                params={
                    "fourier_terms": fourier_terms,
                    "reg_strength": reg_strength,
                    "ar_order": ar_order,
                    "window": window,
                    "polyorder": polyorder,
                    "lags": lags,
                    "N_res": N_res,
                    "rho": rho,
                    "alpha": alpha,
                    "sparsity": sparsity,
                    "lambda_reg": lambda_reg,
                    "n_features": n_features,
                }
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid forecast type")
        
        logger.info(f"Generated files: {output_files}")

        # Retrieve the existing forecast entry
        forecast_entry = db.query(Forecast).filter(Forecast.id == forecast_id).first()

        # Update the filename
        forecast_entry.filename = os.path.basename(output_files[0])
        db.commit()
        db.refresh(forecast_entry)  # Optional but good for safety

        download_urls = [
            f"http://localhost:8000/download/{os.path.basename(file)}"
            for file in output_files
        ]

        return {
                "message": "Files processed successfully",
                "download_urls": download_urls,
            }

    except Exception as e:
        db.rollback()
        return {"message": str(e)}

    finally:
        db.close()

@router.get("/download/{filename}")
async def get_file(filename: str):
    forecast_dir = "forecasts"
    file_path = os.path.join(forecast_dir, filename)
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")
    
    logger.info(f"Serving file: {file_path}")
    return FileResponse(file_path)

@router.get("/api/forecasts/{forecast_id}")
async def get_forecast(forecast_id: int, db: Session = Depends(get_db)):
    try:
        forecast = db.query(Forecast).filter(Forecast.id == forecast_id).first()
        
        if not forecast:
            raise HTTPException(status_code=404, detail=f"Forecast {forecast_id} not found")
            
        return {
            "id": forecast.id,
            "filename": forecast.filename,
            "original_filename": forecast.original_filename,
            "model": forecast.model,
            "maxCapacity": forecast.max_capacity,
            "energyDemand": forecast.energy_demand,
            "steps": forecast.steps,
            "granularity": forecast.granularity,
            "created_at": forecast.created_at.isoformat() if forecast.created_at else None
        }
    except Exception as e:
            logger.error(f"Error fetching forecast {forecast_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
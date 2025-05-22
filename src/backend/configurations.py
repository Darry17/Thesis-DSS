from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from datetime import datetime
import logging
from model import Forecast, DHRForecast, ESNForecast, HybridForecast, DHRForecastCreate, ESNForecastCreate, HybridForecastCreate  # Import models from model.py

# Configure logging
logger = logging.getLogger(__name__)

# Routes
def register_configuration_routes(app: FastAPI, get_db):
    @app.post("/api/dhr-configurations")
    async def create_dhr_configuration(config: DHRForecastCreate, db: Session = Depends(get_db)):
        try:
            logger.info(f"Creating DHR configuration: {config}")
            
            forecast = db.query(Forecast).filter(Forecast.id == config.forecast_id).first()
            if not forecast:
                raise HTTPException(status_code=404, detail="Forecast not found")

            db_config = DHRForecast(
                forecast_id=config.forecast_id,
                fourier_terms=config.fourier_terms,
                reg_strength=config.reg_strength,
                ar_order=config.ar_order,
                window=config.window,
                polyorder=config.polyorder
            )
            
            db.add(db_config)
            db.commit()
            db.refresh(db_config)

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

    @app.get("/api/dhr-configurations/{forecast_id}")
    async def get_dhr_configuration(forecast_id: int, db: Session = Depends(get_db)):
        try:
            config = db.query(DHRForecast).filter(DHRForecast.forecast_id == forecast_id).first()
            if not config:
                raise HTTPException(status_code=404, detail="Configuration not found")
            
            return {
                "id": config.id,
                "forecast_id": config.forecast_id,
                "fourier_terms": config.fourier_terms,
                "reg_strength": config.reg_strength,
                "ar_order": config.ar_order,
                "window": config.window,
                "polyorder": config.polyorder
            }
        except Exception as e:
            logger.error(f"Error fetching DHR configuration: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.put("/api/dhr-configurations/{forecast_id}")
    async def update_dhr_configuration(forecast_id: int, config: DHRForecastCreate, db: Session = Depends(get_db)):
        try:
            forecast = db.query(Forecast).filter(Forecast.id == forecast_id).first()
            if not forecast:
                raise HTTPException(status_code=404, detail="Forecast not found")
                
            existing_config = db.query(DHRForecast).filter(
                DHRForecast.forecast_id == forecast_id
            ).first()
            
            if not existing_config:
                raise HTTPException(status_code=404, detail="DHR configuration not found")
            
            existing_config.fourier_terms = config.fourier_terms
            existing_config.reg_strength = config.reg_strength
            existing_config.ar_order = config.ar_order
            existing_config.window = config.window
            existing_config.polyorder = config.polyorder
            existing_config.updated_at = datetime.now()
            
            db.commit()
            db.refresh(existing_config)
            
            return {
                "id": existing_config.id,
                "message": "DHR configuration updated successfully"
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating DHR configuration: {str(e)}")
            db.rollback()
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/esn-configurations")
    async def create_esn_configuration(config: ESNForecastCreate, db: Session = Depends(get_db)):
        try:
            logger.info(f"Creating ESN configuration: {config}")
            
            forecast = db.query(Forecast).filter(Forecast.id == config.forecast_id).first()
            if not forecast:
                raise HTTPException(status_code=404, detail="Forecast not found")

            db_config = ESNForecast(
                forecast_id=config.forecast_id,
                N_res=config.N_res,
                rho=config.rho,
                sparsity=config.sparsity,
                alpha=config.alpha,
                lambda_reg=config.lambda_reg,
                lags=config.lags
            )
            
            db.add(db_config)
            db.commit()
            db.refresh(db_config)
            
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

    @app.get("/api/esn-configurations/{forecast_id}")
    async def get_esn_configuration(forecast_id: int, db: Session = Depends(get_db)):
        try:
            config = db.query(ESNForecast).filter(ESNForecast.forecast_id == forecast_id).first()
            if not config:
                raise HTTPException(status_code=404, detail="Configuration not found")
            
            return {
                "id": config.id,
                "forecast_id": config.forecast_id,
                "N_res": config.N_res,
                "rho": config.rho,
                "sparsity": config.sparsity,
                "alpha": config.alpha,
                "lambda_reg": config.lambda_reg,
                "lags": config.lags
            }
        except Exception as e:
            logger.error(f"Error fetching ESN configuration: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.put("/api/esn-configurations/{forecast_id}")
    async def update_esn_configuration(forecast_id: int, config: ESNForecastCreate, db: Session = Depends(get_db)):
        try:
            forecast = db.query(Forecast).filter(Forecast.id == forecast_id).first()
            if not forecast:
                raise HTTPException(status_code=404, detail="Forecast not found")
                
            existing_config = db.query(ESNForecast).filter(
                ESNForecast.forecast_id == forecast_id
            ).first()
            
            if not existing_config:
                raise HTTPException(status_code=404, detail="ESN configuration not found")
            
            existing_config.N_res = config.N_res
            existing_config.rho = config.rho
            existing_config.sparsity = config.sparsity
            existing_config.alpha = config.alpha
            existing_config.lambda_reg = config.lambda_reg
            existing_config.lags = config.lags
            existing_config.updated_at = datetime.now()
            
            db.commit()
            db.refresh(existing_config)
            
            return {
                "id": existing_config.id,
                "message": "ESN configuration updated successfully"
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating ESN configuration: {str(e)}")
            db.rollback()
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/hybrid-configurations")
    async def create_hybrid_configuration(config: HybridForecastCreate, db: Session = Depends(get_db)):
        try:
            logger.info(f"Creating Hybrid configuration: {config}")
            
            forecast = db.query(Forecast).filter(Forecast.id == config.forecast_id).first()
            if not forecast:
                raise HTTPException(status_code=404, detail="Forecast not found")

            db_config = HybridForecast(
                forecast_id=config.forecast_id,
                fourier_terms=config.fourier_terms,
                reg_strength=config.reg_strength,
                ar_order=config.ar_order,
                window=config.window,
                polyorder=config.polyorder,
                N_res=config.N_res,
                rho=config.rho,
                sparsity=config.sparsity,
                alpha=config.alpha,
                lambda_reg=config.lambda_reg,
                lags=config.lags,
            )
            
            db.add(db_config)
            db.commit()
            db.refresh(db_config)
            
            return {
                "id": db_config.id,
                "message": "Hybrid configuration created successfully"
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error creating Hybrid configuration: {str(e)}")
            db.rollback()
            raise HTTPException(status_code=500, detail=str(e))


    @app.get("/api/hybrid-configurations/{forecast_id}")
    async def get_hybrid_configuration(forecast_id: int, db: Session = Depends(get_db)):
        try:
            config = db.query(HybridForecast).filter(HybridForecast.forecast_id == forecast_id).first()
            if not config:
                raise HTTPException(status_code=404, detail="Configuration not found")
            
            return {
                "id": config.id,
                "forecast_id": config.forecast_id,
                "fourier_terms": config.fourier_terms,
                "reg_strength": config.reg_strength,
                "ar_order": config.ar_order,
                "window": config.window,
                "polyorder": config.polyorder,
                "N_res": config.N_res,
                "rho": config.rho,
                "sparsity": config.sparsity,
                "alpha": config.alpha,
                "lambda_reg": config.lambda_reg,
                "lags": config.lags,
            }
        except Exception as e:
            logger.error(f"Error fetching Hybrid configuration: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))


    @app.put("/api/hybrid-configurations/{forecast_id}")
    async def update_hybrid_configuration(forecast_id: int, config: HybridForecastCreate, db: Session = Depends(get_db)):
        try:
            forecast = db.query(Forecast).filter(Forecast.id == forecast_id).first()
            if not forecast:
                raise HTTPException(status_code=404, detail="Forecast not found")
                
            existing_config = db.query(HybridForecast).filter(
                HybridForecast.forecast_id == forecast_id
            ).first()
            
            if not existing_config:
                raise HTTPException(status_code=404, detail="Hybrid configuration not found")
            
            existing_config.fourier_terms = config.fourier_terms
            existing_config.reg_strength = config.reg_strength
            existing_config.ar_order = config.ar_order
            existing_config.window = config.window
            existing_config.polyorder = config.polyorder
            existing_config.N_res = config.N_res
            existing_config.rho = config.rho
            existing_config.sparsity = config.sparsity
            existing_config.alpha = config.alpha
            existing_config.lambda_reg = config.lambda_reg
            existing_config.lags = config.lags
            existing_config.updated_at = datetime.now()
            
            db.commit()
            db.refresh(existing_config)
            
            return {
                "id": existing_config.id,
                "message": "Hybrid configuration updated successfully"
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating Hybrid configuration: {str(e)}")
            db.rollback()
            raise HTTPException(status_code=500, detail=str(e))
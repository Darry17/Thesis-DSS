from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy import Column, Integer, String, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime
import logging

# Configure logging
logger = logging.getLogger(__name__)
# SQLAlchemy Models
Base = declarative_base()
# SQLAlchemy Models
class DHRConfiguration(Base):
    __tablename__ = "dhr_configurations"
    id = Column(Integer, primary_key=True, index=True)
    forecast_id = Column(Integer, nullable=False)
    fourier_order = Column(Integer, nullable=False)
    window_length = Column(Integer, nullable=False)
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

# Pydantic Models
class DHRConfigurationCreate(BaseModel):
    forecast_id: int
    fourier_order: int
    window_length: int
    polyorder: float
    regularization_dhr: float
    trend_components: int
    model_config = {'from_attributes': True}

class ESNConfigurationCreate(BaseModel):
    forecast_id: int
    reservoir_size: int
    spectral_radius: float
    sparsity: float
    input_scaling: float
    dropout: float
    lags: int
    regularization_esn: float
    model_config = {'from_attributes': True}

class HybridConfigurationCreate(BaseModel):
    forecast_id: int
    fourier_order: int
    window_length: int
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
    model_config = {'from_attributes': True}

# Routes
def register_configuration_routes(app: FastAPI, get_db, Forecast):
    @app.post("/api/dhr-configurations")
    async def create_dhr_configuration(config: DHRConfigurationCreate, db: Session = Depends(get_db)):
        try:
            logger.info(f"Creating DHR configuration: {config}")
            
            forecast = db.query(Forecast).filter(Forecast.id == config.forecast_id).first()
            if not forecast:
                raise HTTPException(status_code=404, detail="Forecast not found")

            db_config = DHRConfiguration(
                forecast_id=config.forecast_id,
                fourier_order=config.fourier_order,
                window_length=config.window_length,
                polyorder=config.polyorder,
                regularization_dhr=config.regularization_dhr,
                trend_components=config.trend_components
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
            config = db.query(DHRConfiguration).filter(DHRConfiguration.forecast_id == forecast_id).first()
            if not config:
                raise HTTPException(status_code=404, detail="DHR configuration not found")
            return config
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.put("/api/dhr-configurations/{forecast_id}")
    async def update_dhr_configuration(forecast_id: int, config: DHRConfigurationCreate, db: Session = Depends(get_db)):
        try:
            forecast = db.query(Forecast).filter(Forecast.id == forecast_id).first()
            if not forecast:
                raise HTTPException(status_code=404, detail="Forecast not found")
                
            existing_config = db.query(DHRConfiguration).filter(
                DHRConfiguration.forecast_id == forecast_id
            ).first()
            
            if not existing_config:
                raise HTTPException(status_code=404, detail="DHR configuration not found")
            
            existing_config.fourier_order = config.fourier_order
            existing_config.window_length = config.window_length
            existing_config.polyorder = config.polyorder
            existing_config.regularization_dhr = config.regularization_dhr
            existing_config.trend_components = config.trend_components
            existing_config.updated_at = datetime.utcnow()
            
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
    async def create_esn_configuration(config: ESNConfigurationCreate, db: Session = Depends(get_db)):
        try:
            logger.info(f"Creating ESN configuration: {config}")
            
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
            config = db.query(ESNConfiguration).filter(ESNConfiguration.forecast_id == forecast_id).first()
            if not config:
                raise HTTPException(status_code=404, detail="ESN configuration not found")
            return config
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.put("/api/esn-configurations/{forecast_id}")
    async def update_esn_configuration(forecast_id: int, config: ESNConfigurationCreate, db: Session = Depends(get_db)):
        try:
            forecast = db.query(Forecast).filter(Forecast.id == forecast_id).first()
            if not forecast:
                raise HTTPException(status_code=404, detail="Forecast not found")
                
            existing_config = db.query(ESNConfiguration).filter(
                ESNConfiguration.forecast_id == forecast_id
            ).first()
            
            if not existing_config:
                raise HTTPException(status_code=404, detail="ESN configuration not found")
            
            existing_config.reservoir_size = config.reservoir_size
            existing_config.spectral_radius = config.spectral_radius
            existing_config.sparsity = config.sparsity
            existing_config.input_scaling = config.input_scaling
            existing_config.dropout = config.dropout
            existing_config.lags = config.lags
            existing_config.regularization_esn = config.regularization_esn
            existing_config.updated_at = datetime.utcnow()
            
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
    async def create_hybrid_configuration(config: HybridConfigurationCreate, db: Session = Depends(get_db)):
        try:
            logger.info(f"Creating hybrid configuration for forecast_id: {config.forecast_id}")
            
            forecast = db.query(Forecast).filter(Forecast.id == config.forecast_id).first()
            if not forecast:
                logger.error(f"Forecast not found: {config.forecast_id}")
                raise HTTPException(status_code=404, detail=f"Forecast {config.forecast_id} not found")
                
            db_config = HybridConfiguration(
                forecast_id=config.forecast_id,
                fourier_order=config.fourier_order,
                window_length=config.window_length,
                polyorder=config.polyorder,
                regularization_dhr=config.regularization_dhr,
                trend_components=config.trend_components,
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
            
            return {
                "id": db_config.id,
                "message": "Hybrid configuration created successfully"
            }
        except HTTPException:
            raise
        except Exception as e:
            error_msg = f"Error in hybrid configuration endpoint: {str(e)}"
            logger.error(error_msg)
            db.rollback()
            raise HTTPException(status_code=500, detail=error_msg)

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

    @app.put("/api/hybrid-configurations/{forecast_id}")
    async def update_hybrid_configuration(forecast_id: int, config: HybridConfigurationCreate, db: Session = Depends(get_db)):
        try:
            forecast = db.query(Forecast).filter(Forecast.id == forecast_id).first()
            if not forecast:
                raise HTTPException(status_code=404, detail="Forecast not found")
                
            existing_config = db.query(HybridConfiguration).filter(
                HybridConfiguration.forecast_id == forecast_id
            ).first()
            
            if not existing_config:
                raise HTTPException(status_code=404, detail="Hybrid configuration not found")
            
            existing_config.fourier_order = config.fourier_order
            existing_config.window_length = config.window_length
            existing_config.polyorder = config.polyorder
            existing_config.regularization_dhr = config.regularization_dhr
            existing_config.trend_components = config.trend_components
            existing_config.reservoir_size = config.reservoir_size
            existing_config.spectral_radius = config.spectral_radius
            existing_config.sparsity = config.sparsity
            existing_config.input_scaling = config.input_scaling
            existing_config.dropout = config.dropout
            existing_config.lags = config.lags
            existing_config.regularization_esn = config.regularization_esn
            existing_config.updated_at = datetime.utcnow()
            
            db.commit()
            db.refresh(existing_config)
            
            return {
                "id": existing_config.id,
                "message": "Hybrid configuration updated successfully"
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating hybrid configuration: {str(e)}")
            db.rollback()
            raise HTTPException(status_code=500, detail=str(e))
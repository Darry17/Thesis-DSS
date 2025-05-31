from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List
from db import Base

class Forecast(Base):
    __tablename__ = "forecast"
    id = Column(Integer, primary_key=True, index=True)
    original_filename = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    forecast_type = Column(String, nullable=False)
    granularity = Column(String, nullable=False)
    steps = Column(Integer, nullable=False)
    model = Column(String, nullable=False)
    energy_demand = Column(Float, nullable=False)
    max_capacity = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    temp_id = Column(Integer, nullable=True)

class DHRForecast(Base):
    __tablename__ = "dhr_forecast"
    id = Column(Integer, primary_key=True, index=True)
    forecast_id = Column(Integer, nullable=False)
    fourier_terms = Column(Integer, nullable=False)
    reg_strength = Column(Float, nullable=False)
    ar_order = Column(Integer, nullable=False)
    window = Column(Integer, nullable=False)
    polyorder = Column(Integer, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

class ESNForecast(Base):
    __tablename__ = "esn_forecast"
    id = Column(Integer, primary_key=True, index=True)
    forecast_id = Column(Integer, nullable=False)
    N_res = Column(Integer, nullable=False)
    rho = Column(Float, nullable=False)
    sparsity = Column(Float, nullable=False)
    alpha = Column(Float, nullable=False)
    lambda_reg = Column(Float, nullable=False)
    lags = Column(Integer, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

class HybridForecast(Base):
    __tablename__ = "hybrid_forecast"
    id = Column(Integer, primary_key=True, index=True)
    forecast_id = Column(Integer, nullable=False)
    fourier_terms = Column(Integer, nullable=False)
    reg_strength = Column(Float, nullable=False)
    ar_order = Column(Integer, nullable=False)
    window = Column(Integer, nullable=False)
    polyorder = Column(Integer, nullable=False)
    N_res = Column(Integer, nullable=False)
    rho = Column(Float, nullable=False)
    sparsity = Column(Float, nullable=False)
    alpha = Column(Float, nullable=False)
    lambda_reg = Column(Float, nullable=False)
    lags = Column(Integer, nullable=False)
    n_features = Column(Integer, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    access_control = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=datetime.now)

class HistoryLog(Base):
    __tablename__ = "history_logs"
    id = Column(Integer, primary_key=True, index=True)
    forecast_id = Column(Integer, nullable=True)
    file_name = Column(String(255), nullable=False)
    forecast_type = Column(String(50), nullable=False)
    granularity = Column(String(50), nullable=False)
    steps = Column(Integer, nullable=False)
    model = Column(String(50), nullable=False)
    date = Column(DateTime, default=datetime.now)

class DeletedForecast(Base):
    __tablename__ = "deleted_forecasts"
    id = Column(Integer, primary_key=True, index=True)
    forecast_id = Column(Integer, nullable=True)
    file_name = Column(String(255), nullable=False)
    forecast_type = Column(String(50), nullable=False)
    granularity = Column(String(50), nullable=False)
    steps = Column(Integer, nullable=False)
    model = Column(String(50), nullable=False)
    date = Column(DateTime, default=datetime.now)
    deleted_by = Column(String(255), nullable=False)

class Temp(Base):
    __tablename__ = "temp"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.now)

# Pydantic Models

class DHRForecastCreate(BaseModel):
    forecast_id: int
    fourier_terms: int
    reg_strength: float
    ar_order: int
    window: int
    polyorder: int
    model_config = {'from_attributes': True}

class DHRForecastResponse(BaseModel):
    id: int
    forecast_id: int
    fourier_terms: int
    reg_strength: float
    ar_order: int
    window: int
    polyorder: int
    updated_at: datetime
    model_config = {'from_attributes': True}

class ESNForecastCreate(BaseModel):
    forecast_id: int
    N_res: int
    rho: float
    sparsity: float
    alpha: float
    lambda_reg: float
    lags: int
    model_config = {'from_attributes': True}

class ESNForecastResponse(BaseModel):
    id: int
    forecast_id: int
    N_res: int
    rho: float
    sparsity: float
    alpha: float
    lambda_reg: float
    lags: int
    model_config = {'from_attributes': True}

class HybridForecastCreate(BaseModel):
    forecast_id: int
    
    # DHR parameters
    fourier_terms: int
    reg_strength: float
    ar_order: int
    window: int
    polyorder: int
    
    # ESN parameters
    N_res: int
    rho: float
    sparsity: float
    alpha: float
    lambda_reg: float
    lags: int
    n_features: int
    model_config = {'from_attributes': True}

class HybridForecastResponse(BaseModel):
    id: int
    forecast_id: int
    
    # DHR parameters
    fourier_terms: int
    reg_strength: float
    ar_order: int
    window: int
    polyorder: int
    
    # ESN parameters
    N_res: int
    rho: float
    sparsity: float
    alpha: float
    lambda_reg: float
    lags: int
    n_features: int
    model_config = {'from_attributes': True}

class LoginRequest(BaseModel):
    username: str
    password: str

class UserCreate(BaseModel):
    username: str
    password: str
    access_control: str

class UserUpdate(BaseModel):
    username: str
    password: Optional[str] = None
    access_control: str

class UserResponse(BaseModel):
    id: int
    username: str
    access_control: str
    created_at: datetime
    class Config:
        from_attributes = True

# Pydantic Models
class HistoryLogCreate(BaseModel):
    file_name: str
    forecast_type: str
    granularity: str
    steps: int
    model: str
    action: Optional[str] = "Saved Forecast"
    forecast_id: Optional[int] = None
    model_config = {'from_attributes': True}

class HistoryLogResponse(BaseModel):
    id: int
    forecast_id: Optional[int] = None
    file_name: str
    forecast_type: str
    granularity: str
    steps: int
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
    granularity: str
    steps: int
    model: str
    date: datetime
    deleted_by: str
    model_config = {'from_attributes': True}

class PaginatedDeletedForecastResponse(BaseModel):
    logs: List[DeletedForecastResponse]
    total_pages: int
    model_config = {'from_attributes': True}
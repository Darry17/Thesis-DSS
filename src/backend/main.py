from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import create_engine, Column, Integer, String, desc, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Optional, List

import os
import re
import json
import logging
from dotenv import load_dotenv
import jwt  # For JWT token generation
from passlib.context import CryptContext  # For password hashing

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Enable CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
DATABASE_URL = f"mysql+mysqlconnector://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
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

class HistoryLog(Base):
    __tablename__ = "history_logs"
    id = Column(Integer, primary_key=True, index=True)
    forecast_id = Column(Integer, nullable=True)
    file_name = Column(String(255), nullable=False)
    model = Column(String(50), nullable=False)
    date = Column(DateTime, default=datetime.now)
    username = Column(String(255), nullable=True)

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
    original_filename = Column(String(255), nullable=True)
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

# New: User Model for Authentication
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, autoincrement=True)  # This is key
    username = Column(String(50), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    access_control = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=datetime.now)

# Create all tables in the database
Base.metadata.create_all(bind=engine)

# New: Password Hashing Setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# New: JWT Configuration
SECRET_KEY = os.getenv('SECRET_KEY')  # Add to .env file
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# New: Helper Function to Verify Password
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# New: Helper Function to Create JWT Token
def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Data model mapping
DATA_MODELS = {
    "hourly": HourlyData,
    "daily": DailyData,
    "weekly": WeeklyData,
    "json": JsonData
}

# Pydantic Models
class JsonDataResponse(BaseModel):
    id: int
    filename: str
    original_filename: Optional[str] = None
    upload_date: datetime
    model_config = {'from_attributes': True}

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
    username: str  # Already present, just confirming
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

class ForecastCreate(BaseModel):
    filename: str
    original_filename: Optional[str] = None
    forecast_model: str
    steps: str
    granularity: str
    model_config = {'from_attributes': True}

class DHRConfigurationCreate(BaseModel):
    forecast_id: int
    fourier_order: int
    window_length: int
    seasonality_periods: str
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
    model_config = {'from_attributes': True}

# New: Pydantic Model for Login Request
class LoginRequest(BaseModel):
    username: str
    password: str

# Pydantic models for user requests
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

# Database session dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Helper functions
def get_file_path(filename: str) -> str:
    for folder in FOLDERS:
        file_path = os.path.join(BASE_STORAGE_PATH, folder, filename)
        if os.path.exists(file_path):
            return file_path
    return None

# API Routes
@app.get("/")
async def root():
    return {"message": "API is running"}

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

# New: Login Endpoint
@app.post("/login")
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate a user and return a JWT token."""
    user = db.query(User).filter(User.username == request.username).first()
    if user and verify_password(request.password, user.password):
        access_token = create_access_token(data={
            "sub": user.username,
            "access_control": user.access_control
        })
        return {"access_token": access_token, "token_type": "bearer"}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

# Pydantic models for user requests
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

# Dependency to get current user from token
def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        access_control: str = payload.get("access_control")
        if username is None or access_control is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"username": username, "access_control": access_control}
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Dependency to ensure admin access
def get_admin_user(current_user: dict = Depends(get_current_user)):
    if current_user["access_control"] != "ADMIN":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user

@app.get("/api/validate-token")
async def validate_token(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user = db.query(User).filter(User.username == payload.get("sub")).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return {
            "id": user.id,
            "username": user.username,
            "access_control": user.access_control
        }
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

@app.get("/users", response_model=List[UserResponse])
async def read_users(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)  # Add authentication
):
    users = db.query(User).all()
    if not users:
        raise HTTPException(status_code=404, detail="No users found")
    return users

# Create a new user (admin-only)
@app.post("/users", response_model=UserResponse)
async def create_user(
    user: UserCreate, db: Session = Depends(get_db), current_user: dict = Depends(get_admin_user)
):
    # Check if username already exists
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(status_code=400, detail="Username already exists")
    hashed_password = pwd_context.hash(user.password)
    db_user = User(
        username=user.username,
        password=hashed_password,
        access_control=user.access_control,
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# Update a user (admin-only)
@app.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user: UserUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_admin_user),
):
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    # Check for username conflict
    if (
        user.username != db_user.username
        and db.query(User).filter(User.username == user.username).first()
    ):
        raise HTTPException(status_code=400, detail="Username already exists")
    db_user.username = user.username
    db_user.access_control = user.access_control
    if user.password:  # Update password only if provided
        db_user.password = pwd_context.hash(user.password)
    db.commit()
    db.refresh(db_user)
    return db_user

# Delete a user (admin-only)
@app.delete("/users/{user_id}")
async def delete_user(
    user_id: int, db: Session = Depends(get_db), current_user: dict = Depends(get_admin_user)
):
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(db_user)
    db.commit()
    return {"message": "User deleted successfully"}

@app.post("/storage/process_model_data/")
async def process_model_data(
    file: UploadFile = File(...), 
    original_filename: str = Form(None),
    db: Session = Depends(get_db)
):
    try:
        filename = file.filename
        logger.info(f"Processing file: {filename}, Original file: {original_filename}")
        
        # Determine subfolder and data model
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
        
        # Save file to storage
        target_folder = os.path.join(BASE_STORAGE_PATH, subfolder)
        os.makedirs(target_folder, exist_ok=True)
        file_path = os.path.join(target_folder, filename)
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        # Save to database with original filename
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

        # Store the original CSV filename in the database
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
            "original_filename": latest_file.original_filename,
            "upload_date": latest_file.upload_date
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/forecasts")
async def create_forecast(forecast: ForecastCreate, db: Session = Depends(get_db)):
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
        # Check if forecast exists
        forecast = db.query(Forecast).filter(Forecast.id == forecast_id).first()
        if not forecast:
            raise HTTPException(status_code=404, detail="Forecast not found")
            
        # Find existing configuration
        existing_config = db.query(DHRConfiguration).filter(
            DHRConfiguration.forecast_id == forecast_id
        ).first()
        
        if not existing_config:
            raise HTTPException(status_code=404, detail="DHR configuration not found")
        
        # Update fields
        existing_config.fourier_order = config.fourier_order
        existing_config.window_length = config.window_length
        existing_config.seasonality_periods = config.seasonality_periods
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
        # Check if forecast exists
        forecast = db.query(Forecast).filter(Forecast.id == forecast_id).first()
        if not forecast:
            raise HTTPException(status_code=404, detail="Forecast not found")
            
        # Find existing configuration
        existing_config = db.query(ESNConfiguration).filter(
            ESNConfiguration.forecast_id == forecast_id
        ).first()
        
        if not existing_config:
            raise HTTPException(status_code=404, detail="ESN configuration not found")
        
        # Update fields
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
        
        # Check if forecast exists
        forecast = db.query(Forecast).filter(Forecast.id == config.forecast_id).first()
        if not forecast:
            logger.error(f"Forecast not found: {config.forecast_id}")
            raise HTTPException(status_code=404, detail=f"Forecast {config.forecast_id} not found")
            
        # Create configuration object
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

@app.put("/api/hybrid-configurations/{forecast_id}")
async def update_hybrid_configuration(forecast_id: int, config: HybridConfigurationCreate, db: Session = Depends(get_db)):
    try:
        # Check if forecast exists
        forecast = db.query(Forecast).filter(Forecast.id == forecast_id).first()
        if not forecast:
            raise HTTPException(status_code=404, detail="Forecast not found")
            
        # Find existing configuration
        existing_config = db.query(HybridConfiguration).filter(
            HybridConfiguration.forecast_id == forecast_id
        ).first()
        
        if not existing_config:
            raise HTTPException(status_code=404, detail="Hybrid configuration not found")
        
        # Update all fields
        existing_config.fourier_order = config.fourier_order
        existing_config.window_length = config.window_length
        existing_config.seasonality_periods = config.seasonality_periods
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

@app.post("/api/history-logs")
async def create_history_log(
    log: HistoryLogCreate, 
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)  # Add authentication dependency
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
        
        # Create the log entry including the username from the authenticated user
        db_log = HistoryLog(
            file_name=log.file_name,
            model=log.model,
            forecast_id=forecast_id,
            username=current_user["username"]  # Get username from token
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
            "date": db_log.date,
            "username": db_log.username  # Include username in response
        }

    except Exception as e:
        logger.error(f"Error creating history log: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to create history log: {str(e)}"
        )

    except Exception as e:
        logger.error(f"Error creating history log: {str(e)}")
        db.rollback()
        # Return a detailed error message
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
            "date": log.date,
            "username": log.username  # Add username to response
        }
    except Exception as e:
        logger.error(f"Error fetching history log {log_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history-logs", response_model=list[HistoryLogResponse])
async def get_all_history_logs(db: Session = Depends(get_db)):
    try:
        logs = db.query(HistoryLog).order_by(desc(HistoryLog.date)).all()
        return logs
    except Exception as e:
        logger.error(f"Error fetching history logs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
from fastapi import APIRouter, HTTPException
from ..models.dhr_solar_forecast import load_and_prepare_data, main as run_forecast
import pandas as pd
from typing import Dict
import json
import os

router = APIRouter()

@router.post("/generate_forecast")
async def generate_forecast(data: Dict):
    try:
        filename = data.get("filename")
        if not filename:
            raise HTTPException(status_code=400, detail="Filename is required")

        # Load the JSON data
        json_path = os.path.join("storage", "uploads", filename)
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        # Convert JSON to DataFrame
        df = pd.DataFrame(json_data)
        
        # Set time column as index
        time_columns = ["time", "date", "week"]
        time_col = next((col for col in time_columns if col in df.columns), None)
        if time_col:
            df.set_index(time_col, inplace=True)
            df.index = pd.to_datetime(df.index)

        # Ensure column names match the model requirements
        column_mapping = {
            'solar_power': 'solar_power',
            'dhi': 'DHI',
            'dni': 'DNI',
            'ghi': 'GHI',
            'solar_zenith_angle': 'Solar Zenith Angle'
        }
        df.rename(columns=column_mapping, inplace=True)

        # Run the forecast
        forecast_results = run_forecast(df)
        
        # Save forecast results
        forecast_path = os.path.join("storage", "forecasts", f"forecast_{filename}")
        with open(forecast_path, 'w') as f:
            json.dump(forecast_results, f)

        return {
            "status": "success",
            "message": "Forecast generated successfully",
            "forecast_file": f"forecast_{filename}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 
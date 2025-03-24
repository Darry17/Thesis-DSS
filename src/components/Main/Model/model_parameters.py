import json
import os
import numpy as np
from datetime import datetime

def load_json_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def get_granularity_from_filename(filename):
    if 'hourly' in filename:
        return 'hourly'
    elif 'daily' in filename:
        return 'daily'
    elif 'weekly' in filename:
        return 'weekly'
    return None

def get_data_type_from_filename(filename):
    if 'solar' in filename:
        return 'solar'
    elif 'wind' in filename:
        return 'wind'
    return None

def calculate_model_parameters(data, granularity, data_type):
    # Extract values from data
    values = np.array([float(entry['value']) for entry in data])
    
    # Calculate parameters based on data characteristics
    params = {
        # DHR Parameters
        'fourierOrder': calculate_fourier_order(granularity),
        'windowLength': calculate_window_length(granularity),
        'seasonalityPeriods': calculate_seasonality_periods(granularity),
        'polyorder': 2,  # Default polynomial order
        'regularizationDHR': 1e-4,  # Default DHR regularization
        'trendComponents': calculate_trend_components(granularity),
        
        # ESN Parameters
        'reservoirSize': calculate_reservoir_size(len(values), granularity),
        'spectralRadius': 0.9,  # Default spectral radius
        'sparsity': 0.1,  # Default sparsity
        'inputScaling': 0.5,  # Default input scaling
        'dropout': 0.1,  # Default dropout rate
        'lags': calculate_lags(granularity),
        'regularizationESN': 0.1  # Default ESN regularization
    }
    
    return params

def calculate_fourier_order(granularity):
    # Fourier order based on granularity
    orders = {
        'hourly': 24,    # Daily seasonality
        'daily': 7,      # Weekly seasonality
        'weekly': 52     # Yearly seasonality
    }
    return orders.get(granularity, 12)

def calculate_window_length(granularity):
    # Window length based on granularity
    lengths = {
        'hourly': 168,   # One week of hours
        'daily': 30,     # One month of days
        'weekly': 52     # One year of weeks
    }
    return lengths.get(granularity, 24)

def calculate_seasonality_periods(granularity):
    # Seasonality periods based on granularity
    periods = {
        'hourly': "24,168",    # Daily and weekly seasonality
        'daily': "7,30",       # Weekly and monthly seasonality
        'weekly': "52"         # Yearly seasonality
    }
    return periods.get(granularity, "24")

def calculate_trend_components(granularity):
    # Trend components based on granularity
    components = {
        'hourly': 24,
        'daily': 7,
        'weekly': 4
    }
    return components.get(granularity, 12)

def calculate_reservoir_size(data_length, granularity):
    # Reservoir size based on data length and granularity
    base_size = {
        'hourly': 500,
        'daily': 200,
        'weekly': 100
    }
    return base_size.get(granularity, 200)

def calculate_lags(granularity):
    # Lags based on granularity
    lags = {
        'hourly': 24,    # One day
        'daily': 7,      # One week
        'weekly': 4      # One month
    }
    return lags.get(granularity, 12)

def suggest_parameters(file_path):
    # Load and analyze data
    data = load_json_data(file_path)
    filename = os.path.basename(file_path)
    granularity = get_granularity_from_filename(filename)
    data_type = get_data_type_from_filename(filename)
    
    # Calculate parameters
    params = calculate_model_parameters(data, granularity, data_type)
    
    return {
        'filename': filename,
        'granularity': granularity,
        'data_type': data_type,
        'parameters': params
    }

def main():
    # Base directory containing the data folders
    base_dir = "storage"
    
    # Process a sample file from each granularity
    for granularity in ['hourly', 'daily', 'weekly']:
        granularity_dir = os.path.join(base_dir, granularity)
        if os.path.exists(granularity_dir):
            # Get the most recent file in the directory
            files = [f for f in os.listdir(granularity_dir) if f.endswith('.json')]
            if files:
                latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(granularity_dir, x)))
                file_path = os.path.join(granularity_dir, latest_file)
                
                # Get parameter suggestions
                suggestions = suggest_parameters(file_path)
                
                print(f"\nParameters for {suggestions['granularity']} {suggestions['data_type']} data:")
                print(json.dumps(suggestions['parameters'], indent=2))

if __name__ == "__main__":
    main()

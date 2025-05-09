import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter
from datetime import timedelta
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_prepare_data(df):
    """Load and preprocess the dataset from the uploaded DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame from user's uploaded JSON
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame with required columns
    """
    try:
        # Ensure DataFrame has a datetime index
        if 'time' in df.columns:
            df = df.set_index('time')
            df.index = pd.to_datetime(df.index)
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a 'time' column or datetime index")

        # Resample to hourly frequency if needed
        df = df.resample('h').mean().interpolate()

        # Required columns for solar forecasting (case-insensitive)
        required_columns = {
            'solar_power': ['solar_power'],
            'ghi': ['ghi'],
            'dni': ['dni'],
            'dhi': ['dhi'],
            'solar_zenith_angle': ['solar_zenith_angle']
        }

        # Create a mapping of lowercase column names
        df_cols_lower = {col.lower(): col for col in df.columns}
        
        # Check and rename columns if needed
        processed_df = df.copy()
        for standard_name, possible_names in required_columns.items():
            # Try to find a matching column
            found = False
            for possible_name in possible_names:
                if possible_name.lower() in df_cols_lower:
                    if df_cols_lower[possible_name.lower()] != standard_name:
                        processed_df = processed_df.rename(
                            columns={df_cols_lower[possible_name.lower()]: standard_name}
                        )
                    found = True
                    break
            
            if not found:
                raise ValueError(f"Missing required column: {standard_name}")

        # Select only the required columns
        processed_df = processed_df[list(required_columns.keys())]

        # Validate data ranges
        if (processed_df['solar_power'] < 0).any():
            logger.warning("Negative solar power values found, setting to 0")
            processed_df.loc[processed_df['solar_power'] < 0, 'solar_power'] = 0

        if (processed_df['ghi'] < 0).any():
            logger.warning("Negative GHI values found, setting to 0")
            processed_df.loc[processed_df['ghi'] < 0, 'ghi'] = 0

        # Log data summary
        logger.info(f"Data loaded successfully. Time range: {processed_df.index.min()} to {processed_df.index.max()}")
        logger.info(f"Number of records: {len(processed_df)}")

        return processed_df

    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        raise

def get_periods_for_granularity(granularity, steps):
    """Determine appropriate Fourier periods based on granularity and forecast steps.
    
    Args:
        granularity (str): "Hourly", "Daily", or "Weekly"
        steps (str): Step option selected (e.g., "1-hour", "24-hour", "7-day", etc.)
    
    Returns:
        list: List of periods to use for Fourier transforms
    """
    periods = []
    
    if granularity == "Hourly":
        periods.append(24)  # Daily seasonality
        if "168" in steps:  # Week horizon
            periods.append(168)  # Weekly seasonality
    
    elif granularity == "Daily":
        periods.append(7)   # Weekly seasonality
        if "30" in steps:   # Month horizon
            periods.append(30)  # Monthly seasonality
    
    elif granularity == "Weekly":
        periods.append(4)   # Monthly seasonality
        if "52" in steps:   # Year horizon
            periods.append(52)  # Yearly seasonality
    
    return periods

def fourier_transform(t, n_harmonics, periods):
    t = np.array(t)
    X = []

    for period in periods:
        for k in range(1, n_harmonics + 1):
            X.append(np.sin(2 * np.pi * k * t / period))
            X.append(np.cos(2 * np.pi * k * t / period))

    return np.vstack(X).T

def create_features(df, target_col, fourier_terms, ar_order, window, polyorder, granularity="Hourly", steps="24-hour"):
    """Create features including autoregressive, smoothed, Fourier, and exogenous variables."""
    try:
        t = np.arange(len(df))
        
        # Get appropriate periods based on granularity and steps
        periods = get_periods_for_granularity(granularity, steps)
        
        fourier = fourier_transform(t, n_harmonics=fourier_terms, periods=periods)
        target = df[target_col].values
        ghi = df['ghi'].values
        dni = df['dni'].values
        dhi = df['dhi'].values
        sza = df['solar_zenith_angle'].values

        X, y = [], []
        for i in range(max(ar_order, window), len(target)):
            ar_features = target[i - ar_order:i]
            smoothed = savgol_filter(target[i - window:i], window_length=window, polyorder=polyorder)
            exog_features = np.array([ghi[i], dni[i], dhi[i], sza[i]])
            features = np.concatenate([ar_features, smoothed, fourier[i], exog_features])
            X.append(features)
            y.append(target[i])
        return np.array(X), np.array(y)
    except Exception as e:
        logger.error(f"Error creating features: {str(e)}")
        raise

def repeat_last_week(arr, forecast_horizon):
    """Repeat last week's pattern for exogenous variables."""
    period = 24 * 7  # one week
    repeated = np.tile(arr[-period:], int(np.ceil(forecast_horizon / period)))[:forecast_horizon]
    return np.concatenate([arr, repeated])

def create_feature_vector(values, current_pos, fourier_data, ar_order, window, polyorder, ghi, dni, dhi, sza):
    """Create a single feature vector for forecasting."""
    try:
        ar_features = values[current_pos - ar_order:current_pos]
        segment = values[current_pos - window:current_pos]
        window_len = len(segment) if len(segment) % 2 != 0 else len(segment) - 1
        window_len = max(3, window_len)
        smoothed = savgol_filter(segment, window_length=window_len, polyorder=min(polyorder, window_len - 1))
        exog_features = np.array([ghi[current_pos], dni[current_pos], dhi[current_pos], sza[current_pos]])
        return np.concatenate([ar_features, smoothed, fourier_data[current_pos], exog_features])
    except Exception as e:
        logger.error(f"Error creating feature vector: {str(e)}")
        raise

def generate_forecast(model, start_values, fourier_data, steps, params, ghi, dni, dhi, sza):
    """Generate forecast for the specified number of steps."""
    try:
        ar_order = params['ar_order']
        window = params['window']
        polyorder = params['polyorder']
        values = start_values.copy()
        forecasts = []
        current_pos = len(values)
        for _ in range(steps):
            features = create_feature_vector(
                values, current_pos, fourier_data, ar_order, window, polyorder,
                ghi, dni, dhi, sza
            )
            prediction = model.predict(features.reshape(1, -1))[0]
            prediction = max(0, prediction)
            if sza[current_pos] > 90:  # Night condition
                prediction = 0.0
            forecasts.append(prediction)
            values = np.append(values, prediction)
            current_pos += 1
        return np.array(forecasts)
    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}")
        raise

def main(input_df=None, granularity="Hourly", steps="24-hour"):
    """Main function to run the forecasting pipeline."""
    try:
        if input_df is None:
            raise ValueError("Input DataFrame is required")

        # Parameters optimized for hourly solar forecasting
        params = {
            'forecast_id': None,  # Will be set by the API
            'fourier_terms': 3,
            'reg_strength': 0.0001000100524,
            'ar_order': 3,
            'window': 23,
            'polyorder': 3,
            'granularity': granularity,
            'steps': steps
        }

        # Prepare the data - select and validate required columns
        df = load_and_prepare_data(input_df)

        # Ensure minimum data requirements based on granularity
        min_periods = {
            "Hourly": 24 * 7,    # 1 week of hourly data
            "Daily": 7 * 4,      # 4 weeks of daily data
            "Weekly": 4 * 12     # 12 months of weekly data
        }
        min_required = min_periods.get(granularity, 24 * 7)
        
        if len(df) < min_required:
            raise ValueError(f"Insufficient data. Need at least {min_required} periods for {granularity} granularity.")

        # Create features with appropriate periods based on granularity and steps
        X, y = create_features(
            df, 'solar_power', params['fourier_terms'], params['ar_order'],
            params['window'], params['polyorder'], granularity=granularity, steps=steps
        )

        # Train model with time-based split
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = Ridge(alpha=params['reg_strength'])
        model.fit(X_train, y_train)

        # Prepare for forecasting
        last_two_weeks = min(336, len(df))  # Use all data if less than 2 weeks
        target_values = df['solar_power'].values
        target_values_ext = np.concatenate([target_values, np.zeros(168)])  # pre-allocate for horizon
        historical_end_idx = len(target_values)
        historical_start_idx = historical_end_idx - last_two_weeks
        historical_data = target_values[historical_start_idx:historical_end_idx].tolist()
        historical_dates = df.index[historical_start_idx:historical_end_idx].strftime('%Y-%m-%d %H:%M:%S').tolist()

        # Prepare exogenous variables
        ghi = df['ghi'].values
        dni = df['dni'].values
        dhi = df['dhi'].values
        sza = df['solar_zenith_angle'].values

        # Extend variables for forecasting
        ghi_ext = repeat_last_week(ghi, 168)
        dni_ext = repeat_last_week(dni, 168)
        dhi_ext = repeat_last_week(dhi, 168)
        sza_ext = repeat_last_week(sza, 168)

        # Create extended Fourier terms
        extended_df_length = len(df) + 168
        t_extended = np.arange(extended_df_length)
        periods = get_periods_for_granularity(granularity, steps)
        fourier_extended = fourier_transform(t_extended, n_harmonics=params['fourier_terms'], periods=periods)

        # Generate forecasts
        forecast_horizons = [1, 24, 168]  # 1 hour, 1 day, 1 week ahead
        forecasts = {}
        for horizon in forecast_horizons:
            logger.info(f"Generating {horizon}-step ahead forecast...")
            forecast_values = generate_forecast(
                model, target_values_ext, fourier_extended, horizon, params,
                ghi_ext, dni_ext, dhi_ext, sza_ext
            )
            
            forecast_dates = [(df.index[-1] + pd.Timedelta(hours=i + 1)).strftime('%Y-%m-%d %H:%M:%S') 
                            for i in range(horizon)]
            
            forecasts[str(horizon)] = {
                'values': forecast_values,
                'dates': forecast_dates,
                'horizon_type': '1 hour' if horizon == 1 else ('1 day' if horizon == 24 else '1 week')
            }

        # Prepare results dictionary with metadata
        results = {
            'historical': {
                'dates': historical_dates,
                'values': historical_data,
                'start_date': df.index[0].strftime('%Y-%m-%d %H:%M:%S'),
                'end_date': df.index[-1].strftime('%Y-%m-%d %H:%M:%S')
            },
            'forecasts': forecasts,
            'parameters': params,
            'metadata': {
                'total_periods': len(df),
                'forecast_generated_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_frequency': granularity.lower(),
                'configuration': {
                    'fourier_terms': params['fourier_terms'],
                    'reg_strength': params['reg_strength'],
                    'ar_order': params['ar_order'],
                    'window': params['window'],
                    'polyorder': params['polyorder'],
                    'granularity': granularity,
                    'steps': steps
                }
            }
        }

        return results

    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    # Step 1: Load your JSON data into a pandas DataFrame
    input_df = pd.read_json('data.json')  # Replace with your actual file path
    
    # Step 2: Run the forecasting pipeline
    results = main(input_df=input_df)
    
    # Step 3: Save results to file if running as script
    with open('forecast_results.json', 'w') as f:
        json.dump(results, f, indent=2)
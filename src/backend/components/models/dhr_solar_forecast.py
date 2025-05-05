import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter
from dotenv import load_dotenv
from datetime import timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load .env file and dataset path
load_dotenv()
DATA_PATH = os.getenv("ZONE_1_PATH")

def load_and_prepare_data():
    """Load and preprocess the dataset."""
    try:
        df = pd.read_csv(DATA_PATH, parse_dates=['time'], index_col='time')
        df = df.resample('h').mean().interpolate()  # Resample to hourly frequency
        required_columns = ['solar_power', 'GHI', 'DNI', 'DHI', 'Solar Zenith Angle']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("Dataset missing required columns")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def fourier_transform(t, n_harmonics=4, periods=[24, 168]):
    """Generate Fourier terms for daily and weekly periodicity."""
    features = []
    for period in periods:
        sin_terms = [np.sin(2 * np.pi * i * t / period) for i in range(1, n_harmonics + 1)]
        cos_terms = [np.cos(2 * np.pi * i * t / period) for i in range(1, n_harmonics + 1)]
        features.extend(sin_terms + cos_terms)
    return np.column_stack(features)

def create_features(df, target_col, fourier_terms, ar_order, window, polyorder):
    """Create features including autoregressive, smoothed, Fourier, and exogenous variables."""
    try:
        t = np.arange(len(df))
        fourier = fourier_transform(t, n_harmonics=fourier_terms, periods=[24, 168])
        target = df[target_col].values
        ghi = df['GHI'].values
        dni = df['DNI'].values
        dhi = df['DHI'].values
        sza = df['Solar Zenith Angle'].values

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

def main():
    """Main function to run the forecasting pipeline."""
    try:
        # Parameters
        params = {
            'fourier_terms': 3,
            'reg_strength': 0.0001000100524,
            'ar_order': 3,
            'window': 23,
            'polyorder': 3
        }

        # Load and prepare data
        df = load_and_prepare_data()

        # Create features
        X, y = create_features(
            df, 'solar_power', params['fourier_terms'], params['ar_order'],
            params['window'], params['polyorder']
        )

        # Train model
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = Ridge(alpha=params['reg_strength'])
        model.fit(X_train, y_train)

        # Prepare for forecasting
        last_two_weeks = 336
        forecast_horizons = [1, 24, 168]
        target_values = df['solar_power'].values
        historical_end_idx = len(target_values)
        historical_start_idx = historical_end_idx - last_two_weeks
        historical_data = target_values[historical_start_idx:historical_end_idx]
        historical_dates = df.index[historical_start_idx:historical_end_idx]

        # Prepare exogenous variables
        ghi = df['GHI'].values
        dni = df['DNI'].values
        dhi = df['DHI'].values
        sza = df['Solar Zenith Angle'].values
        max_horizon = max(forecast_horizons)
        ghi_ext = repeat_last_week(ghi, max_horizon)
        dni_ext = repeat_last_week(dni, max_horizon)
        dhi_ext = repeat_last_week(dhi, max_horizon)
        sza_ext = repeat_last_week(sza, max_horizon)

        # Create extended Fourier terms
        extended_df_length = len(df) + max_horizon
        t_extended = np.arange(extended_df_length)
        fourier_extended = fourier_transform(t_extended, n_harmonics=params['fourier_terms'], periods=[24, 168])

        # Generate and save forecasts
        forecasts = {}
        for horizon in forecast_horizons:
            logger.info(f"Generating {horizon}-step ahead forecast...")
            forecast_values = generate_forecast(
                model, target_values, fourier_extended, horizon, params,
                ghi_ext, dni_ext, dhi_ext, sza_ext
            )
            forecast_dates = [historical_dates[-1] + timedelta(hours=i + 1) for i in range(horizon)]
            forecasts[horizon] = {
                'values': forecast_values,
                'dates': forecast_dates
            }

            # Save forecast to CSV
            forecast_df = pd.DataFrame({
                'datetime': forecast_dates,
                'forecasted_solar_power': forecast_values
            })
            forecast_df.to_csv(f'dhr_solar_power_forecast_{horizon}h.csv', index=False)
            logger.info(f"Forecast saved to 'dhr_solar_power_forecast_{horizon}h.csv'")

            # Plot
            plt.figure(figsize=(14, 6))
            plt.plot(historical_dates, historical_data, label='Actual (Last 2 weeks)', color='blue', linewidth=2)
            plt.plot(forecast_dates, forecast_values, label=f'Forecast ({horizon} step{"s" if horizon > 1 else ""})', color='red', linestyle='--', linewidth=2)
            plt.axvline(x=historical_dates[-1], color='green', linestyle=':', label='Forecast Start', linewidth=2)
            plt.title(f'Solar Power Forecast - {horizon} Step{"s" if horizon > 1 else ""}')
            plt.xlabel('Time')
            plt.ylabel('Solar Power')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'dhr_solar_power_forecast_{horizon}h.png', dpi=300, bbox_inches='tight')
            plt.close()

    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
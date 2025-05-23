import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.signal import savgol_filter
from dotenv import load_dotenv
from datetime import timedelta, datetime

# --- Fourier Transform ---
def fourier_transform(t, n_harmonics=4, periods=[7, 30, 365]):
    features = []
    for period in periods:
        sin_terms = [np.sin(2 * np.pi * i * t / period) for i in range(1, n_harmonics + 1)]
        cos_terms = [np.cos(2 * np.pi * i * t / period) for i in range(1, n_harmonics + 1)]
        features.extend(sin_terms + cos_terms)
    return np.column_stack(features)

# --- Feature Engineering ---
def create_features(df, target_col, fourier_terms, ar_order, window, polyorder):
    t = np.arange(len(df))
    fourier = fourier_transform(t, n_harmonics=fourier_terms, periods=[7, 30, 365])
    target = df[target_col].values
    wind_speed = df['Wind Speed'].values
    temperature = df['Temperature'].values
    relative_humidity = df['Relative Humidity'].values

    # Normalize features
    scaler_target = MinMaxScaler()
    scaler_exog = MinMaxScaler()
    target_scaled = scaler_target.fit_transform(target.reshape(-1, 1)).flatten()
    exog_scaled = scaler_exog.fit_transform(np.column_stack([wind_speed, temperature, relative_humidity]))

    X, y = [], []
    for i in range(max(ar_order, window), len(target)):
        ar_features = target_scaled[i - ar_order:i]
        smoothed = savgol_filter(target_scaled[i - window:i], window_length=window, polyorder=polyorder)
        exog_features = exog_scaled[i]
        features = np.concatenate([ar_features, smoothed, fourier[i], exog_features])
        X.append(features)
        y.append(target_scaled[i])
    
    return np.array(X), np.array(y), scaler_target, scaler_exog

# --- Extend Data ---
def repeat_last_month(arr, forecast_horizon):
    period = 30  # approximately a month in days
    n_features = arr.shape[1] if len(arr.shape) > 1 else 1
    if n_features > 1:
        result = []
        for col in range(n_features):
            data = arr[:, col]
            start_idx = max(0, len(data) - period)
            last_month = data[start_idx:]
            repeated = np.tile(last_month, (forecast_horizon // len(last_month) + 1))[:forecast_horizon]
            result.append(repeated)
        return np.column_stack(result)
    else:
        start_idx = max(0, len(arr) - period)
        last_month = arr[start_idx:]
        repeated = np.tile(last_month, (forecast_horizon // len(last_month) + 1))[:forecast_horizon]
        return repeated

# --- Create Feature Vector ---
def create_feature_vector(values, current_pos, fourier_data, ar_order, window, polyorder, exog_features, scaler_target):
    ar_features = values[current_pos - ar_order:current_pos]
    segment = values[current_pos - window:current_pos]
    if len(segment) < window:
        window_len = len(segment) if len(segment) % 2 != 0 else len(segment) - 1
        window_len = max(3, window_len)
    else:
        window_len = window
    smoothed = savgol_filter(segment, window_length=window_len, polyorder=min(polyorder, window_len - 1))
    # Use the passed exog_features directly instead of indexing
    return np.concatenate([ar_features, smoothed, fourier_data[current_pos], exog_features])

# --- Forecast Generation ---
def generate_forecast(model, start_values, fourier_data, steps, params, exog_forecast_scaled, scaler_target):
    ar_order = params['ar_order']
    window = params['window']
    polyorder = params['polyorder']
    values = start_values.copy()
    forecasts = []
    historical_len = len(start_values)
    max_target = np.max(scaler_target.inverse_transform(start_values.reshape(-1, 1)))  # For clipping
    
    for step in range(steps):
        current_pos = historical_len + step
        fourier_idx = historical_len + step
        
        # Use step index for exogenous features instead of current_pos
        exog_features = exog_forecast_scaled[step]
        
        features = create_feature_vector(
            values, current_pos, fourier_data, ar_order, window, polyorder,
            exog_features, scaler_target
        )
        prediction = model.predict(features.reshape(1, -1))[0]
        prediction = np.clip(prediction, 0, 1)  # Scaled range [0, 1]
        forecasts.append(prediction)
        values = np.append(values, prediction)
    
    # Inverse transform predictions
    forecasts = np.array(forecasts).reshape(-1, 1)
    forecasts_unscaled = scaler_target.inverse_transform(forecasts)
    forecasts_unscaled = np.clip(forecasts_unscaled, 0, max_target)
    return forecasts_unscaled.flatten()

# --- Main Forecast Function ---
def run_forecast(csv_path, steps, output_dir="forecasts", forecast_type="daily", params=None):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    
    df = pd.read_csv(csv_path, parse_dates=['time'], index_col='time')
    df = df.resample('D').mean().interpolate()  # Resample to daily frequency
    
    # Use provided params or default if missing
    if params is None:
        raise ValueError("Parameters must be provided for the forecast.")
    
    # Create features for the model
    X, y, scaler_target, scaler_exog = create_features(df, target_col='wind_power',
                                                      fourier_terms=params['fourier_terms'],
                                                      ar_order=params['ar_order'],
                                                      window=params['window'],
                                                      polyorder=params['polyorder'])
    
    # Train model
    model = Ridge(alpha=params['reg_strength'])
    model.fit(X, y)
    
    # Historical data for plotting
    last_month = 30  # Last month of data
    target_values = df['wind_power'].values
    historical_end_idx = len(target_values)
    historical_start_idx = max(0, historical_end_idx - last_month)
    historical_data = target_values[historical_start_idx:historical_end_idx]
    historical_dates = df.index[historical_start_idx:historical_end_idx]
    
    # Prepare exogenous variables and extend for forecast horizon
    wind_speed = df['Wind Speed'].values
    temperature = df['Temperature'].values
    relative_humidity = df['Relative Humidity'].values
    exog_data = np.column_stack([wind_speed, temperature, relative_humidity])
    
    # Extend exogenous variables for forecast horizon
    exog_forecast = repeat_last_month(exog_data, steps)
    exog_forecast_scaled = scaler_exog.transform(exog_forecast)
    
    # Scale historical target for forecasting
    target_scaled = scaler_target.transform(target_values.reshape(-1, 1)).flatten()
    
    # Extend Fourier features
    extended_df_length = len(df) + steps
    t_extended = np.arange(extended_df_length)
    fourier_extended = fourier_transform(t_extended, n_harmonics=params['fourier_terms'], periods=[7, 30, 365])
    
    # Generate forecast
    forecast_values = generate_forecast(
        model,
        target_scaled,
        fourier_extended,
        steps,
        params,
        exog_forecast_scaled,
        scaler_target
    )
    
    # Generate forecast dates
    last_historical_date = historical_dates[-1]
    forecast_dates = pd.date_range(start=last_historical_date, periods=steps+1, freq='D')[1:]
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame({
        'datetime': forecast_dates,
        'forecasted_wind_power': forecast_values
    })
    
    # Save forecast as CSV
    csv_name = f"wind_dhr_daily_{timestamp}_{steps}.csv"
    csv_path_out = os.path.join(output_dir, csv_name)
    forecast_df.to_csv(csv_path_out, index=False)
    print(f"Forecast saved to '{csv_path_out}'")
    
    # Plotting
    plt.figure(figsize=(14, 6))
    plt.plot(historical_dates, historical_data, label=f'Actual (Last {len(historical_data)} days)', color='blue', linewidth=2)
    plt.plot(forecast_dates, forecast_values, label=f'Forecast ({steps}d)', color='red', linestyle='--', linewidth=2)
    
    # For visual continuity, create a connection point between historical data and forecast
    plt.plot([historical_dates[-1], forecast_dates[0]], 
             [historical_data[-1], forecast_values[0]], 
             color='red', linestyle='--', linewidth=2)
             
    plt.axvline(x=historical_dates[-1], color='green', linestyle=':', label='Forecast Start', linewidth=2)
    plt.title(f'Wind Power Forecast - {steps} Day{"s" if steps > 1 else ""}')
    plt.xlabel('Time')
    plt.ylabel('Wind Power')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"wind_dhr_daily_{timestamp}_{steps}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return [csv_path_out, plot_path], params
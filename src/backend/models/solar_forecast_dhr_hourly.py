import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.signal import savgol_filter
from datetime import timedelta, datetime

# --- Fourier Transform ---
def fourier_transform(t, n_harmonics=4, periods=[24, 168]):
    features = []
    for period in periods:
        sin_terms = [np.sin(2 * np.pi * i * t / period) for i in range(1, n_harmonics + 1)]
        cos_terms = [np.cos(2 * np.pi * i * t / period) for i in range(1, n_harmonics + 1)]
        features.extend(sin_terms + cos_terms)
    return np.column_stack(features)

# --- Feature Engineering ---
def create_features(df, target_col, fourier_terms, ar_order, window, polyorder):
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

# --- Extend Data ---
def repeat_last_week(arr, forecast_horizon):
    period = 24 * 7
    repeated = np.tile(arr[-period:], int(np.ceil(forecast_horizon / period)))[:forecast_horizon]
    return np.concatenate([arr, repeated])

# --- Create Feature Vector ---
def create_feature_vector(values, current_pos, fourier_data, ar_order, window, polyorder, ghi, dni, dhi, sza):
    ar_features = values[current_pos - ar_order:current_pos]
    segment = values[current_pos - window:current_pos]
    window_len = min(len(segment), window)
    if window_len % 2 == 0:
        window_len -= 1
    window_len = max(3, window_len)
    smoothed = savgol_filter(segment, window_length=window_len, polyorder=min(polyorder, window_len - 1))
    exog_features = np.array([ghi[current_pos], dni[current_pos], dhi[current_pos], sza[current_pos]])
    return np.concatenate([ar_features, smoothed, fourier_data[current_pos], exog_features])

# --- Forecast Generation ---
def generate_forecast(model, start_values, fourier_data, steps, params, ghi, dni, dhi, sza):
    ar_order = params['ar_order']
    window = params['window']
    polyorder = params['polyorder']
    values = start_values.copy()
    forecasts = []
    current_pos = len(values) - 1
    for _ in range(steps):
        features = create_feature_vector(
            values, current_pos, fourier_data, ar_order, window, polyorder,
            ghi, dni, dhi, sza
        )
        prediction = model.predict(features.reshape(1, -1))[0]
        prediction = max(0, prediction)
        if sza[current_pos] > 90:
            prediction = 0.0
        forecasts.append(prediction)
        values = np.append(values, prediction)
        current_pos += 1
    return np.array(forecasts)

# --- Main Forecast Function ---
def run_forecast(csv_path, steps, output_dir="forecasts", forecast_type="hourly", params=None):

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    
    # Load dataset
    df = pd.read_csv(csv_path, parse_dates=['time'], index_col='time')
    df = df.resample('h').mean().interpolate()  # Resample to hourly data and interpolate

    # Use provided params or raise an error if missing
    if params is None:
        raise ValueError("Parameters must be provided for the forecast.")

    # Create features for the model
    X, y = create_features(df, target_col='solar_power',
                           fourier_terms=params['fourier_terms'],
                           ar_order=params['ar_order'],
                           window=params['window'],
                           polyorder=params['polyorder'])

    # Train model
    model = Ridge(alpha=params['reg_strength'])
    model.fit(X, y)

    # Historical data and exogenous variables
    last_two_weeks = 336  # Last 2 weeks of data (336 hours)
    target_values = df['solar_power'].values
    historical_end_idx = len(target_values)
    historical_start_idx = historical_end_idx - last_two_weeks
    historical_data = target_values[historical_start_idx:historical_end_idx]
    historical_dates = df.index[historical_start_idx:historical_end_idx]

    ghi = df['GHI'].values
    dni = df['DNI'].values
    dhi = df['DHI'].values
    sza = df['Solar Zenith Angle'].values

    # Extend exogenous variables for forecast horizon
    max_horizon = steps  # Use user-specified forecast steps (e.g., 24 hours)
    ghi_ext = repeat_last_week(ghi, max_horizon)
    dni_ext = repeat_last_week(dni, max_horizon)
    dhi_ext = repeat_last_week(dhi, max_horizon)
    sza_ext = repeat_last_week(sza, max_horizon)

    # Extend Fourier features
    extended_df_length = len(df) + max_horizon
    t_extended = np.arange(extended_df_length)
    fourier_extended = fourier_transform(t_extended, n_harmonics=params['fourier_terms'], periods=[24, 168])

    # Forecast generation
    forecast_values = generate_forecast(
        model,
        target_values,
        fourier_extended,
        steps,
        params,
        ghi_ext,
        dni_ext,
        dhi_ext,
        sza_ext
    )

    # Generate forecast dates
    forecast_dates = [historical_dates[-1] + timedelta(hours=i + 1) for i in range(steps)]
    forecast_df = pd.DataFrame({
        'datetime': forecast_dates,
        'forecasted_solar_power': forecast_values
    })

    # Save forecast as CSV
    csv_name = f"solar_dhr_hourly_{timestamp}_{steps}.csv"
    csv_path = os.path.join(output_dir, csv_name)
    forecast_df.to_csv(csv_path, index=False)

    # Plotting
    plt.figure(figsize=(14, 6))
    plt.plot(historical_dates, historical_data, label='Actual (Last 2 weeks)', color='blue', linewidth=2)
    plt.plot(forecast_dates, forecast_values, label=f'Forecast ({steps}h)', color='red', linestyle='--', linewidth=2)
    plt.axvline(x=historical_dates[-1], color='green', linestyle=':', label='Forecast Start', linewidth=2)
    plt.title(f'Solar Power Forecast - {steps} Hour{"s" if steps > 1 else ""}')
    plt.xlabel('Time')
    plt.ylabel('Solar Power')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"solar_dhr_hourly_{timestamp}_{steps}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return [csv_path, plot_path], params


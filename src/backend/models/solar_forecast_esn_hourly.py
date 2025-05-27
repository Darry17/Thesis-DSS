import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import reservoirpy as rp
from reservoirpy.nodes import Reservoir, Ridge, Input
from datetime import timedelta, datetime

def run_forecast(csv_path, steps, output_dir="forecasts", forecast_type="hourly", params=None):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    
    # Use default params if not provided
    if params is None:
        raise ValueError("Parameters must be provided for the forecast.")
    
    # Read and preprocess data
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Assuming the exog variables are present in the data
    exog_columns = ['GHI', 'DNI', 'DHI', 'Solar Zenith Angle']
    
    # Resample and fill missing data
    df_resampled = df.resample('h').mean()
    df_resampled = df_resampled.fillna(method='ffill').fillna(method='bfill')
    
    # Handle missing target variable 'solar_power'
    if df_resampled['solar_power'].isna().any():
        print("Warning: Still have NaN values after filling. Removing them.")
        df_resampled = df_resampled.dropna(subset=['solar_power'])
    
    data = df_resampled['solar_power'].values.astype(float)
    min_non_zero = np.min(data[data > 0]) if np.any(data > 0) else 1e-10
    data_cleaned = np.maximum(data, min_non_zero)
    data_log = np.log1p(data_cleaned)
    
    if np.any(np.isnan(data_log)) or np.any(np.isinf(data_log)):
        print("Warning: Found NaN or Inf values in log-transformed data.")
        data_log = np.nan_to_num(data_log, nan=np.nanmean(data_log),
                                posinf=np.nanmax(data_log), neginf=np.nanmin(data_log))
    
    # Scaling the target (solar_power) and exogenous variables
    scaler_target = MinMaxScaler(feature_range=(0.1, 0.9))
    scaler_exog = MinMaxScaler(feature_range=(0.1, 0.9))
    
    data_scaled_target = scaler_target.fit_transform(data_log.reshape(-1, 1)).flatten()
    
    # Make sure exogenous data has the same length as target data
    exog_data = df_resampled[exog_columns].values
    if len(exog_data) != len(data_scaled_target):
        min_len = min(len(exog_data), len(data_scaled_target))
        exog_data = exog_data[:min_len]
        data_scaled_target = data_scaled_target[:min_len]
        print(f"Adjusted data lengths to match: {min_len}")
    
    data_scaled_exog = scaler_exog.fit_transform(exog_data)
    
    # Create features (lagged target and exog variables)
    lags = params['lags']
    X = []
    y = []
    
    for i in range(lags, len(data_scaled_target)):
        target_seq = data_scaled_target[i-lags:i]
        exog_seq = data_scaled_exog[i-lags:i]
        features = []
        for j in range(lags):
            timestep_features = np.concatenate([[target_seq[j]], exog_seq[j]])
            features.append(timestep_features)
        X.append(features)
        y.append(data_scaled_target[i])
    
    # Convert to numpy arrays and flatten
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    n_samples, lags, n_features = X.shape
    X_flat = X.reshape(n_samples, lags * n_features)
    
    print(f"X_flat shape: {X_flat.shape}, y shape: {y.shape}")
    
    # Split the data
    train_size = int(n_samples * 0.8)
    val_size = int(n_samples * 0.1)
    
    X_train = X_flat[:train_size]
    y_train = y[:train_size]
    
    X_val = X_flat[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    
    X_test = X_flat[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    print(f"Training shapes: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"Validation shapes: X_val {X_val.shape}, y_val {y_val.shape}")
    print(f"Test shapes: X_test {X_test.shape}, y_test {y_test.shape}")
    
    # Build the ESN model
    N_res = params['N_res']
    rho = params['rho']
    alpha = params['alpha']
    sparsity = params['sparsity']
    lambda_reg = params['lambda_reg']
    
    input_node = Input()
    reservoir = Reservoir(
        units=N_res,
        sr=rho,
        lr=alpha,
        input_scaling=1.0,
        rc_connectivity=1 - sparsity,
        seed=42
    )
    readout = Ridge(ridge=lambda_reg)
    model = input_node >> reservoir >> readout
    
    # Train model
    print("Starting training...")
    try:
        model.fit(X_train, y_train, reset=True)
        print("Training complete!")
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    
    # Validate model
    if len(X_val) > 0:
        print("Validating model...")
        y_val_pred = model.run(X_val)
        if isinstance(y_val_pred, list):
            y_val_pred = np.array(y_val_pred)
        if y_val_pred.shape != y_val.shape:
            print(f"Warning: Reshaping predictions from {y_val_pred.shape} to match {y_val.shape}")
            y_val_pred = y_val_pred.reshape(y_val.shape)
        y_val_pred_inv = scaler_target.inverse_transform(y_val_pred)
        y_val_true_inv = scaler_target.inverse_transform(y_val)
        y_val_pred_original = np.expm1(y_val_pred_inv)
        y_val_true_original = np.expm1(y_val_true_inv)
        y_val_pred_original = np.maximum(y_val_pred_original, 0)
        val_rmse = np.sqrt(mean_squared_error(y_val_true_original, y_val_pred_original))
        val_mae = mean_absolute_error(y_val_true_original, y_val_pred_original)
        print(f"Validation RMSE: {val_rmse:.4f}")
        print(f"Validation MAE: {val_mae:.4f}")
    
    # Test model if test set is available
    if len(X_test) > 0:
        print("Generating forecasts on test set...")
        y_test_pred = model.run(X_test)
        if isinstance(y_test_pred, list):
            y_test_pred = np.array(y_test_pred)
        if y_test_pred.shape != y_test.shape:
            print(f"Warning: Reshaping test predictions from {y_test_pred.shape} to match {y_test.shape}")
            y_test_pred = y_test_pred.reshape(y_test.shape)
        y_test_pred_inv = scaler_target.inverse_transform(y_test_pred)
        y_test_true_inv = scaler_target.inverse_transform(y_test)
        y_test_pred_original = np.expm1(y_test_pred_inv)
        y_test_true_original = np.expm1(y_test_true_inv)
        y_test_pred_original = np.maximum(y_test_pred_original, 0)
        rmse = np.sqrt(mean_squared_error(y_test_true_original, y_test_pred_original))
        mae = mean_absolute_error(y_test_true_original, y_test_pred_original)
        cvrmse = (rmse / np.mean(y_test_true_original)) * 100
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test MAE: {mae:.4f}")
        print(f"Test CVRMSE: {cvrmse:.2f}%")
    
    # Prepare for forecasting
    historical_period = 336  # 2 weeks (14 days * 24 hours)
    exog_history_period = 168  # 1 week (7 days * 24 hours) for exogenous variables
    
    # Extract the last segment of data, including historical period
    historical_end_idx = len(data_scaled_target)
    historical_start_idx = historical_end_idx - historical_period - lags
    historical_data = data_scaled_target[historical_start_idx:historical_end_idx]
    historical_data_log = scaler_target.inverse_transform(historical_data.reshape(-1, 1)).flatten()
    historical_data_original = np.expm1(historical_data_log)
    historical_data_original = np.maximum(historical_data_original, 0)
    
    # Get actual dates for the x-axis
    try:
        historical_dates = df_resampled.index[historical_start_idx:historical_end_idx]
        plot_with_dates = True
    except:
        plot_with_dates = False
        print("Could not extract dates for plotting")
    
    # Get the last available sequence for starting the forecast
    last_idx = len(data_scaled_target) - lags
    last_sequence = []
    for i in range(lags):
        timestep_features = np.concatenate([[data_scaled_target[last_idx + i]], 
                                         data_scaled_exog[last_idx + i]])
        last_sequence.append(timestep_features)
    last_sequence = np.array([last_sequence]).reshape(1, lags * n_features)
    
    # Function to generate multi-step forecasts
    def generate_forecast(start_sequence, steps, exog_future=None):
        print(f"Generating {steps}-step ahead forecast...")
        forecast_values = []
        current_sequence = start_sequence.copy()
        for step in range(steps):
            next_pred = model.run(current_sequence)
            pred_value = np.array(next_pred).flatten()[0]
            forecast_values.append(pred_value)
            if exog_future is not None and step < len(exog_future):
                next_exog = exog_future[step]
            else:
                next_exog = np.zeros(n_features - 1)
            next_input = np.concatenate([[pred_value], next_exog])
            current_sequence = np.roll(current_sequence, -n_features, axis=1)
            current_sequence[:, -n_features:] = next_input
        return np.array(forecast_values)
    
    # Use the last week's exogenous variables for future predictions
    last_week_exog = data_scaled_exog[-exog_history_period:]  # Last 168 hours
    # Inverse-transform exogenous variables to check Solar Zenith Angle
    last_week_exog_original = scaler_exog.inverse_transform(last_week_exog)
    # Index 3 corresponds to 'Solar Zenith Angle' (after GHI, DNI, DHI)
    solar_zenith_angles = last_week_exog_original[:, 3]
    
    # Prepare exogenous inputs for the forecast horizon
    if steps <= len(last_week_exog):
        exog_future = last_week_exog[:steps]  # Use first 'steps' steps
        exog_future_original = last_week_exog_original[:steps]
        zenith_angles = solar_zenith_angles[:steps]
    else:
        # Repeat the last week's exog pattern to cover longer horizons
        exog_future = np.tile(last_week_exog, (steps // len(last_week_exog) + 1, 1))[:steps]
        exog_future_original = np.tile(last_week_exog_original, (steps // len(last_week_exog) + 1, 1))[:steps]
        zenith_angles = np.tile(solar_zenith_angles, (steps // len(last_week_exog) + 1))[:steps]
    
    # Generate forecast
    forecast_scaled = generate_forecast(last_sequence, steps, exog_future)
    
    # Inverse transform and post-process
    forecast_log = scaler_target.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
    forecast_original = np.expm1(forecast_log)
    forecast_original = np.maximum(forecast_original, 0)
    
    # Post-processing: Set forecast to 0 when Solar Zenith Angle >= 90 degrees
    nighttime_mask = zenith_angles >= 90
    forecast_original[nighttime_mask] = 0
    
    # Generate forecast dates
    forecast_dates = [df_resampled.index[-1] + timedelta(hours=i + 1) for i in range(steps)]
    forecast_df = pd.DataFrame({
        'datetime': forecast_dates,
        'forecasted_solar_power': forecast_original
    })
    
    # Save forecast as CSV
    csv_name = f"solar_esn_hourly_{timestamp}_{steps}.csv"
    csv_path_output = os.path.join(output_dir, csv_name)
    forecast_df.to_csv(csv_path_output, index=False)
    
    # Create and save the forecast plot
    plt.figure(figsize=(14, 6))
    if plot_with_dates:
        plt.plot(historical_dates, historical_data_original, label='Historical Solar Power', color='blue', linewidth=2)
        plt.plot(forecast_dates, forecast_original, label=f'{steps}-step Forecast', color='red', linestyle='--', linewidth=2)
        plt.axvline(x=df_resampled.index[-1], color='green', linestyle=':', label='Forecast Start', linewidth=2)
        plt.gcf().autofmt_xdate()
    else:
        plt.plot(range(historical_period), historical_data_original, label='Historical Solar Power', color='blue', linewidth=2)
        plt.plot(range(historical_period, historical_period + steps), forecast_original, 
                 label=f'{steps}-step Forecast', color='red', linestyle='--', linewidth=2)
        plt.axvline(x=historical_period, color='green', linestyle=':', label='Forecast Start', linewidth=2)
        
    plt.title(f'ESN Solar Power Forecast - {steps} Hour{"s" if steps > 1 else ""}')
    plt.xlabel('Time')
    plt.ylabel('Solar Power')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_name = f"solar_esn_hourly_{timestamp}_{steps}.png"
    plot_path = os.path.join(output_dir, plot_name)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Forecast saved to {csv_path_output}")
    print(f"Plot saved to {plot_path}")
    
    # Return file paths
    return [csv_path_output, plot_path]
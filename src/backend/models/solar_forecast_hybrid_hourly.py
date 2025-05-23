import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter
from datetime import timedelta, datetime
import reservoirpy as rp
from reservoirpy.nodes import Reservoir, Ridge as RPRidge, Input

def run_forecast(csv_path, steps, output_dir="forecasts", forecast_type="hourly", params=None):
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    
    # Output file paths
    csv_name = f"solar_hybrid_{timestamp}_{steps}.csv"
    csv_path_output = os.path.join(output_dir, csv_name)
    plot_name = f"solar_hybrid_{timestamp}_{steps}.png"
    plot_path = os.path.join(output_dir, plot_name)
    
    # Structure parameters properly if they come as flat dict
    if params is None:
        params = {}
    
    # Convert flat parameter structure to nested structure
    structured_params = {
        'dhr': {
            'fourier_terms': params.get('fourier_terms', 4),
            'reg_strength': params.get('reg_strength', 0.006),
            'ar_order': params.get('ar_order', 1),
            'window': params.get('window', 23),
            'polyorder': params.get('polyorder', 2)
        },
        'esn': {
            'N_res': params.get('N_res', 800),
            'rho': params.get('rho', 0.9308202574),
            'sparsity': params.get('sparsity', 0.1335175715),
            'alpha': params.get('alpha', 0.7191611348),
            'lambda_reg': params.get('lambda_reg', 2.1e-8),
            'lags': params.get('lags', 24),
            'n_features': 5  # Number of features (solar_power + 4 exog)
        },
        'hybrid': {
            'weights': [0.5, 0.5]  # Default weights [DHR, ESN]
        }
    }
    
    # Load and preprocess data
    print("Loading data...")
    df, target_col, exog_cols = load_data(csv_path)
    print(f"Dataset shape: {df.shape}, Columns: {df.columns.tolist()}")
    
    # Train DHR model
    print("Training DHR model...")
    dhr_model = train_complete_dhr_model(df, target_col, exog_cols, structured_params['dhr'])
    
    # Train ESN model
    print("Training ESN model...")
    esn_model, esn_scaler_target, esn_scaler_exog = train_complete_esn_model(df, target_col, exog_cols, structured_params['esn'])
    
    # Generate forecast
    print(f"Generating {steps}-step forecast...")
    forecast_df = generate_forecast(
        df, 
        dhr_model, 
        esn_model, 
        esn_scaler_target, 
        esn_scaler_exog, 
        steps, 
        structured_params['dhr'], 
        structured_params['esn'], 
        structured_params['hybrid']['weights'],
        exog_cols
    )
    
    # Save forecast to CSV
    forecast_df.to_csv(csv_path_output, index=False)
    print(f"Forecast saved to {csv_path_output}")
    
    return [csv_path_output, plot_path]

def load_data(csv_path):
    """Load and preprocess data"""
    df = pd.read_csv(csv_path, parse_dates=['time'], index_col='time')
    
    # Define target and exogenous variables
    target_col = 'solar_power'
    exog_cols = ['GHI', 'DNI', 'DHI', 'Solar Zenith Angle']
    
    # Select relevant columns and handle missing data
    if all(col in df.columns for col in [target_col] + exog_cols):
        df = df[[target_col] + exog_cols]
    else:
        available_cols = [col for col in ([target_col] + exog_cols) if col in df.columns]
        df = df[available_cols]
        print(f"Warning: Not all expected columns available. Using {available_cols}")
        
        # If exogenous variables are missing, create dummy ones
        for col in exog_cols:
            if col not in df.columns:
                df[col] = 0.0
    
    # Resample to hourly and interpolate
    df_resampled = df.resample('h').mean().interpolate()
    
    return df_resampled, target_col, exog_cols

# DHR MODEL COMPONENTS
# -------------------
def fourier_transform(df, n_harmonics=4):
    """Fixed Fourier Transform (Period = 24)"""
    t = np.arange(len(df))
    period = 24
    sin_terms = [np.sin(2 * np.pi * i * t / period) for i in range(1, n_harmonics + 1)]
    cos_terms = [np.cos(2 * np.pi * i * t / period) for i in range(1, n_harmonics + 1)]
    return np.column_stack(sin_terms + cos_terms)

def create_dhr_features(df, target_col, exog_cols, fourier_terms, ar_order, window, polyorder):
    """Create features for DHR model including exogenous variables"""
    fourier = fourier_transform(df, n_harmonics=fourier_terms)
    target = df[target_col].values
    exog_data = df[exog_cols].values

    X, y = [], []
    for i in range(max(ar_order, window), len(target)):
        ar_features = target[i - ar_order:i]
        smoothed = savgol_filter(target[i - window:i], window_length=window, polyorder=polyorder)
        exog_features = exog_data[i]
        features = np.concatenate([ar_features, smoothed, fourier[i], exog_features])
        X.append(features)
        y.append(target[i])
    return np.array(X), np.array(y)

def train_dhr_model(X_train, y_train, reg_strength):
    """Train DHR model"""
    model = Ridge(alpha=reg_strength)
    model.fit(X_train, y_train)
    return model

def train_complete_dhr_model(df, target_col, exog_cols, params):
    """Train complete DHR model using all data"""
    # Create features
    X, y = create_dhr_features(
        df, target_col, exog_cols, 
        params['fourier_terms'], params['ar_order'], 
        params['window'], params['polyorder']
    )
    
    # Train model
    model = Ridge(alpha=params['reg_strength'])
    model.fit(X, y)
    
    return model

def predict_dhr(model, X):
    """Generate and post-process DHR predictions"""
    predictions = model.predict(X)
    predictions = np.maximum(predictions, 0)
    return predictions

# ESN MODEL COMPONENTS
# -------------------
def prepare_esn_data(data, exog_data):
    """Prepare data for ESN model with log transformation and scaling"""
    # Clean and transform target data
    min_non_zero = np.min(data[data > 0]) if np.any(data > 0) else 1e-10
    data_cleaned = np.maximum(data, min_non_zero)
    data_log = np.log1p(data_cleaned)
    
    if np.any(np.isnan(data_log)) or np.any(np.isinf(data_log)):
        data_log = np.nan_to_num(data_log, nan=np.nanmean(data_log),
                                 posinf=np.nanmax(data_log), neginf=np.nanmin(data_log))
    
    # Scale target
    scaler_target = MinMaxScaler(feature_range=(0.1, 0.9))
    data_scaled = scaler_target.fit_transform(data_log.reshape(-1, 1)).flatten()
    
    # Scale exogenous variables
    scaler_exog = MinMaxScaler(feature_range=(0.1, 0.9))
    exog_scaled = scaler_exog.fit_transform(exog_data)
    
    # Combine scaled target and exogenous data
    combined_data = np.column_stack([data_scaled, exog_scaled])
    return combined_data, scaler_target, scaler_exog

def create_sequences(data, lags):
    """Create sequences for ESN with exogenous variables"""
    X, y = [], []
    for i in range(len(data) - lags):
        X.append(data[i:i + lags])
        y.append(data[i + lags, 0])  # Target is first column
    X = np.array(X)  # Shape: [n_samples, lags, n_features]
    y = np.array(y).reshape(-1, 1)  # Shape: [n_samples, 1]
    return X, y

def build_esn_model(N_res, rho, sparsity, alpha, lambda_reg, input_dim):
    """Build ESN model with specified input dimension"""
    input_node = Input(input_dim=input_dim)
    reservoir = Reservoir(
        units=N_res,
        sr=rho,
        lr=alpha,
        input_scaling=1.0,
        rc_connectivity=1 - sparsity,
        seed=42
    )
    readout = RPRidge(ridge=lambda_reg)
    model = input_node >> reservoir >> readout
    return model

def train_esn_model(model, X_train, y_train):
    """Train ESN model using model.fit with reshaped input"""
    try:
        rp.verbosity(0)  # Suppress progress bars for speed
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        
        # Reshape X_train to [n_samples, lags * n_features]
        n_samples, lags, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(n_samples, lags * n_features)
        print(f"Reshaped X_train shape: {X_train_reshaped.shape}")
        
        # Train with reshaped input
        model.fit(X_train_reshaped, y_train, warmup=0)
        print("ESN training complete!")
        return model
    except Exception as e:
        print(f"Error during ESN training: {e}")
        raise e

def train_complete_esn_model(df, target_col, exog_cols, params):
    """Train complete ESN model using all data"""
    # Prepare data
    solar_data = df[target_col].values
    exog_data = df[exog_cols].values
    combined_data, scaler_target, scaler_exog = prepare_esn_data(solar_data, exog_data)
    
    # Create sequences
    X, y = create_sequences(combined_data, params['lags'])
    
    # Build model
    model = build_esn_model(
        params['N_res'], params['rho'], params['sparsity'], 
        params['alpha'], params['lambda_reg'], 
        input_dim=params['lags'] * params['n_features']
    )
    
    # Train model (suppress progress bars)
    rp.verbosity(0)
    
    # Reshape input for training
    n_samples, lags, n_features = X.shape
    X_reshaped = X.reshape(n_samples, lags * n_features)
    
    # Train model
    model.fit(X_reshaped, y, warmup=0)
    
    return model, scaler_target, scaler_exog

def predict_esn(model, X, scaler_target):
    """Generate and post-process ESN predictions with reshaped input"""
    n_samples, lags, n_features = X.shape
    y_pred = []

    for i in range(n_samples):
        sequence = X[i]  # Shape: [lags, n_features]
        sequence_reshaped = sequence.reshape(lags * n_features)  # Shape: [lags * n_features]
        pred = model.run(sequence_reshaped.reshape(1, -1))  # Shape: [1, lags * n_features]
        y_pred.append(pred[0])

    y_pred = np.array(y_pred).reshape(-1, 1)
    y_pred_inv = scaler_target.inverse_transform(y_pred)
    y_pred_original = np.expm1(y_pred_inv)
    y_pred_original = np.maximum(y_pred_original, 0)
    return y_pred_original.flatten()

# HYBRID MODEL COMPONENTS
# ----------------------
def weighted_average(dhr_preds, esn_preds, weights=None):
    """Combine predictions using weighted moving average"""
    if weights is None:
        weights = [0.5, 0.5]
    weights = np.array(weights) / np.sum(weights)
    combined = weights[0] * dhr_preds + weights[1] * esn_preds
    return combined

def optimize_weights(dhr_preds, esn_preds, y_true):
    """Find optimal weights using grid search"""
    best_rmse = float('inf')
    best_weights = [0.5, 0.5]
    for w1 in np.linspace(0.1, 0.9, 9):
        w2 = 1 - w1
        weights = [w1, w2]
        combined = weighted_average(dhr_preds, esn_preds, weights)
        rmse = np.sqrt(mean_squared_error(y_true, combined))
        if rmse < best_rmse:
            best_rmse = rmse
            best_weights = weights
    print(f"Optimized weights: DHR={best_weights[0]:.2f}, ESN={best_weights[1]:.2f}")
    return best_weights

def evaluate_model(y_true, y_pred):
    """Calculate and print evaluation metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    cvrmse = (rmse / np.mean(y_true)) * 100 if np.mean(y_true) != 0 else float('inf')
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"CV-RMSE: {cvrmse:.2f}%")
    return rmse, mae, cvrmse

def plot_results(dates, actual, dhr_preds, esn_preds, hybrid_preds, title="Model Comparison"):
    """Plot actual vs predictions for all models"""
    plt.figure(figsize=(14, 8))
    plt.plot(dates, actual, label='Actual', color='black', linewidth=2)
    plt.plot(dates, dhr_preds, label='DHR Prediction', color='blue', linestyle='--', alpha=0.7)
    plt.plot(dates, esn_preds, label='ESN Prediction', color='green', linestyle='--', alpha=0.7)
    plt.plot(dates, hybrid_preds, label='Hybrid Prediction', color='red', linewidth=2)
    plt.title(title, fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Solar Power', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.savefig('hybrid_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_forecast(df, dhr_model, esn_model, esn_scaler_target, esn_scaler_exog, 
                      horizon, dhr_params, esn_params, weights, exog_cols):
    """Generate multi-step forecasts"""
    target_col = 'solar_power'
    target = df[target_col].values
    exog_data = df[exog_cols].values
    
    # Prepare ESN data
    combined_data, _, _ = prepare_esn_data(target, exog_data)
    lags = esn_params['lags']
    
    # Create forecast dates
    forecast_dates = [df.index[-1] + timedelta(hours=i+1) for i in range(horizon)]
    
    # Get last week's exog values for forecasting (seasonal pattern)
    week_ago_idx = len(df) - 168
    if week_ago_idx < 0:
        print(f"Warning: Not enough data for seasonal forecasting. Using last available exog values.")
        exog_forecast = np.tile(exog_data[-1], (horizon, 1))
    else:
        exog_forecast = exog_data[week_ago_idx:week_ago_idx + horizon]
        if len(exog_forecast) < horizon:
            exog_forecast = np.pad(exog_forecast, ((0, horizon - len(exog_forecast)), (0, 0)), mode='edge')
    
    # Scale exog forecast
    exog_forecast_scaled = esn_scaler_exog.transform(exog_forecast)
    
    # Fourier terms for forecast period
    t = np.arange(len(df), len(df) + horizon)
    period = 24
    n_harmonics = dhr_params['fourier_terms']
    sin_terms = [np.sin(2 * np.pi * i * t / period) for i in range(1, n_harmonics + 1)]
    cos_terms = [np.cos(2 * np.pi * i * t / period) for i in range(1, n_harmonics + 1)]
    fourier_forecast = np.column_stack(sin_terms + cos_terms)
    
    # Initialize forecast lists
    dhr_forecast = []
    esn_forecast = []
    hybrid_forecast = []
    
    # Current state for recursive forecasting
    current_target = target.copy()
    last_sequence = combined_data[-lags:].reshape(1, lags, -1)
    
    # Generate forecasts step by step
    for h in range(horizon):
        # DHR FORECASTING
        ar_order = dhr_params['ar_order']
        window = dhr_params['window']
        polyorder = dhr_params['polyorder']
        
        ar_features = current_target[-ar_order:]
        smoothed = savgol_filter(current_target[-window:], window_length=window, polyorder=polyorder)
        dhr_features = np.concatenate([ar_features, smoothed, fourier_forecast[h], exog_forecast[h]])
        dhr_features = dhr_features.reshape(1, -1)
        dhr_pred = dhr_model.predict(dhr_features)[0]
        dhr_pred = max(dhr_pred, 0)
        
        # Set to 0 if Solar Zenith Angle > 90 (nighttime)
        if len(exog_forecast[h]) > 3 and exog_forecast[h][3] > 90:
            dhr_pred = 0
            
        dhr_forecast.append(dhr_pred)
        current_target = np.append(current_target, dhr_pred)
        
        # ESN FORECASTING
        n_features = last_sequence.shape[2]
        esn_pred = esn_model.run(last_sequence.reshape(1, lags * n_features))
        esn_pred = np.array(esn_pred).reshape(-1, 1)
        esn_pred_value = esn_pred[0, 0]
        esn_pred_inv = esn_scaler_target.inverse_transform(esn_pred)
        esn_pred_original = np.maximum(np.expm1(esn_pred_inv)[0][0], 0)
        
        # Set to 0 if Solar Zenith Angle > 90 (nighttime)
        if len(exog_forecast[h]) > 3 and exog_forecast[h][3] > 90:
            esn_pred_original = 0
            
        esn_forecast.append(esn_pred_original)
        
        # Update ESN sequence for next prediction
        last_sequence = np.roll(last_sequence, -1, axis=1)
        last_sequence[0, -1, :] = np.concatenate([[esn_pred_value], exog_forecast_scaled[h]])
        
        # HYBRID FORECAST (weighted average)
        hybrid_value = weights[0] * dhr_pred + weights[1] * esn_pred_original
        
        # Set to 0 if Solar Zenith Angle > 90 (nighttime)
        if len(exog_forecast[h]) > 3 and exog_forecast[h][3] > 90:
            hybrid_value = 0
            
        hybrid_forecast.append(hybrid_value)
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'timestamp': forecast_dates,
        'dhr_forecast': dhr_forecast,
        'esn_forecast': esn_forecast,
        'hybrid_forecast': hybrid_forecast
    })
    
    return forecast_df
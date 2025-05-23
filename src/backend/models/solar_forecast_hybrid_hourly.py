import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter
from datetime import datetime, timedelta
import reservoirpy as rp
from reservoirpy.nodes import Reservoir, Ridge as RPRidge, Input
from dotenv import load_dotenv

# Common data loading and preprocessing
def load_data(data_path):
    """Load and preprocess data for both models"""
    df = pd.read_csv(data_path, parse_dates=['time'], index_col='time')
    # Define target and exogenous variables
    target_col = 'solar_power'
    exog_cols = ['GHI', 'DNI', 'DHI', 'Solar Zenith Angle']
    # Select relevant columns
    df = df[[target_col] + exog_cols]
    # Resample to daily and interpolate
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
        print(f"Expected input dimension: {lags * n_features}")
        
        # Train with reshaped input
        model.fit(X_train_reshaped, y_train, warmup=0)
        print("ESN training complete!")
        return model
    except Exception as e:
        print(f"Error during ESN training: {e}")
        print(f"Model input dimension: {model.nodes[0].input_dim}")
        print(f"Actual data dimension: {X_train_reshaped.shape[1]}")
        raise e

def predict_esn(model, X, scaler_target):
    """Generate and post-process ESN predictions with reshaped input"""
    reservoir = model.nodes[1]  # Reservoir node
    readout = model.nodes[2]    # Ridge readout node
    n_samples, lags, n_features = X.shape
    y_pred = []

    for i in range(n_samples):
        sequence = X[i]  # Shape: [lags, n_features]
        sequence_reshaped = sequence.reshape(lags * n_features)  # Shape: [lags * n_features]
        state = reservoir.run(sequence_reshaped)  # Shape: [N_res]
        pred = readout.run(state)  # Shape: [1]
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

def plot_results(dates, actual, dhr_preds, esn_preds, hybrid_preds, title="Model Comparison", save_path=None):
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
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.savefig('hybrid_model_comparison.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    return save_path if save_path else 'hybrid_model_comparison.png'

# FORECAST GENERATION FUNCTION
# ---------------------------
def generate_forecast(df, dhr_model, esn_model, esn_scaler_target, esn_scaler_exog, best_weights, steps, lags=24, exog_cols=['GHI', 'DNI', 'DHI', 'Solar Zenith Angle'], output_dir="forecasts", timestamp=""):
    """Generate forecast for specified number of steps"""
    target_col = 'solar_power'
    target = df[target_col].values
    exog_data = df[exog_cols].values

    # Prepare ESN data
    combined_data, _, _ = prepare_esn_data(target, exog_data)
    
    # Get actual dimensions from the data
    n_samples, n_features = combined_data.shape
    print(f"Combined data shape: {combined_data.shape} (samples: {n_samples}, features: {n_features})")
    
    print(f"\nGenerating {steps}-step forecast...")
    forecast_dates = [df.index[-1] + timedelta(hours=i+1) for i in range(steps)]
    
    # Get last week's exog values (168 hours = 7 days ago) for forecasting
    week_ago_idx = len(df) - 168 - 1
    if week_ago_idx < 0:
        print(f"Warning: Not enough data for {steps}-step forecast. Using last available exog values.")
        exog_forecast = np.tile(exog_data[-1], (steps, 1))
    else:
        exog_forecast = exog_data[week_ago_idx:week_ago_idx + steps]
        if len(exog_forecast) < steps:
            exog_forecast = np.pad(exog_forecast, ((0, steps - len(exog_forecast)), (0, 0)), mode='edge')
    
    # Scale exog forecast
    exog_forecast_scaled = esn_scaler_exog.transform(exog_forecast)
    
    # Fourier terms for forecast period
    t = np.arange(len(df), len(df) + steps)
    period = 24
    n_harmonics = 4
    sin_terms = [np.sin(2 * np.pi * i * t / period) for i in range(1, n_harmonics + 1)]
    cos_terms = [np.cos(2 * np.pi * i * t / period) for i in range(1, n_harmonics + 1)]
    fourier_forecast = np.column_stack(sin_terms + cos_terms)
    
    # Initialize forecast lists
    dhr_forecast = []
    esn_forecast = []
    hybrid_forecast = []
    current_target = target.copy()
    last_sequence = combined_data[-lags:].reshape(1, lags, -1)
    
    # Calculate actual input dimension for ESN
    actual_input_dim = lags * n_features
    print(f"ESN input dimension: {actual_input_dim} (lags: {lags}, features: {n_features})")
    
    for h in range(steps):
        # DHR FORECASTING
        ar_order = 1
        window = 23
        polyorder = 2
        ar_features = current_target[-ar_order:]
        smoothed = savgol_filter(current_target[-window:], window_length=window, polyorder=polyorder) if len(current_target) >= window else np.zeros(window)
        dhr_features = np.concatenate([ar_features, smoothed, fourier_forecast[h], exog_forecast[h]])
        dhr_features = dhr_features.reshape(1, -1)
        dhr_pred = dhr_model.predict(dhr_features)[0]
        dhr_pred = max(dhr_pred, 0)
        # Set to 0 if Solar Zenith Angle > 90
        if exog_forecast[h][3] > 90:  # Solar Zenith Angle is last column
            dhr_pred = 0
        dhr_forecast.append(dhr_pred)
        current_target = np.append(current_target, dhr_pred)
        
        # ESN FORECASTING
        esn_pred = esn_model.run(last_sequence.reshape(1, actual_input_dim))  # Use calculated dimension
        esn_pred = np.array(esn_pred).reshape(-1, 1)
        esn_pred_value = esn_pred[0, 0]
        esn_pred_inv = esn_scaler_target.inverse_transform(esn_pred)
        esn_pred_original = np.maximum(np.expm1(esn_pred_inv)[0][0], 0)
        # Set to 0 if Solar Zenith Angle > 90
        if exog_forecast[h][3] > 90:  # Solar Zenith Angle is last column
            esn_pred_original = 0
        esn_forecast.append(esn_pred_original)
        
        # Update ESN sequence
        last_sequence = np.roll(last_sequence, -1, axis=1)
        last_sequence[0, -1, :] = np.concatenate([[esn_pred_value], exog_forecast_scaled[h]])
        
        # HYBRID FORECAST
        hybrid_value = weighted_average(np.array([dhr_pred]), np.array([esn_pred_original]), best_weights)[0]
        # Set to 0 if Solar Zenith Angle > 90
        if exog_forecast[h][3] > 90:  # Solar Zenith Angle is last column
            hybrid_value = 0
        hybrid_forecast.append(hybrid_value)
    
    # Get last two weeks (336 hours) of actual data
    two_weeks_ago_idx = len(df) - 336
    if two_weeks_ago_idx < 0:
        print(f"Warning: Not enough data for two weeks. Using available actual data.")
        two_weeks_ago_idx = 0
    actual_dates = df.index[two_weeks_ago_idx:].tolist()
    actual_values = df[target_col].values[two_weeks_ago_idx:]
    
    # Combine actual and forecast data
    all_dates = actual_dates + forecast_dates
    actual_extended = np.concatenate([actual_values, np.full(steps, np.nan)])  # Pad actual with NaN for forecast period
    dhr_extended = np.concatenate([np.full(len(actual_values), np.nan), dhr_forecast])  # Pad forecast with NaN for actual period
    esn_extended = np.concatenate([np.full(len(actual_values), np.nan), esn_forecast])
    hybrid_extended = np.concatenate([np.full(len(actual_values), np.nan), hybrid_forecast])
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'datetime': forecast_dates,
        'dhr_forecast': dhr_forecast,
        'esn_forecast': esn_forecast,
        'hybrid_forecast': hybrid_forecast
    })
    
    # Save forecast
    csv_filename = f'solar_hybrid_hourly_{timestamp}_{steps}.csv'
    csv_path_output = os.path.join(output_dir, csv_filename)
    forecast_df.to_csv(csv_path_output, index=False)
    print(f"Forecast saved to '{csv_path_output}'")
    
    # Plot forecast with actual data and separation line
    plot_filename = f'solar_hybrid_hourly_{timestamp}_{steps}.png'
    plot_path = os.path.join(output_dir, plot_filename)
    
    plt.figure(figsize=(14, 7))
    plt.plot(all_dates, actual_extended, label='Actual', color='black', linewidth=2)
    plt.plot(all_dates, dhr_extended, label='DHR Forecast', color='blue', linestyle='--')
    plt.plot(all_dates, esn_extended, label='ESN Forecast', color='green', linestyle='--')
    plt.plot(all_dates, hybrid_extended, label='Hybrid Forecast', color='red', linewidth=2)
    plt.axvline(x=df.index[-1], color='black', linestyle=':', label='Forecast Start', alpha=0.7)
    plt.title(f'{steps}-Hour Solar Power Forecast with Historical Data', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Solar Power', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return forecast_df, csv_path_output, plot_path

def train_models(df, target_col, exog_cols, params=None):
    """Train both DHR and ESN models and return trained models with scalers"""
    # DHR Model Parameters
    if params is None:
        params = {}
    
    # DHR Model Parameters - Now customizable!
    dhr_params = {
        'fourier_terms': params.get('fourier_terms', 4),
        'reg_strength': params.get('reg_strength', 0.006),
        'ar_order': params.get('ar_order', 1),
        'window': params.get('window', 23),
        'polyorder': params.get('polyorder', 2)
    }
    
    # ESN Model Parameters - Now customizable!
    esn_params = {
        'N_res': params.get('N_res', 800),
        'rho': params.get('rho', 0.9308202574),
        'sparsity': params.get('sparsity', 0.1335175715),
        'alpha': params.get('alpha', 0.7191611348),
        'lambda_reg': params.get('lambda_reg', 2.10E-08),
        'lags': params.get('lags', 24),
        'n_features': params.get('n_features', 5)
    }
    
    # Create DHR features
    print("Creating DHR features...")
    X_dhr, y_dhr = create_dhr_features(
        df,
        target_col=target_col,
        exog_cols=exog_cols,
        fourier_terms=dhr_params['fourier_terms'],
        ar_order=dhr_params['ar_order'],
        window=dhr_params['window'],
        polyorder=dhr_params['polyorder']
    )
    
    # Split data for DHR
    print("Splitting data for DHR...")
    X_train_dhr, X_temp_dhr, y_train_dhr, y_temp_dhr = train_test_split(
        X_dhr, y_dhr, test_size=0.2, shuffle=False
    )
    X_val_dhr, X_test_dhr, y_val_dhr, y_test_dhr = train_test_split(
        X_temp_dhr, y_temp_dhr, test_size=0.5, shuffle=False
    )
    
    # Train DHR model
    print("Training DHR model...")
    dhr_model = train_dhr_model(X_train_dhr, y_train_dhr, dhr_params['reg_strength'])
    
    # Prepare data for ESN model
    print("Preparing ESN data...")
    solar_data = df[target_col].values
    exog_data = df[exog_cols].values
    combined_data, esn_scaler_target, esn_scaler_exog = prepare_esn_data(solar_data, exog_data)
    
    # Create sequences for ESN
    print("Creating ESN sequences...")
    X_esn, y_esn = create_sequences(combined_data, esn_params['lags'])
    print(f"ESN sequences shape: X_esn {X_esn.shape}, y_esn {y_esn.shape}")
    
    # Get actual dimensions from the data
    n_samples, lags, n_features = X_esn.shape
    actual_input_dim = lags * n_features
    print(f"Calculated input dimension: {actual_input_dim} (lags: {lags}, features: {n_features})")
    
    # Update ESN parameters with actual dimensions
    esn_params['lags'] = lags
    esn_params['n_features'] = n_features
    esn_params['input_dim'] = actual_input_dim
    
    # Split ESN data
    print("Splitting data for ESN...")
    total_size = len(X_esn)
    train_size = int(total_size * 0.8)
    val_size = len(y_val_dhr)
    test_size = len(y_test_dhr)
    X_train_esn = X_esn[:train_size]
    y_train_esn = y_esn[:train_size]
    X_val_esn = X_esn[train_size:train_size + val_size]
    y_val_esn = y_esn[train_size:train_size + val_size]
    
    # Build and train ESN model with actual input dimension
    print("Building and training ESN model...")
    esn_model = build_esn_model(
        esn_params['N_res'],
        esn_params['rho'],
        esn_params['sparsity'],
        esn_params['alpha'],
        esn_params['lambda_reg'],
        input_dim=actual_input_dim  # Use calculated dimension
    )
    esn_model = train_esn_model(esn_model, X_train_esn, y_train_esn)
    
    # Optimize weights using validation set
    print("Optimizing weights...")
    dhr_val_preds = predict_dhr(dhr_model, X_val_dhr)
    esn_val_preds = predict_esn(esn_model, X_val_esn, esn_scaler_target)
    best_weights = optimize_weights(dhr_val_preds, esn_val_preds.flatten(), y_val_dhr)
    
    return dhr_model, esn_model, esn_scaler_target, esn_scaler_exog, best_weights

def run_forecast(csv_path, steps, output_dir="forecasts", forecast_type="daily", params=None):

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    
    print(f"Starting solar power forecast...")
    print(f"Input CSV: {csv_path}")
    print(f"Forecast steps: {steps}")
    print(f"Output directory: {output_dir}")
    print(f"Custom parameters: {params}")
    print(f"Timestamp: {timestamp}")
    
    try:
        # Load data
        print("Loading data...")
        df, target_col, exog_cols = load_data(csv_path)
        print(f"Dataset shape: {df.shape}, Columns: {df.columns.tolist()}")
        
        # Train models with custom parameters
        print("Training models...")
        dhr_model, esn_model, esn_scaler_target, esn_scaler_exog, best_weights = train_models(
            df, target_col, exog_cols, params=params
        )
        
        # Generate forecast
        print("Generating forecast...")
        forecast_df, csv_path_output, plot_path = generate_forecast(
            df, dhr_model, esn_model, esn_scaler_target, esn_scaler_exog, 
            best_weights, steps, output_dir=output_dir, timestamp=timestamp
        )
        
        print(f"\nForecast completed successfully!")
        print(f"CSV saved to: {csv_path_output}")
        print(f"Plot saved to: {plot_path}")
        
        return [csv_path_output, plot_path]
        
    except Exception as e:
        print(f"Error in run_forecast: {str(e)}")
        raise e
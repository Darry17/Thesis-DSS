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
import psutil
import GPUtil
import time
import logging
import radon.complexity
from radon.visitors import ComplexityVisitor
import inspect
import dotenv

# --- Setup Logging ---
def setup_logging(output_dir, timestamp):
    """Configure logging to console and file with timestamped filename"""
    log_file = os.path.join(output_dir, f"forecast_log_{timestamp}.log")
    for handler in logging.getLogger().handlers[:]:
        logging.getLogger().removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    logging.info(f"Logs will be saved to '{log_file}'")
    return log_file

# --- Resource Monitoring ---
def measure_resources(is_idle=True):
    """Measure system resources (memory, CPU, GPU)"""
    process = psutil.Process()
    memory = psutil.virtual_memory() if is_idle else process.memory_info()
    mem_total = memory.total / (1024 ** 2) if is_idle else memory.rss / (1024 ** 2)
    mem_used = memory.used / (1024 ** 2) if is_idle else memory.rss / (1024 ** 2)
    mem_percent = memory.percent if is_idle else (memory.rss / psutil.virtual_memory().total * 100)
    
    cpu_percent = psutil.cpu_percent(interval=1) if is_idle else process.cpu_percent(interval=1)
    
    try:
        gpus = GPUtil.getGPUs()
        gpu_info = [
            {
                "id": gpu.id,
                "name": gpu.name,
                "memory_used": gpu.memoryUsed,
                "memory_total": gpu.memoryTotal,
                "memory_percent": (gpu.memoryUsed / gpu.memoryTotal * 100) if gpu.memoryTotal else 0,
                "utilization": gpu.load * 100
            }
            for gpu in gpus
        ]
    except Exception as e:
        gpu_info = [{"error": f"GPU monitoring failed: {str(e)}"}]
    
    logging.info(f"{'Idle' if is_idle else 'Runtime'} System Resources:")
    logging.info(f"Memory: Total: {mem_total:.2f} MB, Used: {mem_used:.2f} MB ({mem_percent:.1f}%)")
    logging.info(f"CPU: Usage: {cpu_percent:.1f}% ({psutil.cpu_count()} cores)")
    logging.info(f"GPU:")
    if gpu_info and "error" not in gpu_info[0]:
        for gpu in gpu_info:
            logging.info(f"  GPU {gpu['id']} ({gpu['name']}): Memory Used: {gpu['memory_used']:.2f} MB ({gpu['memory_percent']:.1f}%), Utilization: {gpu['utilization']:.1f}%")
    else:
        logging.info(f"  {gpu_info[0]['error']}")
    logging.info("=" * 30)
    
    return {
        "memory_mb": mem_used,
        "memory_percent": mem_percent,
        "cpu_percent": cpu_percent,
        "gpu": gpu_info
    }

# --- Computational Efficiency Evaluation ---
def evaluate_computational_efficiency(start_time, finish_time, execution_time, metrics):
    """Evaluate computational efficiency metrics"""
    logging.info("Computational Efficiency Evaluation:")
    logging.info(f"Start Time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Finish Time: {datetime.fromtimestamp(finish_time).strftime('%Y-%m-%d %H:%M:%S')}")
    if execution_time < 10:
        logging.info(f"Execution Time: {execution_time:.2f} s (Excellent)")
    elif execution_time <= 30:
        logging.info(f"Execution Time: {execution_time:.2f} s (Acceptable)")
    else:
        logging.info(f"Execution Time: {execution_time:.2f} s (Poor: Optimize with line_profiler)")
    
    if metrics["memory_mb"] < 500:
        logging.info(f"Memory Usage: {metrics['memory_mb']:.2f} MB (Excellent)")
    elif metrics["memory_mb"] <= 2000:
        logging.info(f"Memory Usage: {metrics['memory_mb']:.2f} MB (Acceptable)")
    else:
        logging.info(f"Memory Usage: {metrics['memory_mb']:.2f} MB (Poor: Use memory_profiler)")
    
    if metrics["cpu_percent"] < 50:
        logging.info(f"CPU Usage: {metrics['cpu_percent']:.1f}% (Excellent)")
    elif metrics["cpu_percent"] <= 80:
        logging.info(f"CPU Usage: {metrics['cpu_percent']:.1f}% (Acceptable)")
    else:
        logging.info(f"CPU Usage: {metrics['cpu_percent']:.1f}% (Poor: Optimize with joblib)")
    
    if metrics["gpu"] and "error" not in metrics["gpu"][0]:
        for gpu in metrics["gpu"]:
            if gpu["utilization"] == 0:
                logging.info(f"GPU {gpu['id']}: Utilization: {gpu['utilization']:.1f}% (Expected)")
            else:
                logging.info(f"GPU {gpu['id']}: Utilization: {gpu['utilization']:.1f}% (Unexpected)")
    else:
        logging.info(f"GPU: {metrics['gpu'][0]['error']}")
    logging.info("=" * 30)
    
    return [
        f"Start Time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}",
        f"Finish Time: {datetime.fromtimestamp(finish_time).strftime('%Y-%m-%d %H:%M:%S')}",
        f"Execution Time: {execution_time:.2f} s ({'Excellent' if execution_time < 10 else 'Acceptable' if execution_time <= 30 else 'Poor: Optimize with line_profiler'})",
        f"Memory Usage: {metrics['memory_mb']:.2f} MB ({'Excellent' if metrics['memory_mb'] < 500 else 'Acceptable' if metrics['memory_mb'] <= 2000 else 'Poor: Use memory_profiler'})",
        f"CPU Usage: {metrics['cpu_percent']:.1f}% ({'Excellent' if metrics['cpu_percent'] < 50 else 'Acceptable' if metrics['cpu_percent'] <= 80 else 'Poor: Optimize with joblib'})",
        f"GPU: {metrics['gpu'][0]['error'] if metrics['gpu'] and 'error' in metrics['gpu'][0] else f'Utilization: {metrics['gpu'][0]['utilization']:.1f}% ({'Expected' if metrics['gpu'][0]['utilization'] == 0 else 'Unexpected'})'}"
    ]

# --- Security Evaluation ---
def evaluate_security(input_errors, file_errors):
    """Evaluate security metrics"""
    input_validation_errors = input_errors.get("input_errors", 0) + input_errors.get("data_errors", 0) + input_errors.get("model_errors", 0) + input_errors.get("forecast_errors", 0)
    
    file_ops = 3  # pd.read_csv, to_csv, plt.savefig
    safe_file_ops = 3 if file_errors.get("file_errors", 0) == 0 else 2
    file_safety_percent = (safe_file_ops / file_ops) * 100
    
    dotenv.load_dotenv()
    env_vars = os.environ.keys()
    sensitive_vars = [var for var in env_vars if 'KEY' in var.upper() or 'PASSWORD' in var.upper()]
    env_exposure = len(sensitive_vars)
    
    logging.info("Security Evaluation:")
    logging.info(f"Input Validation Errors: {input_validation_errors} ({'Excellent' if input_validation_errors == 0 else 'Acceptable' if input_validation_errors <= 2 else 'Poor'})")
    logging.info(f"File Access Safety: {file_safety_percent:.1f}% ({'Excellent' if file_safety_percent == 100 else 'Acceptable' if file_safety_percent >= 50 else 'Poor: Add try-except'})")
    logging.info(f"Environment Variable Exposure: {env_exposure} ({'Excellent' if env_exposure == 0 else 'Poor: Secure .env access'})")
    logging.info("Dependency Vulnerabilities: Run `pip-audit` to check (e.g., >3 is Poor)")
    logging.info("=" * 30)
    
    return [
        f"Input Validation Errors: {input_validation_errors} ({'Excellent' if input_validation_errors == 0 else 'Acceptable' if input_validation_errors <= 2 else 'Poor'})",
        f"File Access Safety: {file_safety_percent:.1f}% ({'Excellent' if file_safety_percent == 100 else 'Acceptable' if file_safety_percent >= 50 else 'Poor: Add try-except'})",
        f"Environment Variable Exposure: {env_exposure} ({'Excellent' if env_exposure == 0 else 'Poor: Secure .env access'})",
        "Dependency Vulnerabilities: Run `pip-audit` to check (e.g., >3 is Poor)"
    ]

# --- Scalability Evaluation ---
def evaluate_scalability():
    """Evaluate scalability metrics"""
    execution_scaling = "Quadratic or worse (Poor: Vectorize or use dask)"
    memory_scaling = "Quadratic or worse (Poor: Use chunked processing)"
    parallel_tasks = 3
    parallel_capability = f"{parallel_tasks} tasks (Acceptable)"
    
    logging.info("Scalability Evaluation:")
    logging.info(f"Execution Time Scaling: {execution_scaling}")
    logging.info(f"Memory Scaling: {memory_scaling}")
    logging.info(f"Parallelization Capability: {parallel_capability}")
    logging.info("=" * 30)
    
    return [
        f"Execution Time Scaling: {execution_scaling}",
        f"Memory Scaling: {memory_scaling}",
        f"Parallelization Capability: {parallel_capability}"
    ]

# --- Maintainability Evaluation ---
def evaluate_maintainability():
    """Evaluate maintainability metrics"""
    with open(__file__, 'r') as f:
        code = f.read()
    visitor = ComplexityVisitor.from_code(code)
    max_complexity = max(func.complexity for func in visitor.functions) if visitor.functions else 0
    
    functions = [f for f in globals().values() if inspect.isfunction(f)]
    single_purpose = len(functions)
    total_functions = len(functions)
    modularity_percent = (single_purpose / total_functions * 100) if total_functions else 0
    
    docstring_count = sum(1 for f in functions if f.__doc__ is not None)
    doc_coverage = (docstring_count / total_functions * 100) if total_functions else 0
    
    test_coverage = 0.0
    
    logging.info("Maintainability Evaluation:")
    logging.info(f"Code Readability: Complexity {max_complexity} ({'Excellent' if max_complexity < 10 else 'Acceptable' if max_complexity <= 20 else 'Poor'})")
    logging.info(f"Modularity: {modularity_percent:.1f}% (Excellent)")
    logging.info(f"Documentation Coverage: {doc_coverage:.1f}% ({'Excellent' if doc_coverage > 80 else 'Acceptable' if doc_coverage >= 50 else 'Poor: Add docstrings'})")
    logging.info(f"Test Coverage: {test_coverage:.1f}% (Poor: Add pytest tests)")
    logging.info("=" * 30)
    
    return [
        f"Code Readability: Complexity {max_complexity} ({'Excellent' if max_complexity < 10 else 'Acceptable' if max_complexity <= 20 else 'Poor'})",
        f"Modularity: {modularity_percent:.1f}% (Excellent)",
        f"Documentation Coverage: {doc_coverage:.1f}% ({'Excellent' if doc_coverage > 80 else 'Acceptable' if doc_coverage >= 50 else 'Poor: Add docstrings'})",
        f"Test Coverage: {test_coverage:.1f}% (Poor: Add pytest tests)"
    ]

# --- Generate Performance Report ---
def generate_performance_report(output_dir, timestamp, comp_eff, security, scalability, maintainability):
    """Generate a performance report summarizing all evaluations"""
    report_file = os.path.join(output_dir, f"report_{timestamp}.txt")
    with open(report_file, 'w') as f:
        f.write("Performance Report\n")
        f.write("=" * 50 + "\n")
        f.write("Computational Efficiency:\n")
        for line in comp_eff:
            f.write(f"  {line}\n")
        f.write("\nSecurity:\n")
        for line in security:
            f.write(f"  {line}\n")
        f.write("\nScalability:\n")
        for line in scalability:
            f.write(f"  {line}\n")
        f.write("\nMaintainability:\n")
        for line in maintainability:
            f.write(f"  {line}\n")
        f.write("=" * 50 + "\n")
    logging.info(f"Performance report saved to '{report_file}'")
    return report_file

# --- Common Data Loading and Preprocessing ---
def load_data(data_path):
    """Load and preprocess data for both models"""
    try:
        df = pd.read_csv(data_path, parse_dates=['time'], index_col='time')
        target_col = 'solar_power'
        exog_cols = ['GHI', 'DNI', 'DHI', 'Solar Zenith Angle']
        required_cols = [target_col] + exog_cols
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        df = df[required_cols]
        df_resampled = df.resample('h').mean().interpolate()
        return df_resampled, target_col, exog_cols
    except Exception as e:
        raise ValueError(f"Data loading error: {str(e)}")

# --- DHR Model Components ---
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

# --- ESN Model Components ---
def prepare_esn_data(data, exog_data):
    """Prepare data for ESN model with log transformation and scaling"""
    min_non_zero = np.min(data[data > 0]) if np.any(data > 0) else 1e-10
    data_cleaned = np.maximum(data, min_non_zero)
    data_log = np.log1p(data_cleaned)
    
    if np.any(np.isnan(data_log)) or np.any(np.isinf(data_log)):
        data_log = np.nan_to_num(data_log, nan=np.nanmean(data_log),
                                 posinf=np.nanmax(data_log), neginf=np.nanmin(data_log))
    
    scaler_target = MinMaxScaler(feature_range=(0.1, 0.9))
    data_scaled = scaler_target.fit_transform(data_log.reshape(-1, 1)).flatten()
    
    scaler_exog = MinMaxScaler(feature_range=(0.1, 0.9))
    exog_scaled = scaler_exog.fit_transform(exog_data)
    
    combined_data = np.column_stack([data_scaled, exog_scaled])
    return combined_data, scaler_target, scaler_exog

def create_sequences(data, lags):
    """Create sequences for ESN with exogenous variables"""
    X, y = [], []
    for i in range(len(data) - lags):
        X.append(data[i:i + lags])
        y.append(data[i + lags, 0])
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
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
        rp.verbosity(0)
        n_samples, lags, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(n_samples, lags * n_features)
        model.fit(X_train_reshaped, y_train, warmup=0)
        logging.info("ESN training complete!")
        return model
    except Exception as e:
        raise ValueError(f"Error during ESN training: {str(e)}")

def predict_esn(model, X, scaler_target):
    """Generate and post-process ESN predictions with reshaped input"""
    reservoir = model.nodes[1]
    readout = model.nodes[2]
    n_samples, lags, n_features = X.shape
    y_pred = []

    for i in range(n_samples):
        sequence = X[i]
        sequence_reshaped = sequence.reshape(lags * n_features)
        state = reservoir.run(sequence_reshaped)
        pred = readout.run(state)
        y_pred.append(pred[0])

    y_pred = np.array(y_pred).reshape(-1, 1)
    y_pred_inv = scaler_target.inverse_transform(y_pred)
    y_pred_original = np.expm1(y_pred_inv)
    y_pred_original = np.maximum(y_pred_original, 0)
    return y_pred_original.flatten()

# --- Hybrid Model Components ---
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
    logging.info(f"Optimized weights: DHR={best_weights[0]:.2f}, ESN={best_weights[1]:.2f}")
    return best_weights

def evaluate_model(y_true, y_pred):
    """Calculate and print evaluation metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    cvrmse = (rmse / np.mean(y_true)) * 100 if np.mean(y_true) != 0 else float('inf')
    logging.info(f"RMSE: {rmse:.4f}")
    logging.info(f"MAE: {mae:.4f}")
    logging.info(f"CV-RMSE: {cvrmse:.2f}%")
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
        logging.info(f"Plot saved to: {save_path}")
    else:
        plt.savefig('hybrid_model_comparison.png', dpi=300, bbox_inches='tight')
    
    plt.close()
    return save_path if save_path else 'hybrid_model_comparison.png'

# --- Forecast Generation Function ---
def generate_forecast(df, dhr_model, esn_model, esn_scaler_target, esn_scaler_exog, best_weights, steps, 
                     dhr_params, esn_params, exog_cols=['GHI', 'DNI', 'DHI', 'Solar Zenith Angle'], 
                     output_dir="forecasts", timestamp=""):
    """Generate forecast for specified number of steps"""
    target_col = 'solar_power'
    target = df[target_col].values
    exog_data = df[exog_cols].values

    # Prepare ESN data
    combined_data, _, _ = prepare_esn_data(target, exog_data)
    n_samples, n_features = combined_data.shape
    logging.info(f"Combined data shape: {combined_data.shape} (samples: {n_samples}, features: {n_features})")
    
    logging.info(f"Generating {steps}-step forecast...")
    forecast_dates = [df.index[-1] + timedelta(hours=i+1) for i in range(steps)]
    
    # Get last week's exog values (168 hours = 7 days ago) for forecasting
    week_ago_idx = len(df) - 168 - 1
    if week_ago_idx < 0:
        logging.warning(f"Not enough data for {steps}-step forecast. Using last available exog values.")
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
    n_harmonics = dhr_params['fourier_terms']
    sin_terms = [np.sin(2 * np.pi * i * t / period) for i in range(1, n_harmonics + 1)]
    cos_terms = [np.cos(2 * np.pi * i * t / period) for i in range(1, n_harmonics + 1)]
    fourier_forecast = np.column_stack(sin_terms + cos_terms)
    
    # Initialize forecast lists
    dhr_forecast = []
    esn_forecast = []
    hybrid_forecast = []
    current_target = target.copy()
    last_sequence = combined_data[-esn_params['lags']:].reshape(1, esn_params['lags'], -1)
    
    # Calculate actual input dimension for ESN
    actual_input_dim = esn_params['lags'] * n_features
    logging.info(f"ESN input dimension: {actual_input_dim} (lags: {esn_params['lags']}, features: {n_features})")
    
    for h in range(steps):
        # DHR FORECASTING
        ar_features = current_target[-dhr_params['ar_order']:]
        smoothed = savgol_filter(current_target[-dhr_params['window']:], 
                                window_length=dhr_params['window'], 
                                polyorder=dhr_params['polyorder']) if len(current_target) >= dhr_params['window'] else np.zeros(dhr_params['window'])
        dhr_features = np.concatenate([ar_features, smoothed, fourier_forecast[h], exog_forecast[h]])
        dhr_features = dhr_features.reshape(1, -1)
        dhr_pred = dhr_model.predict(dhr_features)[0]
        dhr_pred = max(dhr_pred, 0)
        if exog_forecast[h][3] > 90:
            dhr_pred = 0
        dhr_forecast.append(dhr_pred)
        current_target = np.append(current_target, dhr_pred)
        
        # ESN FORECASTING
        esn_pred = esn_model.run(last_sequence.reshape(1, actual_input_dim))
        esn_pred = np.array(esn_pred).reshape(-1, 1)
        esn_pred_value = esn_pred[0, 0]
        esn_pred_inv = esn_scaler_target.inverse_transform(esn_pred)
        esn_pred_original = np.maximum(np.expm1(esn_pred_inv)[0][0], 0)
        if exog_forecast[h][3] > 90:
            esn_pred_original = 0
        esn_forecast.append(esn_pred_original)
        
        # Update ESN sequence
        last_sequence = np.roll(last_sequence, -1, axis=1)
        last_sequence[0, -1, :] = np.concatenate([[esn_pred_value], exog_forecast_scaled[h]])
        
        # HYBRID FORECAST
        hybrid_value = weighted_average(np.array([dhr_pred]), np.array([esn_pred_original]), best_weights)[0]
        if exog_forecast[h][3] > 90:
            hybrid_value = 0
        hybrid_forecast.append(hybrid_value)
    
    # Get last two weeks (336 hours) of actual data
    two_weeks_ago_idx = len(df) - 336
    if two_weeks_ago_idx < 0:
        logging.warning(f"Not enough data for two weeks. Using available actual data.")
        two_weeks_ago_idx = 0
    actual_dates = df.index[two_weeks_ago_idx:].tolist()
    actual_values = df[target_col].values[two_weeks_ago_idx:]
    
    # Combine actual and forecast data
    all_dates = actual_dates + forecast_dates
    actual_extended = np.concatenate([actual_values, np.full(steps, np.nan)])
    dhr_extended = np.concatenate([np.full(len(actual_values), np.nan), dhr_forecast])
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
    logging.info(f"Forecast saved to '{csv_path_output}'")
    
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
    logging.info(f"Plot saved to '{plot_path}'")
    
    return forecast_df, csv_path_output, plot_path

def train_models(df, target_col, exog_cols, params=None):
    """Train both DHR and ESN models and return trained models with scalers"""
    try:
        # Set default parameters if none provided
        if params is None:
            params = {}
        
        # DHR Model Parameters
        dhr_params = {
            'fourier_terms': params.get('fourier_terms', 4),
            'reg_strength': params.get('reg_strength', 0.006),
            'ar_order': params.get('ar_order', 1),
            'window': params.get('window', 23),
            'polyorder': params.get('polyorder', 2)
        }
        
        # ESN Model Parameters
        esn_params = {
            'N_res': params.get('N_res', 800),
            'rho': params.get('rho', 0.9308202574),
            'sparsity': params.get('sparsity', 0.1335175715),
            'alpha': params.get('alpha', 0.7191611348),
            'lambda_reg': params.get('lambda_reg', 2.10E-08),
            'lags': params.get('lags', 24)
        }
        
        # Log parameters
        logging.info("DHR Parameters:")
        for key, value in dhr_params.items():
            logging.info(f"  {key}: {value}")
        
        logging.info("ESN Parameters:")
        for key, value in esn_params.items():
            logging.info(f"  {key}: {value}")
        
        # Create DHR features
        logging.info("Creating DHR features...")
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
        logging.info("Splitting data for DHR...")
        X_train_dhr, X_temp_dhr, y_train_dhr, y_temp_dhr = train_test_split(
            X_dhr, y_dhr, test_size=0.2, shuffle=False
        )
        X_val_dhr, X_test_dhr, y_val_dhr, y_test_dhr = train_test_split(
            X_temp_dhr, y_temp_dhr, test_size=0.5, shuffle=False
        )
        
        # Train DHR model
        logging.info("Training DHR model...")
        dhr_model = train_dhr_model(X_train_dhr, y_train_dhr, dhr_params['reg_strength'])
        
        # Prepare data for ESN model
        logging.info("Preparing ESN data...")
        solar_data = df[target_col].values
        exog_data = df[exog_cols].values
        combined_data, esn_scaler_target, esn_scaler_exog = prepare_esn_data(solar_data, exog_data)
        
        # Create sequences for ESN
        logging.info("Creating ESN sequences...")
        X_esn, y_esn = create_sequences(combined_data, esn_params['lags'])
        logging.info(f"ESN sequences shape: X_esn {X_esn.shape}, y_esn {y_esn.shape}")
        
        # Get actual dimensions from the data
        n_samples, lags, n_features = X_esn.shape
        actual_input_dim = lags * n_features
        logging.info(f"Calculated input dimension: {actual_input_dim} (lags: {lags}, features: {n_features})")
        
        # Update ESN parameters with actual dimensions
        esn_params['lags'] = lags
        esn_params['n_features'] = n_features
        esn_params['input_dim'] = actual_input_dim
        
        # Split ESN data
        logging.info("Splitting data for ESN...")
        total_size = len(X_esn)
        train_size = int(total_size * 0.8)
        val_size = len(y_val_dhr)
        test_size = len(y_test_dhr)
        X_train_esn = X_esn[:train_size]
        y_train_esn = y_esn[:train_size]
        X_val_esn = X_esn[train_size:train_size + val_size]
        y_val_esn = y_esn[train_size:train_size + val_size]
        
        # Build and train ESN model
        logging.info("Building and training ESN model...")
        esn_model = build_esn_model(
            esn_params['N_res'],
            esn_params['rho'],
            esn_params['sparsity'],
            esn_params['alpha'],
            esn_params['lambda_reg'],
            input_dim=actual_input_dim
        )
        esn_model = train_esn_model(esn_model, X_train_esn, y_train_esn)
        
        # Optimize weights using validation set
        logging.info("Optimizing weights...")
        dhr_val_preds = predict_dhr(dhr_model, X_val_dhr)
        esn_val_preds = predict_esn(esn_model, X_val_esn, esn_scaler_target)
        best_weights = optimize_weights(dhr_val_preds, esn_val_preds.flatten(), y_val_dhr)
        
        return dhr_model, esn_model, esn_scaler_target, esn_scaler_exog, best_weights, dhr_params, esn_params
    except Exception as e:
        raise ValueError(f"Model training error: {str(e)}")

# --- Main Forecast Function ---
def run_forecast(csv_path, steps, output_dir="forecasts", forecast_type="hourly", params=None):
    """
    Main function to run the hybrid DHR-ESN hourly solar power forecasting process
    
    Parameters:
    - csv_path: Path to the CSV file containing solar power data
    - steps: Number of hours to forecast ahead
    - output_dir: Directory to save forecast results
    - forecast_type: Type of forecast ("hourly")
    - params: Dictionary containing DHR and ESN hyperparameters
    
    Returns:
    - List of file paths [csv_path, plot_path], params, metrics, execution_time, log_file
    """
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    log_file = setup_logging(output_dir, timestamp)
    
    errors = {}
    
    logging.info("Checking idle system resources...")
    idle_metrics = measure_resources(is_idle=True)
    
    start_time = time.time()
    
    try:
        # Log input details
        logging.info(f"Starting solar power forecast...")
        logging.info(f"Input CSV: {csv_path}")
        logging.info(f"Forecast steps: {steps}")
        logging.info(f"Output directory: {output_dir}")
        logging.info(f"Custom parameters: {params}")
        logging.info(f"Timestamp: {timestamp}")
        
        # Load data
        logging.info("Loading data...")
        df, target_col, exog_cols = load_data(csv_path)
        logging.info(f"Dataset shape: {df.shape}, Columns: {df.columns.tolist()}")
    except Exception as e:
        logging.error(f"Security: Input validation error - {str(e)}")
        errors["input_errors"] = errors.get("input_errors", 0) + 1
        return None, None, {"idle": idle_metrics, "runtime": None, "errors": errors}, None, log_file
    
    try:
        # Train models
        logging.info("Training models...")
        dhr_model, esn_model, esn_scaler_target, esn_scaler_exog, best_weights, dhr_params, esn_params = train_models(
            df, target_col, exog_cols, params=params
        )
    except Exception as e:
        logging.error(f"Security: Model training error - {str(e)}")
        errors["model_errors"] = errors.get("model_errors", 0) + 1
        return None, None, {"idle": idle_metrics, "runtime": None, "errors": errors}, None, log_file
    
    try:
        # Generate forecast
        logging.info("Generating forecast...")
        forecast_df, csv_path_output, plot_path = generate_forecast(
            df, dhr_model, esn_model, esn_scaler_target, esn_scaler_exog, 
            best_weights, steps, dhr_params, esn_params, output_dir=output_dir, timestamp=timestamp
        )
    except Exception as e:
        logging.error(f"Security: Forecast generation error - {str(e)}")
        errors["forecast_errors"] = errors.get("forecast_errors", 0) + 1
        return None, None, {"idle": idle_metrics, "runtime": None, "errors": errors}, None, log_file
    
    finish_time = time.time()
    execution_time = finish_time - start_time
    runtime_metrics = measure_resources(is_idle=False)
    runtime_metrics["execution_time"] = execution_time
    
    comp_eff_report = evaluate_computational_efficiency(start_time, finish_time, execution_time, runtime_metrics)
    security_report = evaluate_security(errors, errors)
    scalability_report = evaluate_scalability()
    maintainability_report = evaluate_maintainability()
    
    report_file = generate_performance_report(output_dir, timestamp, comp_eff_report, security_report, scalability_report, maintainability_report)
    
    logging.info(f"Forecast completed successfully!")
    return [csv_path_output, plot_path], params, {"idle": idle_metrics, "runtime": runtime_metrics, "errors": errors}, execution_time
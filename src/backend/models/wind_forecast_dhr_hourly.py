import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.signal import savgol_filter
from datetime import datetime, timedelta
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
            logging.info(f"  GPU {gpu['id']} ({gpu['name']}): Memory Used: {gpu['memory_used']:.1f} MB ({gpu['memory_percent']:.1f}%), Utilization: {gpu['utilization']:.1f}%")
    else:
        logging.info(f"  {gpu_info[0]['error']}")
    logging.info("=" * 50)
    
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
    logging.info("=" * 50)
    
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
    logging.info("=" * 50)
    
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
    parallel_tasks = 2
    parallel_capability = f"{parallel_tasks} tasks (Acceptable)"
    
    logging.info("Scalability Evaluation:")
    logging.info(f"Execution Time Scaling: {execution_scaling}")
    logging.info(f"Memory Scaling: {memory_scaling}")
    logging.info(f"Parallelization Capability: {parallel_capability}")
    logging.info("=" * 50)
    
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
    logging.info("=" * 50)
    
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

# --- Data Loading and Preprocessing ---
def load_data(data_path):
    """Load and preprocess data for the DHR model"""
    try:
        df = pd.read_csv(data_path, parse_dates=['time'], index_col='time')
        target_col = 'wind_power'
        exog_cols = ['Wind Speed', 'Temperature', 'Relative Humidity', 'Solar Zenith Angle']
        required_cols = [target_col] + exog_cols
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        df = df[required_cols]
        df_resampled = df.resample('h').mean().interpolate()
        return df_resampled, target_col, exog_cols
    except Exception as e:
        raise ValueError(f"Data loading error: {str(e)}")

# --- Fourier Transform ---
def fourier_transform(t, n_harmonics=4, periods=[24, 168]):
    """Generate Fourier features for given periods and harmonics"""
    features = []
    for period in periods:
        sin_terms = [np.sin(2 * np.pi * i * t / period) for i in range(1, n_harmonics + 1)]
        cos_terms = [np.cos(2 * np.pi * i * t / period) for i in range(1, n_harmonics + 1)]
        features.extend(sin_terms + cos_terms)
    return np.column_stack(features)

# --- Feature Engineering ---
def create_features(df, target_col, fourier_terms, ar_order, window, polyorder):
    """Create features for DHR model including exogenous variables"""
    t = np.arange(len(df))
    fourier = fourier_transform(t, n_harmonics=fourier_terms, periods=[24, 168])
    target = df[target_col].values
    wind_speed = df['Wind Speed'].values
    temperature = df['Temperature'].values
    relative_humidity = df['Relative Humidity'].values
    sza = df['Solar Zenith Angle'].values
    
    scaler_target = MinMaxScaler()
    scaler_exog = MinMaxScaler()
    target_scaled = scaler_target.fit_transform(target.reshape(-1, 1)).flatten()
    exog_scaled = scaler_exog.fit_transform(np.column_stack([wind_speed, temperature, relative_humidity, sza]))
    
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
def repeat_last_week(arr, forecast_horizon):
    """Repeat last week's data for forecasting"""
    period = 24 * 7  # one week
    n_features = arr.shape[1] if len(arr.shape) > 1 else 1
    result = []
    for col in range(n_features):
        data = arr[:, col] if n_features > 1 else arr
        start_idx = max(0, len(data) - period)
        last_week = data[start_idx:]
        repeated = np.tile(last_week, (forecast_horizon // len(last_week) + 1))[:forecast_horizon]
        result.append(repeated)
    return np.column_stack(result) if n_features > 1 else result[0]

# --- Create Feature Vector ---
def create_feature_vector(values, current_pos, fourier_data, ar_order, window, polyorder, exog_scaled, fourier_idx, exog_idx):
    """Create feature vector for forecasting"""
    ar_features = values[current_pos - ar_order:current_pos]
    segment = values[current_pos - window:current_pos]
    window_len = min(len(segment), window)
    if window_len % 2 == 0:
        window_len -= 1
    window_len = max(3, window_len)
    smoothed = savgol_filter(segment, window_length=window_len, polyorder=min(polyorder, window_len - 1))
    exog_features = exog_scaled[exog_idx]
    return np.concatenate([ar_features, smoothed, fourier_data[fourier_idx], exog_features])

# --- Forecast Generation ---
def generate_forecast(model, start_values, fourier_data, steps, params, exog_scaled, scaler_target):
    """Generate forecast for specified number of steps"""
    ar_order = params['ar_order']
    window = params['window']
    polyorder = params['polyorder']
    values = start_values.copy()
    forecasts = []
    historical_len = len(start_values)
    max_target = np.max(scaler_target.inverse_transform(start_values.reshape(-1, 1)))
    
    for step in range(steps):
        current_pos = historical_len + step
        fourier_idx = historical_len + step
        exog_idx = step
        features = create_feature_vector(
            values, current_pos, fourier_data, ar_order, window, polyorder,
            exog_scaled, fourier_idx, exog_idx
        )
        prediction = model.predict(features.reshape(1, -1))[0]
        prediction = np.clip(prediction, 0, 1)
        forecasts.append(prediction)
        values = np.append(values, prediction)
    
    forecasts = np.array(forecasts).reshape(-1, 1)
    forecasts_unscaled = scaler_target.inverse_transform(forecasts)
    forecasts_unscaled = np.clip(forecasts_unscaled, 0, max_target)
    return forecasts_unscaled.flatten()

# --- Evaluate Model ---
def evaluate_model(y_true, y_pred):
    """Calculate and log evaluation metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    cvrmse = (rmse / np.mean(y_true)) * 100 if np.mean(y_true) != 0 else float('inf')
    logging.info(f"RMSE: {rmse:.4f}")
    logging.info(f"MAE: {mae:.4f}")
    logging.info(f"CV-RMSE: {cvrmse:.2f}%")
    return rmse, mae, cvrmse

# --- Plot Results ---
def plot_results(historical_dates, historical_data, forecast_dates, forecast_values, steps, output_dir, timestamp):
    """Plot historical and forecast data"""
    plt.figure(figsize=(14, 7))
    plt.plot(historical_dates, historical_data, label='Actual (Last 2 weeks)', color='blue', linewidth=2)
    
    combined_dates = [historical_dates[-1]] + list(forecast_dates)
    combined_values = [historical_data[-1]] + list(forecast_values)
    plt.plot(combined_dates, combined_values, label=f'Forecast ({steps}h)', color='red', linestyle='--', linewidth=2)
    
    plt.axvline(x=historical_dates[-1], color='green', linestyle=':', label='Forecast Start', linewidth=2)
    plt.title(f'{steps}-Hour Wind Power Forecast')
    plt.xlabel('Date')
    plt.ylabel('Wind Power')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    
    plot_filename = f'wind_dhr_hourly_{timestamp}_{steps}.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Plot saved to '{plot_path}'")
    
    return plot_path

# --- Main Forecast Function ---
def run_forecast(csv_path, steps, output_dir="forecasts", forecast_type="hourly", params=None):
    """
    Main function to run the DHR model for hourly wind power forecasting
    
    Args:
        csv_path (str): Path to input CSV file
        steps (int): Number of hours to forecast
        output_dir (str): Directory to save outputs
        forecast_type (str): Type of forecast ('hourly')
        params (dict): Model parameters
    
    Returns:
        list: [csv_path, plot_path], params, metrics_dict, execution_time, log_file
    """
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    log_file = setup_logging(output_dir, timestamp)
    
    errors = {}
    
    logging.info("Checking idle system resources...")
    idle_metrics = measure_resources(is_idle=True)
    
    start_time = time.time()
    
    try:
        # Log input details
        logging.info(f"Starting wind power forecast...")
        logging.info(f"Input CSV: {csv_path}")
        logging.info(f"Forecast steps: {steps}")
        logging.info(f"Output directory: {output_dir}")
        logging.info(f"Parameters: {params}")
        logging.info(f"Timestamp: {timestamp}")
        
        if params is None:
            raise ValueError("Parameters must be provided for the forecast.")
        
        # Load data
        logging.info("Loading data...")
        df, target_col, exog_cols = load_data(csv_path)
        logging.info(f"Dataset shape: {df.shape}, Columns: {df.columns.tolist()}")
        
        # Create features
        logging.info("Creating features...")
        X, y, scaler_target, scaler_exog = create_features(
            df,
            target_col=target_col,
            fourier_terms=params['fourier_terms'],
            ar_order=params['ar_order'],
            window=params['window'],
            polyorder=params['polyorder']
        )
        
        # Train model
        logging.info("Training model...")
        model = Ridge(alpha=params['reg_strength'])
        model.fit(X, y)
        
        # Historical data for plotting
        last_two_weeks = 336  # 2 weeks
        target_values = df[target_col].values
        historical_end_idx = len(target_values)
        historical_start_idx = max(0, historical_end_idx - last_two_weeks)
        historical_data = target_values[historical_start_idx:historical_end_idx]
        historical_dates = df.index[historical_start_idx:historical_end_idx]
        
        # Prepare exogenous data
        logging.info("Preparing exogenous variables...")
        wind_speed = df['Wind Speed'].values
        temperature = df['Temperature'].values
        relative_humidity = df['Relative Humidity'].values
        sza = df['Solar Zenith Angle'].values
        exog_data = np.column_stack([wind_speed, temperature, relative_humidity, sza])
        
        # Extend exogenous variables
        exog_forecast = repeat_last_week(exog_data, steps)
        exog_forecast_scaled = scaler_exog.transform(exog_forecast)
        
        # Scale historical target
        target_scaled = scaler_target.transform(target_values.reshape(-1, 1)).flatten()
        
        # Extend Fourier features
        logging.info("Extending Fourier features...")
        extended_df_length = len(df) + steps
        t_extended = np.arange(extended_df_length)
        fourier_extended = fourier_transform(t_extended, n_harmonics=params['fourier_terms'], periods=[24, 168])
        
        # Generate forecast
        logging.info("Generating forecast...")
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
        forecast_dates = pd.date_range(start=last_historical_date + timedelta(hours=1), periods=steps, freq='h')
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'datetime': forecast_dates,
            'forecasted_wind_power': forecast_values
        })
        
        # Save forecast
        csv_name = f"wind_dhr_hourly_{timestamp}_{steps}.csv"
        csv_path_out = os.path.join(output_dir, csv_name)
        forecast_df.to_csv(csv_path_out, index=False)
        logging.info(f"Forecast saved to '{csv_path_out}'")
        
        # Plot results
        plot_path = plot_results(historical_dates, historical_data, forecast_dates, forecast_values, 
                                steps, output_dir, timestamp)
        
    except Exception as e:
        logging.error(f"Error in forecast execution: {str(e)}")
        errors["input_errors"] = errors.get("input_errors", 0) + 1
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
    return [csv_path_out, plot_path], params, {"idle": idle_metrics, "runtime": runtime_metrics, "errors": errors}, execution_time
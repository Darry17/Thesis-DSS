import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from scipy.signal import savgol_filter
from datetime import timedelta, datetime
import psutil
import GPUtil
import time
import logging

# Setup logging with console and file output
def setup_logging(output_dir, timestamp):
    log_file = os.path.join(output_dir, f"forecast_log_{timestamp}.log")
    # Clear any existing handlers to avoid duplicate logs
    for handler in logging.getLogger().handlers[:]:
        logging.getLogger().removeHandler(handler)
    # Set up new logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S,%f'[:-3],  # Milliseconds precision
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    logging.info(f"Logs will be saved to '{log_file}'")

# --- Resource Monitoring Function ---
def measure_resources(is_idle=True):
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

# --- Evaluate Computational Efficiency ---
def evaluate_computational_efficiency(start_time, finish_time, execution_time, metrics):
    logging.info("Computational Efficiency Evaluation:")
    logging.info(f"Start Time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S,%f'[:-3])}")
    logging.info(f"Finish Time: {datetime.fromtimestamp(finish_time).strftime('%Y-%m-%d %H:%M:%S,%f'[:-3])}")
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

# --- Your Existing Functions (unchanged) ---
def fourier_transform(t, n_harmonics=4, periods=[7, 30]):
    features = []
    for period in periods:
        sin_terms = [np.sin(2 * np.pi * i * t / period) for i in range(1, n_harmonics + 1)]
        cos_terms = [np.cos(2 * np.pi * i * t / period) for i in range(1, n_harmonics + 1)]
        features.extend(sin_terms + cos_terms)
    return np.column_stack(features)

def create_features(df, target_col, fourier_terms, ar_order, window, polyorder):
    t = np.arange(len(df))
    fourier = fourier_transform(t, n_harmonics=fourier_terms, periods=[7, 30])
    target = df[target_col].values
    ghi = df['GHI'].values
    dni = df['DNI'].values
    dhi = df['DHI'].values
    X, y = [], []
    for i in range(max(ar_order, window), len(target)):
        ar_features = target[i - ar_order:i]
        smoothed = savgol_filter(target[i - window:i], window_length=window, polyorder=polyorder)
        exog_features = np.array([ghi[i], dni[i], dhi[i]])
        features = np.concatenate([ar_features, smoothed, fourier[i], exog_features])
        X.append(features)
        y.append(target[i])
    return np.array(X), np.array(y)

def repeat_last_month(arr, forecast_horizon):
    period = 30
    repeated = np.tile(arr[-period:], int(np.ceil(forecast_horizon/period)))[:forecast_horizon]
    return np.concatenate([arr, repeated])

def create_feature_vector(values, current_pos, fourier_data, ar_order, window, polyorder, ghi, dni, dhi):
    ar_features = values[current_pos - ar_order:current_pos]
    segment = values[current_pos - window:current_pos]
    window_len = min(len(segment), window)
    if window_len % 2 == 0:
        window_len -= 1
    window_len = max(3, window_len)
    smoothed = savgol_filter(segment, window_length=window_len, polyorder=min(polyorder, window_len - 1))
    exog_features = np.array([ghi[current_pos], dni[current_pos], dhi[current_pos]])
    return np.concatenate([ar_features, smoothed, fourier_data[current_pos], exog_features])

def generate_forecast(model, start_values, fourier_data, steps, params, ghi, dni, dhi):
    ar_order = params['ar_order']
    window = params['window']
    polyorder = params['polyorder']
    values = start_values.copy()
    forecasts = []
    current_pos = len(values) - 1
    for _ in range(steps):
        features = create_feature_vector(
            values, current_pos, fourier_data, ar_order, window, polyorder,
            ghi, dni, dhi
        )
        prediction = model.predict(features.reshape(1, -1))[0]
        prediction = max(0, prediction)
        forecasts.append(prediction)
        values = np.append(values, prediction)
        current_pos += 1
    return np.array(forecasts)

# --- Modified run_forecast ---
def run_forecast(csv_path, steps, output_dir="forecasts", forecast_type="daily", params=None):
    # Setup timestamp for file naming
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    
    # Initialize logging
    setup_logging(output_dir, timestamp)
    
    # Measure idle resources
    logging.info("Checking idle system resources...")
    idle_metrics = measure_resources(is_idle=True)
    
    # Start timing
    start_time = time.perf_counter()
    
    # Input validation
    try:
        df = pd.read_csv(csv_path, parse_dates=['time'], index_col='time')
        required_cols = ['solar_power', 'GHI', 'DNI', 'DHI']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        if params is None:
            raise ValueError("Parameters must be provided")
    except Exception as e:
        logging.error(f"Security: Input validation error - {str(e)}")
        return None, None, {"input_errors": 1}, None
    
    # File access safety
    try:
        df = df.resample('d').mean().interpolate()
    except Exception as e:
        logging.error(f"Security: File processing error - {str(e)}")
        return None, None, {"file_errors": 1}, None
    
    # Create features
    try:
        X, y = create_features(df, target_col='solar_power',
                              fourier_terms=params['fourier_terms'],
                              ar_order=params['ar_order'],
                              window=params['window'],
                              polyorder=params['polyorder'])
    except Exception as e:
        logging.error(f"Feature creation error - {str(e)}")
        return None, None, {"feature_errors": 1}, None
    
    # Train model
    try:
        model = Ridge(alpha=params['reg_strength'])
        model.fit(X, y)
    except Exception as e:
        logging.error(f"Model training error - {str(e)}")
        return None, None, {"model_errors": 1}, None
    
    # Historical data
    last_month = 30
    target_values = df['solar_power'].values
    historical_end_idx = len(target_values)
    historical_start_idx = max(0, historical_end_idx - last_month)
    historical_data = target_values[historical_start_idx:historical_end_idx]
    historical_dates = df.index[historical_start_idx:historical_end_idx]
    
    # Extend exogenous variables
    ghi = df['GHI'].values
    dni = df['DNI'].values
    dhi = df['DHI'].values
    ghi_ext = repeat_last_month(ghi, steps)
    dni_ext = repeat_last_month(dni, steps)
    dhi_ext = repeat_last_month(dhi, steps)
    
    # Extend Fourier features
    extended_df_length = len(df) + steps
    t_extended = np.arange(extended_df_length)
    fourier_extended = fourier_transform(t_extended, n_harmonics=params['fourier_terms'], periods=[7, 30])
    
    # Generate forecast
    try:
        forecast_values = generate_forecast(
            model,
            target_values,
            fourier_extended,
            steps,
            params,
            ghi_ext,
            dni_ext,
            dhi_ext
        )
    except Exception as e:
        logging.error(f"Forecast generation error - {str(e)}")
        return None, None, {"forecast_errors": 1}, None
    
    # End timing and measure runtime resources
    finish_time = time.perf_counter()
    execution_time = finish_time - start_time
    runtime_metrics = measure_resources(is_idle=False)
    runtime_metrics["execution_time"] = execution_time
    
    # Evaluate computational efficiency
    evaluate_computational_efficiency(start_time, finish_time, execution_time, runtime_metrics)
    
    # Save forecast
    try:
        os.makedirs(output_dir, exist_ok=True)
        csv_name = f"solar_dhr_daily_{timestamp}_{steps}.csv"
        csv_path = os.path.join(output_dir, csv_name)
        forecast_dates = pd.date_range(start=historical_dates[-1], periods=steps+1, freq='D')[1:]
        forecast_df = pd.DataFrame({
            'datetime': forecast_dates,
            'forecasted_solar_power': forecast_values,
            'execution_time_seconds': [execution_time] * len(forecast_dates),
            'start_time': [datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S,%f'[:-3])] * len(forecast_dates),
            'finish_time': [datetime.fromtimestamp(finish_time).strftime('%Y-%m-%d %H:%M:%S,%f'[:-3])] * len(forecast_dates)
        })
        forecast_df.to_csv(csv_path, index=False)
        logging.info(f"Forecast saved to '{csv_path}'")
    except Exception as e:
        logging.error(f"Security: File save error - {str(e)}")
        return None, None, {"file_errors": 1}, None
    
    # Plotting
    try:
        plt.figure(figsize=(14, 6))
        plt.plot(historical_dates, historical_data, label=f'Actual (Last {len(historical_data)} days)', color='blue', linewidth=2)
        plt.plot(forecast_dates, forecast_values, label=f'Forecast ({steps}d)', color='red', linestyle='--', linewidth=2)
        plt.plot([historical_dates[-1], forecast_dates[0]], 
                 [historical_data[-1], forecast_values[0]], 
                 color='red', linestyle='--', linewidth=2)
        plt.axvline(x=historical_dates[-1], color='green', linestyle=':', label='Forecast Start', linewidth=2)
        plt.title(f'Solar Power Forecast - {steps} Day{"s" if steps > 1 else ""}')
        plt.xlabel('Time')
        plt.ylabel('Solar Power')
        plt.text(0.02, 0.98, f'Execution Time: {execution_time:.2f} s\nStart: {datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")}\nFinish: {datetime.fromtimestamp(finish_time).strftime("%Y-%m-%d %H:%M:%S")}', 
                 transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, f"solar_dhr_daily_{timestamp}_{steps}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Plot saved to '{plot_path}'")
    except Exception as e:
        logging.error(f"Security: Plot save error - {str(e)}")
        return None, None, {"file_errors": 1}, None
    
    return [csv_path, plot_path], params, {"idle": idle_metrics, "runtime": runtime_metrics}, execution_time
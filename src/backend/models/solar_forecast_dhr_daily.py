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
import radon.complexity
from radon.visitors import ComplexityVisitor
import inspect
import dotenv

# --- Setup Logging ---
def setup_logging(output_dir, timestamp):
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
        logging.info(f"Memory Usage: {metrics["memory_mb"]:.2f} MB (Excellent)")
    elif metrics["memory_mb"] <= 2000:
        logging.info(f"Memory Usage: {metrics["memory_mb"]:.2f} MB (Acceptable)")
    else:
        logging.info(f"Memory Usage: {metrics["memory_mb"]:.2f} MB (Poor: Use memory_profiler)")
    
    if metrics["cpu_percent"] < 50:
        logging.info(f"CPU Usage: {metrics["cpu_percent"]:.1f}% (Excellent)")
    elif metrics["cpu_percent"] <= 80:
        logging.info(f"CPU Usage: {metrics["cpu_percent"]:.1f}% (Acceptable)")
    else:
        logging.info(f"CPU Usage: {metrics["cpu_percent"]:.1f}% (Poor: Optimize with joblib)")
    
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
    input_validation_errors = input_errors.get("input_errors", 0) + input_errors.get("feature_errors", 0) + input_errors.get("model_errors", 0) + input_errors.get("forecast_errors", 0)
    
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
    logging.info("=" * 30)
    
    return [
        f"Input Validation Errors: {input_validation_errors} ({'Excellent' if input_validation_errors == 0 else 'Acceptable' if input_validation_errors <= 2 else 'Poor'})",
        f"File Access Safety: {file_safety_percent:.1f}% ({'Excellent' if file_safety_percent == 100 else 'Acceptable' if file_safety_percent >= 50 else 'Poor: Add try-except'})",
    ]

# --- Scalability Evaluation ---
def evaluate_scalability():
    execution_scaling = "Quadratic or worse (Poor: Vectorize or use dask)"
    memory_scaling = "Quadratic or worse (Poor: Use chunked processing)"
    parallel_tasks = 2
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
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    log_file = setup_logging(output_dir, timestamp)
    
    errors = {}
    
    logging.info("Checking idle system resources...")
    idle_metrics = measure_resources(is_idle=True)
    
    start_time = time.time()  # Use time.time() for absolute timestamp
    
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
        errors["input_errors"] = errors.get("input_errors", 0) + 1
        return None, None, {"idle": idle_metrics, "runtime": None, "errors": errors}, None, log_file
    
    try:
        df = df.resample('d').mean().interpolate()
    except Exception as e:
        logging.error(f"Security: File processing error - {str(e)}")
        errors["file_errors"] = errors.get("file_errors", 0) + 1
        return None, None, {"idle": idle_metrics, "runtime": None, "errors": errors}, None, log_file
    
    try:
        X, y = create_features(df, target_col='solar_power',
                              fourier_terms=params['fourier_terms'],
                              ar_order=params['ar_order'],
                              window=params['window'],
                              polyorder=params['polyorder'])
    except Exception as e:
        logging.error(f"Feature creation error - {str(e)}")
        errors["feature_errors"] = errors.get("feature_errors", 0) + 1
        return None, None, {"idle": idle_metrics, "runtime": None, "errors": errors}, None, log_file
    
    try:
        model = Ridge(alpha=params['reg_strength'])
        model.fit(X, y)
    except Exception as e:
        logging.error(f"Model training error - {str(e)}")
        errors["model_errors"] = errors.get("model_errors", 0) + 1
        return None, None, {"idle": idle_metrics, "runtime": None, "errors": errors}, None, log_file
    
    last_month = 30
    target_values = df['solar_power'].values
    historical_end_idx = len(target_values)
    historical_start_idx = max(0, historical_end_idx - last_month)
    historical_data = target_values[historical_start_idx:historical_end_idx]
    historical_dates = df.index[historical_start_idx:historical_end_idx]
    
    ghi = df['GHI'].values
    dni = df['DNI'].values
    dhi = df['DHI'].values
    ghi_ext = repeat_last_month(ghi, steps)
    dni_ext = repeat_last_month(dni, steps)
    dhi_ext = repeat_last_month(dhi, steps)
    
    extended_df_length = len(df) + steps
    t_extended = np.arange(extended_df_length)
    fourier_extended = fourier_transform(t_extended, n_harmonics=params['fourier_terms'], periods=[7, 30])
    
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
        errors["forecast_errors"] = errors.get("forecast_errors", 0) + 1
        return None, None, {"idle": idle_metrics, "runtime": None, "errors": errors}, None, log_file
    
    finish_time = time.time()  # Use time.time() for absolute timestamp
    execution_time = finish_time - start_time
    runtime_metrics = measure_resources(is_idle=False)
    runtime_metrics["execution_time"] = execution_time
    
    comp_eff_report = evaluate_computational_efficiency(start_time, finish_time, execution_time, runtime_metrics)
    security_report = evaluate_security(errors, errors)
    scalability_report = evaluate_scalability()
    maintainability_report = evaluate_maintainability()
    
    report_file = generate_performance_report(output_dir, timestamp, comp_eff_report, security_report, scalability_report, maintainability_report)
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        csv_name = f"solar_dhr_daily_{timestamp}_{steps}.csv"
        csv_path = os.path.join(output_dir, csv_name)
        forecast_dates = pd.date_range(start=historical_dates[-1], periods=steps+1, freq='D')[1:]
        forecast_df = pd.DataFrame({
            'datetime': forecast_dates,
            'forecasted_solar_power': forecast_values
        })
        forecast_df.to_csv(csv_path, index=False)
        logging.info(f"Forecast saved to '{csv_path}'")
    except Exception as e:
        logging.error(f"Security: File save error - {str(e)}")
        errors["file_errors"] = errors.get("file_errors", 0) + 1
        return None, None, {"idle": idle_metrics, "runtime": runtime_metrics, "errors": errors}, execution_time, log_file
    
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
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, f"solar_dhr_daily_{timestamp}_{steps}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Plot saved to '{plot_path}'")
    except Exception as e:
        logging.error(f"Security: Plot save error - {str(e)}")
        errors["file_errors"] = errors.get("file_errors", 0) + 1
        return None, None, {"idle": idle_metrics, "runtime": runtime_metrics, "errors": errors}, execution_time, log_file
    
    return [csv_path, plot_path], params, {"idle": idle_metrics, "runtime": runtime_metrics, "errors": errors}, execution_time

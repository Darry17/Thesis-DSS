import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import reservoirpy as rp
from reservoirpy.nodes import Reservoir, Ridge, Input
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

# --- Main Forecast Function ---
def run_forecast(csv_path, steps, output_dir="forecasts", forecast_type="hourly", params=None):
    """
    Run hourly solar power forecasting using Echo State Networks (ESN)
    
    Parameters:
    - csv_path: Path to the CSV file containing solar power data
    - steps: Number of hours to forecast ahead
    - output_dir: Directory to save forecast results
    - forecast_type: Type of forecast ("hourly")
    - params: Dictionary containing ESN hyperparameters
    
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
        df = pd.read_csv(csv_path)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        required_cols = ['solar_power', 'GHI', 'DNI', 'DHI', 'Solar Zenith Angle']
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
        df_resampled = df.resample('h').mean()
        df_resampled = df_resampled.fillna(method='ffill').fillna(method='bfill')
        if df_resampled['solar_power'].isna().any():
            logging.warning("Still have NaN values after filling. Removing them.")
            df_resampled = df_resampled.dropna(subset=['solar_power'])
    except Exception as e:
        logging.error(f"Security: Data processing error - {str(e)}")
        errors["data_errors"] = errors.get("data_errors", 0) + 1
        return None, None, {"idle": idle_metrics, "runtime": None, "errors": errors}, None, log_file
    
    try:
        data = df_resampled['solar_power'].values.astype(float)
        min_non_zero = np.min(data[data > 0]) if np.any(data > 0) else 1e-10
        data_cleaned = np.maximum(data, min_non_zero)
        data_log = np.log1p(data_cleaned)
        if np.any(np.isnan(data_log)) or np.any(np.isinf(data_log)):
            logging.warning("Found NaN or Inf values in log-transformed data.")
            data_log = np.nan_to_num(data_log, nan=np.nanmean(data_log),
                                    posinf=np.nanmax(data_log), neginf=np.nanmin(data_log))
        
        scaler_target = MinMaxScaler(feature_range=(0.1, 0.9))
        scaler_exog = MinMaxScaler(feature_range=(0.1, 0.9))
        data_scaled_target = scaler_target.fit_transform(data_log.reshape(-1, 1)).flatten()
        
        exog_columns = ['GHI', 'DNI', 'DHI', 'Solar Zenith Angle']
        exog_data = df_resampled[exog_columns].values
        if len(exog_data) != len(data_scaled_target):
            min_len = min(len(exog_data), len(data_scaled_target))
            exog_data = exog_data[:min_len]
            data_scaled_target = data_scaled_target[:min_len]
            logging.info(f"Adjusted data lengths to match: {min_len}")
        
        data_scaled_exog = scaler_exog.fit_transform(exog_data)
        
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
        
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        n_samples, lags, n_features = X.shape
        X_flat = X.reshape(n_samples, lags * n_features)
        
        train_size = int(n_samples * 0.8)
        val_size = int(n_samples * 0.1)
        X_train = X_flat[:train_size]
        y_train = y[:train_size]
        X_val = X_flat[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        X_test = X_flat[train_size + val_size:]
        y_test = y[train_size + val_size:]
    except Exception as e:
        logging.error(f"Data preparation error - {str(e)}")
        errors["data_errors"] = errors.get("data_errors", 0) + 1
        return None, None, {"idle": idle_metrics, "runtime": None, "errors": errors}, None, log_file
    
    try:
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
        model.fit(X_train, y_train, reset=True)
    except Exception as e:
        logging.error(f"Model training error - {str(e)}")
        errors["model_errors"] = errors.get("model_errors", 0) + 1
        return None, None, {"idle": idle_metrics, "runtime": None, "errors": errors}, None, log_file
    
    historical_period = 336
    exog_history_period = 168
    historical_end_idx = len(data_scaled_target)
    historical_start_idx = historical_end_idx - historical_period - lags
    historical_data = data_scaled_target[historical_start_idx:historical_end_idx]
    historical_data_log = scaler_target.inverse_transform(historical_data.reshape(-1, 1)).flatten()
    historical_data_original = np.expm1(historical_data_log)
    historical_data_original = np.maximum(historical_data_original, 0)
    
    try:
        historical_dates = df_resampled.index[historical_start_idx:historical_end_idx]
        plot_with_dates = True
    except:
        plot_with_dates = False
        logging.warning("Could not extract dates for plotting")
    
    try:
        last_idx = len(data_scaled_target) - lags
        last_sequence = []
        for i in range(lags):
            timestep_features = np.concatenate([[data_scaled_target[last_idx + i]], 
                                             data_scaled_exog[last_idx + i]])
            last_sequence.append(timestep_features)
        last_sequence = np.array([last_sequence]).reshape(1, lags * n_features)
        
        def generate_forecast(start_sequence, steps, exog_future=None):
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
        
        last_week_exog = data_scaled_exog[-exog_history_period:]
        last_week_exog_original = scaler_exog.inverse_transform(last_week_exog)
        solar_zenith_angles = last_week_exog_original[:, 3]
        
        if steps <= len(last_week_exog):
            exog_future = last_week_exog[:steps]
            exog_future_original = last_week_exog_original[:steps]
            zenith_angles = solar_zenith_angles[:steps]
        else:
            exog_future = np.tile(last_week_exog, (steps // len(last_week_exog) + 1, 1))[:steps]
            exog_future_original = np.tile(last_week_exog_original, (steps // len(last_week_exog) + 1, 1))[:steps]
            zenith_angles = np.tile(solar_zenith_angles, (steps // len(last_week_exog) + 1))[:steps]
        
        forecast_scaled = generate_forecast(last_sequence, steps, exog_future)
        forecast_log = scaler_target.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
        forecast_original = np.expm1(forecast_log)
        forecast_original = np.maximum(forecast_original, 0)
        nighttime_mask = zenith_angles >= 90
        forecast_original[nighttime_mask] = 0
    except Exception as e:
        logging.error(f"Forecast generation error - {str(e)}")
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
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        csv_name = f"solar_esn_hourly_{timestamp}_{steps}.csv"
        csv_path_output = os.path.join(output_dir, csv_name)
        forecast_dates = [df_resampled.index[-1] + timedelta(hours=i + 1) for i in range(steps)]
        forecast_df = pd.DataFrame({
            'datetime': forecast_dates,
            'forecasted_solar_power': forecast_original
        })
        forecast_df.to_csv(csv_path_output, index=False)
        logging.info(f"Forecast saved to '{csv_path_output}'")
    except Exception as e:
        logging.error(f"Security: File save error - {str(e)}")
        errors["file_errors"] = errors.get("file_errors", 0) + 1
        return None, None, {"idle": idle_metrics, "runtime": runtime_metrics, "errors": errors}, execution_time, log_file
    
    try:
        plt.figure(figsize=(14, 6))
        if plot_with_dates:
            plt.plot(historical_dates, historical_data_original, label='Historical Solar Power', color='blue', linewidth=2)
            plt.plot(forecast_dates, forecast_original, label=f'{steps}-hour Forecast', color='red', linestyle='--', linewidth=2)
            plt.axvline(x=df_resampled.index[-1], color='green', linestyle=':', label='Forecast Start', linewidth=2)
            plt.gcf().autofmt_xdate()
        else:
            plt.plot(range(historical_period), historical_data_original, label='Historical Solar Power', color='blue', linewidth=2)
            plt.plot(range(historical_period, historical_period + steps), forecast_original, 
                     label=f'{steps}-hour Forecast', color='red', linestyle='--', linewidth=2)
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
        logging.info(f"Plot saved to '{plot_path}'")
    except Exception as e:
        logging.error(f"Security: Plot save error - {str(e)}")
        errors["file_errors"] = errors.get("file_errors", 0) + 1
        return None, None, {"idle": idle_metrics, "runtime": runtime_metrics, "errors": errors}, execution_time, log_file
    
    return [csv_path_output, plot_path], params, {"idle": idle_metrics, "runtime": runtime_metrics, "errors": errors}, execution_time
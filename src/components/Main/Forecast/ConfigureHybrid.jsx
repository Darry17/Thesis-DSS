import React, { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import axios from "axios";

// Debug utility function
const DEBUG = true;
const debugLog = (message, data = null) => {
  if (DEBUG) {
    console.log(`[DEBUG] ${message}`, data || "");
  }
};

// Add axios interceptors to log all requests and responses
if (DEBUG) {
  axios.interceptors.request.use((request) => {
    console.log("Starting Request", JSON.stringify(request, null, 2));
    return request;
  });

  axios.interceptors.response.use(
    (response) => {
      console.log("Response:", JSON.stringify(response.data, null, 2));
      return response;
    },
    (error) => {
      console.error(
        "Response Error:",
        error.response ? error.response.data : error.message
      );
      return Promise.reject(error);
    }
  );
}

function useQuery() {
  const location = useLocation();
  debugLog("Location search params:", location.search);
  return new URLSearchParams(location.search);
}

const ConfigureHybrid = () => {
  const [step, setStep] = useState(1);

  const navigate = useNavigate();
  const location = useLocation();
  const query = useQuery();

  debugLog("Location state:", location.state);

  const originalFilename = query.get("originalFileName");
  const forecastType = query.get("forecastType");
  const tempFilename = query.get("tempFilename");
  const steps = query.get("steps");
  const granularity = query.get("granularity");
  const model = query.get("model");
  const tempId = query.get("tempId");
  const forecastId = query.get("forecastId");

  debugLog("Query parameters:", {
    originalFilename,
    forecastType,
    tempFilename,
    steps,
    granularity,
    model,
    tempId,
    forecastId,
  });

  // Determine if we're in edit mode
  const isEditing = location.state?.isEditing || false;
  const existingConfig = location.state?.existingConfig || null;

  debugLog("Edit mode:", isEditing);
  debugLog("Existing config:", existingConfig);

  const [dhrParams, setDhrParams] = useState({
    fourier_terms: 4,
    reg_strength: 0.006,
    ar_order: 1,
    window: 23,
    polyorder: 2,
  });

  const [esnParams, setEsnParams] = useState({
    lags: 24,
    N_res: 800,
    rho: 0.9308202574,
    alpha: 0.7191611348,
    sparsity: 0.1335175715,
    lambda_reg: 0.000000021,
  });

  const [message, setMessage] = useState("");
  const [tooltipVisible, setTooltipVisible] = useState({});
  const [isLoading, setIsLoading] = useState(false);
  const [debugInfo, setDebugInfo] = useState(null);

  // Load existing configuration if in edit mode
  useEffect(() => {
    debugLog("useEffect triggered for configuration loading");
    if (isEditing && existingConfig) {
      debugLog("Loading existing config from props:", existingConfig);
      setDhrParams({
        fourier_terms: existingConfig.fourier_terms || 4,
        reg_strength: existingConfig.reg_strength || 0.006,
        ar_order: existingConfig.ar_order || 1,
        window: existingConfig.window || 23,
        polyorder: existingConfig.polyorder || 2,
      });

      setEsnParams({
        lags: existingConfig.lags || 24,
        N_res: existingConfig.N_res || 800,
        rho: existingConfig.rho || 0.9308202574,
        alpha: existingConfig.alpha || 0.7191611348,
        sparsity: existingConfig.sparsity || 0.1335175715,
        lambda_reg: existingConfig.lambda_reg || 0.000000021,
      });
    } else if (isEditing && forecastId) {
      debugLog("Need to fetch configuration for forecast ID:", forecastId);
      fetchExistingConfig();
    }
  }, [isEditing, existingConfig, forecastId]);

  // Function to fetch existing configuration
  const fetchExistingConfig = async () => {
    if (!forecastId) {
      debugLog("No forecast ID provided, cannot fetch config");
      return;
    }

    setIsLoading(true);
    try {
      debugLog(`Fetching config for forecast ID: ${forecastId}`);
      const endpoint = `http://localhost:8000/api/hybrid-configurations/${forecastId}`;
      debugLog("Fetching from endpoint:", endpoint);

      const response = await axios.get(endpoint);
      const config = response.data;
      debugLog("Received configuration:", config);

      setDhrParams({
        fourier_terms: config.fourier_terms || 4,
        reg_strength: config.reg_strength || 0.006,
        ar_order: config.ar_order || 1,
        window: config.window || 23,
        polyorder: config.polyorder || 2,
      });

      setEsnParams({
        lags: config.lags || 24,
        N_res: config.N_res || 800,
        rho: config.rho || 0.9308202574,
        alpha: config.alpha || 0.7191611348,
        sparsity: config.sparsity || 0.1335175715,
        lambda_reg: config.lambda_reg || 0.000000021,
      });

      debugLog("Updated state with fetched config");
    } catch (error) {
      console.error("Error fetching configuration:", error);
      setDebugInfo({
        type: "error",
        context: "fetchExistingConfig",
        error: error.message,
        response: error.response?.data,
      });
      setMessage(`Failed to load configuration: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  // Show tooltip on hover for specific field
  const showTooltip = (field) =>
    setTooltipVisible((prev) => ({ ...prev, [field]: true }));

  // Hide tooltip on mouse leave
  const hideTooltip = (field) =>
    setTooltipVisible((prev) => ({ ...prev, [field]: false }));

  const [errors, setErrors] = useState({});

  const descriptions = {
    fourier_terms: "Number of Fourier terms in the harmonic regression model.",
    reg_strength: "Regularization strength to avoid overfitting.",
    ar_order: "Order of the autoregressive model component.",
    window: "Window size for moving average calculations.",
    polyorder: "Polynomial order for trend fitting.",
    lags: "Number of past time steps used in ESN model.",
    N_res: "Number of reservoir neurons in ESN.",
    rho: "Spectral radius of the reservoir matrix.",
    alpha: "Leaking rate of the reservoir units.",
    sparsity: "Proportion of zero weights in the reservoir.",
    lambda_reg: "Regularization parameter for ESN training.",
  };

  const handleChange = (e, setParams) => {
    const { name, value } = e.target;
    debugLog(`Field change: ${name} = "${value}"`);

    setParams((prev) => {
      const parsed = parseFloat(value);
      const newValue = value === "" ? "" : isNaN(parsed) ? "" : parsed;
      debugLog(`Parsed value: ${name} = ${newValue}`);

      return {
        ...prev,
        [name]: newValue,
      };
    });
  };

  const validateStep = () => {
    debugLog(`Validating step ${step}`);
    const newErrors = {};
    if (step === 1) {
      if (
        !Number.isInteger(dhrParams.fourier_terms) ||
        dhrParams.fourier_terms <= 0
      ) {
        newErrors.fourier_terms = "Must be a positive integer.";
      }
      if (dhrParams.reg_strength < 0) {
        newErrors.reg_strength = "Must be zero or positive.";
      }
      if (!Number.isInteger(dhrParams.ar_order) || dhrParams.ar_order < 0) {
        newErrors.ar_order = "Must be zero or positive integer.";
      }
      if (!Number.isInteger(dhrParams.window) || dhrParams.window <= 0) {
        newErrors.window = "Must be a positive integer.";
      }
      if (!Number.isInteger(dhrParams.polyorder) || dhrParams.polyorder < 0) {
        newErrors.polyorder = "Must be zero or positive integer.";
      }
    } else if (step === 2) {
      if (!Number.isInteger(esnParams.lags) || esnParams.lags <= 0) {
        newErrors.lags = "Must be a positive integer.";
      }
      if (!Number.isInteger(esnParams.N_res) || esnParams.N_res <= 0) {
        newErrors.N_res = "Must be a positive integer.";
      }
      if (esnParams.rho <= 0 || esnParams.rho > 1) {
        newErrors.rho = "Must be > 0 and ≤ 1.";
      }
      if (esnParams.alpha < 0 || esnParams.alpha > 1) {
        newErrors.alpha = "Must be between 0 and 1.";
      }
      if (esnParams.sparsity < 0 || esnParams.sparsity > 1) {
        newErrors.sparsity = "Must be between 0 and 1.";
      }
      if (esnParams.lambda_reg < 0) {
        newErrors.lambda_reg = "Must be zero or positive.";
      }
    }
    debugLog("Validation errors:", newErrors);
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleNextStep = () => {
    debugLog("Moving to next step");
    if (validateStep()) {
      setStep((prev) => prev + 1);
    }
  };

  const handleBack = () => {
    debugLog("Moving to previous step");
    setStep((prev) => Math.max(prev - 1, 1));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    debugLog("Form submitted, validating...");

    if (!validateStep()) {
      debugLog("Validation failed, aborting submission");
      return;
    }

    setIsLoading(true);
    setMessage("Processing...");
    debugLog(`Processing ${isEditing ? "update" : "create"} operation`);

    try {
      if (isEditing) {
        debugLog("Handling update operation");
        await handleUpdate();
      } else {
        debugLog("Handling create operation");
        await handleCreate();
      }
    } catch (err) {
      console.error("Error submitting forecast:", err);
      setDebugInfo({
        type: "error",
        context: "handleSubmit",
        error: err.message,
        response: err.response?.data,
      });
      setMessage(`${isEditing ? "Update" : "Forecast"} failed: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCreate = async () => {
    debugLog("Starting create operation");
    const formData = new FormData();
    formData.append("original_filename", originalFilename);
    formData.append("tempFilename", tempFilename);
    formData.append("forecast_type", forecastType);
    formData.append("granularity", granularity);
    formData.append("steps", steps);
    formData.append("model", model);
    formData.append("temp_id", tempId);
    formData.append("fourier_terms", dhrParams.fourier_terms);
    formData.append("reg_strength", dhrParams.reg_strength);
    formData.append("ar_order", dhrParams.ar_order);
    formData.append("window", dhrParams.window);
    formData.append("polyorder", dhrParams.polyorder);
    formData.append("lags", esnParams.lags);
    formData.append("N_res", esnParams.N_res);
    formData.append("rho", esnParams.rho);
    formData.append("alpha", esnParams.alpha);
    formData.append("sparsity", esnParams.sparsity);
    formData.append("lambda_reg", esnParams.lambda_reg);

    // Debug log all form fields
    debugLog("FormData entries:");
    for (let pair of formData.entries()) {
      debugLog(`${pair[0]}: ${pair[1]}`);
    }

    const normalizedModel =
      model.toLowerCase() === "dhr-esn" ? "hybrid" : model.toLowerCase();
    const endpoint = `http://localhost:8000/upload/${normalizedModel}/${granularity.toLowerCase()}`;

    debugLog(`Submitting to endpoint: ${endpoint}`);

    try {
      const response = await axios.post(endpoint, formData);
      debugLog("Create response received:", response.data);

      // Instead of using array indices, use the returned keys
      const downloadUrls = response.data?.download_urls || [];
      const forecastId = response.data?.forecast_id || null;

      // Find the CSV and image URLs
      const csvUrl = downloadUrls.find((url) => url.endsWith(`_${steps}.csv`));
      const imageUrl = downloadUrls.find((url) =>
        url.endsWith(`_${steps}.png`)
      );

      debugLog("CSV URL:", csvUrl);
      debugLog("Image URL:", imageUrl);
      debugLog("Forecast ID:", forecastId);

      let forecastData = [];
      if (csvUrl) {
        try {
          const csvResponse = await axios.get(csvUrl);
          debugLog(
            "CSV data fetched",
            csvResponse.data.substring(0, 100) + "..."
          );

          const lines = csvResponse.data.split("\n").slice(1);
          forecastData = lines
            .filter((line) => line.trim() !== "")
            .map((line) => {
              const [timestamp, forecast] = line.split(",");
              return { timestamp, forecast: parseFloat(forecast) };
            });

          debugLog(`Processed ${forecastData.length} forecast data points`);
        } catch (csvErr) {
          console.error("Error fetching CSV data:", csvErr);
          setDebugInfo({
            type: "warning",
            context: "fetchCSV",
            error: csvErr.message,
          });
        }
      }

      navigate("/result", {
        state: {
          imageUrl, // This should now be properly passed
          forecastData: {
            id: forecastId,
            original_filename: originalFilename,
            model: model,
            steps: steps,
            forecastType: forecastType,
            granularity: granularity,
            data: forecastData,
            tempFilename: tempFilename,
          },
        },
      });
    } catch (error) {
      console.error("API call error:", error);
      setDebugInfo({
        type: "error",
        context: "createAPI",
        error: error.message,
        response: error.response?.data,
        request: {
          url: endpoint,
          method: "POST",
          formData: Object.fromEntries(formData),
        },
      });
      throw error;
    }
  };

  const handleUpdate = async () => {
    try {
      // Log the forecast ID being used to ensure it's valid
      debugLog("Updating forecast with ID:", forecastId);

      // Ensure forecastId is a valid integer
      const parsedForecastId = parseInt(forecastId);
      if (isNaN(parsedForecastId)) {
        throw new Error("Invalid forecast ID");
      }

      // Create configuration data matching the FastAPI HybridForecastCreate model structure
      const configData = {
        // Add forecast_id to match the HybridForecastCreate model requirements
        forecast_id: parsedForecastId,

        // Flattened structure instead of nested dhr/esn objects
        fourier_terms: parseInt(dhrParams.fourier_terms),
        reg_strength: parseFloat(dhrParams.reg_strength),
        ar_order: parseInt(dhrParams.ar_order),
        window: parseInt(dhrParams.window),
        polyorder: parseInt(dhrParams.polyorder),
        lags: parseInt(esnParams.lags),
        N_res: parseInt(esnParams.N_res),
        rho: parseFloat(esnParams.rho),
        alpha: parseFloat(esnParams.alpha),
        sparsity: parseFloat(esnParams.sparsity),
        lambda_reg: parseFloat(esnParams.lambda_reg),
      };

      // Log the data being sent for debugging
      debugLog(
        "Sending configuration data:",
        JSON.stringify(configData, null, 2)
      );

      // First API call to update configuration
      const endpoint = `http://localhost:8000/api/hybrid-configurations/${parsedForecastId}`;

      // Add debug headers and response logging
      try {
        debugLog(`Making PUT request to: ${endpoint}`);
        const configResponse = await axios.put(endpoint, configData, {
          headers: {
            "Content-Type": "application/json",
          },
        });
        debugLog("Configuration update response:", configResponse.data);
        setMessage("Configuration updated successfully!");
      } catch (configErr) {
        console.error(
          "Configuration update error details:",
          configErr.response?.data || configErr.message
        );
        setDebugInfo({
          type: "error",
          context: "updateConfig",
          error: configErr.message,
          response: configErr.response?.data,
          request: {
            url: endpoint,
            method: "PUT",
            data: configData,
          },
        });
        throw configErr;
      }

      // Prepare form data for the second API call
      const formData = new FormData();
      formData.append("tempFilename", tempFilename || "");
      formData.append("forecast_type", forecastType || "");
      formData.append("steps", steps || "");
      formData.append("forecast_id", parsedForecastId); // Add forecast_id here

      // DHR parameters
      formData.append("fourier_terms", dhrParams.fourier_terms);
      formData.append("reg_strength", dhrParams.reg_strength);
      formData.append("ar_order", dhrParams.ar_order);
      formData.append("window", dhrParams.window);
      formData.append("polyorder", dhrParams.polyorder);

      // ESN parameters
      formData.append("lags", esnParams.lags);
      formData.append("N_res", esnParams.N_res);
      formData.append("rho", esnParams.rho);
      formData.append("alpha", esnParams.alpha);
      formData.append("sparsity", esnParams.sparsity);
      formData.append("lambda_reg", esnParams.lambda_reg);

      // Debug log the FormData
      debugLog("FormData entries:");
      for (let pair of formData.entries()) {
        debugLog(`${pair[0]}: ${pair[1]}`);
      }

      // Second API call to generate new forecast
      const uploadEndpoint = `http://localhost:8000/upload/edit-hybrid/${
        granularity?.toLowerCase() || ""
      }`;
      debugLog("Making request to:", uploadEndpoint);

      // Try/catch the second API call separately
      let response;
      try {
        response = await axios.post(uploadEndpoint, formData);
        debugLog("Upload response:", response.data);
      } catch (uploadErr) {
        console.error(
          "Upload error details:",
          uploadErr.response?.data || uploadErr.message
        );

        setDebugInfo({
          type: "warning",
          context: "regenerateForecast",
          error: uploadErr.message,
          response: uploadErr.response?.data,
          request: {
            url: uploadEndpoint,
            method: "POST",
            formData: Object.fromEntries(formData),
          },
        });

        // Continue execution - we can still navigate with just the updated config
        setMessage("Configuration updated but forecast regeneration failed.");
      }

      let imageUrl = null;
      let forecastData = [];

      // Check if we got a successful response from the second call
      if (response && response.data) {
        // Check if the response has download_urls
        if (
          response.data.download_urls &&
          Array.isArray(response.data.download_urls)
        ) {
          const downloadUrls = response.data.download_urls;
          debugLog("Download URLs:", downloadUrls);

          const csvUrl = downloadUrls.find((url) =>
            url.endsWith(`_${steps}.csv`)
          );
          imageUrl = downloadUrls.find((url) => url.endsWith(`_${steps}.png`));

          debugLog("CSV URL:", csvUrl);
          debugLog("Image URL:", imageUrl);

          // Fetch forecast data if CSV URL is available
          if (csvUrl) {
            try {
              const csvResponse = await axios.get(csvUrl);
              debugLog(
                "CSV data fetched",
                csvResponse.data.substring(0, 100) + "..."
              );

              const lines = csvResponse.data.split("\n").slice(1);
              forecastData = lines
                .filter((line) => line.trim() !== "")
                .map((line) => {
                  const [timestamp, forecast] = line.split(",");
                  return { timestamp, forecast: parseFloat(forecast) };
                });

              debugLog(`Processed ${forecastData.length} forecast data points`);
            } catch (csvErr) {
              console.error("Error fetching CSV data:", csvErr);
              setDebugInfo({
                type: "warning",
                context: "fetchCSV",
                error: csvErr.message,
              });
            }
          }
        }
      }

      // Navigate to result page with whatever data we have
      debugLog("Navigating to result page with:", {
        imageUrl,
        forecastId: parsedForecastId,
      });

      navigate("/result", {
        state: {
          imageUrl,
          forecastData: {
            id: parsedForecastId,
            original_filename: originalFilename,
            model: model,
            steps: steps,
            granularity: granularity,
            data: forecastData,
            tempFilename: tempFilename,
            forecastType: forecastType,
          },
          message: !imageUrl
            ? "Configuration updated, but could not generate new forecast visualization."
            : null,
        },
      });
    } catch (error) {
      console.error("Error in handleUpdate:", error);
      const errorDetail = error.response?.data?.detail || error.message;
      setMessage(`Update failed: ${errorDetail}`);
      throw error; // Re-throw to be caught by handleSubmit
    }
  };

  const getStepTitle = (stepNum) => {
    const action = isEditing ? "Edit" : "Configure";
    if (stepNum === 1) {
      return `${action} Dynamic Harmonic Regression`;
    } else {
      return `${action} Echo State Network`;
    }
  };

  const renderInput = (name, value, setParams) => (
    <div key={name} className="w-full md:w-1/2 mb-6 relative">
      <div className="flex space-x-10">
        <div className="flex-1">
          <label className="block text-m font-medium mb-1 capitalize">
            {name.replace(/_/g, " ")}
            <span
              className="text-gray-500 cursor-pointer ml-2"
              onMouseEnter={() => showTooltip(name)}
              onMouseLeave={() => hideTooltip(name)}>
              ⓘ
            </span>
          </label>

          {tooltipVisible[name] && descriptions[name] && (
            <div className="absolute bg-gray-800 text-white p-2 rounded text-xs max-w-xs z-10">
              {descriptions[name]}
            </div>
          )}

          <input
            type="text"
            name={name}
            value={value === undefined || value === null ? "" : String(value)}
            onChange={(e) => handleChange(e, setParams)}
            className={`w-full p-2 border rounded-md ${
              errors[name] ? "border-red-500" : "border-black-300"
            }`}
          />

          {errors[name] && (
            <p className="text-red-600 text-sm mt-1">{errors[name]}</p>
          )}
        </div>
      </div>
    </div>
  );

  // Debug panel to display relevant state
  const renderDebugPanel = () => {
    if (!DEBUG) return null;

    return (
      <div className="fixed bottom-0 right-0 bg-gray-900 text-white p-4 max-w-md max-h-96 overflow-auto opacity-80 z-50 text-xs">
        <h3 className="font-bold mb-2">DEBUG INFO</h3>
        <button
          className="absolute top-2 right-2 text-xs bg-red-500 px-2 py-1 rounded"
          onClick={() => setDebugInfo(null)}>
          Clear
        </button>

        <div>
          <p>Mode: {isEditing ? "Edit" : "Create"}</p>
          <p>Step: {step}</p>
          <p>Loading: {isLoading ? "Yes" : "No"}</p>
          <p>Forecast ID: {forecastId || "None"}</p>

          {message && (
            <div className="mt-2">
              <p className="font-bold">Message:</p>
              <p className="text-yellow-300">{message}</p>
            </div>
          )}

          {Object.keys(errors).length > 0 && (
            <div className="mt-2">
              <p className="font-bold">Validation Errors:</p>
              <pre className="text-red-300">
                {JSON.stringify(errors, null, 2)}
              </pre>
            </div>
          )}

          {debugInfo && (
            <div className="mt-2 border-t border-gray-600 pt-2">
              <p className="font-bold text-red-300">
                {debugInfo.type.toUpperCase()}: {debugInfo.context}
              </p>
              <pre className="break-words whitespace-pre-wrap">
                {JSON.stringify(debugInfo, null, 2)}
              </pre>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen relative">
      {message && <p className="mb-4 text-center text-red-600">{message}</p>}
      {step === 1 && (
        <>
          <div className="fixed inset-0 bg-gray-100" style={{ zIndex: -1 }} />
          <div className="relative z-10 flex justify-center flex-1 min-h-screen">
            <div className="w-1/3 h-1/3 p-10 px-20 bg-white rounded-lg shadow-md">
              <h2 className="text-4xl text-left font-bold mb-10">
                {getStepTitle(1)}
              </h2>
              {Object.entries(dhrParams).map(([key, value]) =>
                renderInput(key, value, setDhrParams)
              )}
              <div className="flex justify-between mt-6">
                {isEditing && (
                  <button
                    onClick={() => navigate(-1)}
                    className="bg-red-700 text-white py-2 px-4 rounded hover:bg-red-600">
                    Cancel
                  </button>
                )}
                <button
                  onClick={handleNextStep}
                  className="ml-auto bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700">
                  Next
                </button>
              </div>
            </div>
          </div>
        </>
      )}

      {step === 2 && (
        <div className="relative z-10 flex justify-center flex-1 min-h-screen">
          <div className="w-1/3 h-1/3 p-10 px-20 bg-white rounded-lg shadow-md">
            <h2 className="text-4xl text-left font-bold mb-10">
              {getStepTitle(2)}
            </h2>
            {Object.entries(esnParams).map(([key, value]) =>
              renderInput(key, value, setEsnParams)
            )}
            <div className="flex justify-between mt-6">
              <button
                onClick={handleBack}
                className="bg-gray-400 text-white py-2 px-4 rounded hover:bg-gray-500">
                Back
              </button>
              <button
                onClick={handleSubmit}
                className="bg-green-600 text-white py-2 px-4 rounded hover:bg-green-700"
                disabled={isLoading}>
                {isLoading ? "Processing..." : isEditing ? "Update" : "Submit"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Debug panel */}
      {renderDebugPanel()}
    </div>
  );
};

export default ConfigureHybrid;

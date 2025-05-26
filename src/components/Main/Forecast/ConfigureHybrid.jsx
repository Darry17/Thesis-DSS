import React, { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import axios from "axios";

function useQuery() {
  return new URLSearchParams(useLocation().search);
}

const ConfigureHybrid = () => {
  const [step, setStep] = useState(1);
  const navigate = useNavigate();
  const location = useLocation();
  const query = useQuery();

  const originalFilename = query.get("originalFileName");
  const forecastType = query.get("forecastType");
  const tempFilename = query.get("tempFilename");
  const steps = query.get("steps");
  const energyDemand = query.get("energyDemand");
  const maxCapacity = query.get("maxCapacity");
  const granularity = query.get("granularity");
  const model = query.get("model");
  const tempId = query.get("tempId");
  const forecastId = query.get("forecastId");

  const isEditing = location.state?.isEditing || false;
  const existingConfig = location.state?.existingConfig || null;

  const [dhrParams, setDhrParams] = useState(
    granularity === "Hourly"
      ? forecastType === "solar"
        ? {
            fourier_terms: 4,
            reg_strength: 0.006,
            ar_order: 1,
            window: 23,
            polyorder: 2,
          }
        : forecastType === "wind"
        ? {
            fourier_terms: 10,
            reg_strength: 0.000001,
            ar_order: 10,
            window: 16,
            polyorder: 1,
          }
        : null
      : granularity === "Daily"
      ? forecastType === "solar"
        ? {
            fourier_terms: 4,
            reg_strength: 0.006,
            ar_order: 1,
            window: 7,
            polyorder: 2,
          }
        : forecastType === "wind"
        ? {
            fourier_terms: 4,
            reg_strength: 0.0033872555658521508,
            ar_order: 7,
            window: 9,
            polyorder: 2,
          }
        : null
      : null
  );

  const [esnParams, setEsnParams] = useState(
    granularity === "Hourly"
      ? forecastType === "solar"
        ? {
            N_res: 800,
            rho: 0.9308202574,
            sparsity: 0.1335175715,
            alpha: 0.7191611348,
            lambda_reg: 0.000000021,
            lags: 24,
            n_features: 5,
          }
        : forecastType === "wind"
        ? {
            N_res: 1999,
            rho: 0.08708791675,
            sparsity: 0.1254732037,
            alpha: 0.494757914,
            lambda_reg: 0.48,
            lags: 24,
            n_features: 5,
          }
        : null
      : granularity === "Daily"
      ? forecastType === "solar"
        ? {
            N_res: 963,
            rho: 0.12455826,
            sparsity: 0.6855266625,
            alpha: 0.2769944104,
            lambda_reg: 0.0167,
            lags: 2,
            n_features: 4,
          }
        : forecastType === "wind"
        ? {
            N_res: 610,
            rho: 0.1032449387,
            sparsity: 0.8910025925,
            alpha: 0.9637796974,
            lambda_reg: 0.41,
            lags: 4,
            n_features: 4,
          }
        : null
      : null
  );

  const [message, setMessage] = useState("");
  const [tooltipVisible, setTooltipVisible] = useState({});
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (isEditing && existingConfig) {
      console.log("Loading existing config:", existingConfig);
      setDhrParams({
        fourier_terms: existingConfig.fourier_terms,
        window: existingConfig.window,
        ar_order: existingConfig.ar_order,
        reg_strength: existingConfig.reg_strength,
        polyorder: existingConfig.polyorder,
      });

      setEsnParams({
        lags: existingConfig.lags,
        N_res: existingConfig.N_res,
        rho: existingConfig.rho,
        alpha: existingConfig.alpha,
        sparsity: existingConfig.sparsity,
        lambda_reg: existingConfig.lambda_reg,
        n_features: existingConfig.n_features,
      });
    } else if (isEditing && forecastId) {
      fetchExistingConfig();
    }
  }, [isEditing, existingConfig, forecastId]);

  const fetchExistingConfig = async () => {
    if (!forecastId) return;

    setIsLoading(true);
    try {
      const endpoint = `http://localhost:8000/api/hybrid-configurations/${forecastId}`;
      const response = await axios.get(endpoint);
      const config = response.data;

      setDhrParams({
        fourier_terms: config.dhr?.fourier_terms,
        window: config.dhr?.window,
        ar_order: config.dhr?.ar_order,
        reg_strength: config.dhr?.reg_strength,
        polyorder: config.dhr?.polyorder,
      });

      setEsnParams({
        lags: config.esn?.lags,
        N_res: config.esn?.N_res,
        rho: config.esn?.rho,
        alpha: config.esn?.alpha,
        sparsity: config.esn?.sparsity,
        lambda_reg: config.esn?.lambda_reg,
        n_features: config.esn?.n_features,
      });
    } catch (error) {
      console.error("Error fetching configuration:", error);
      setMessage("Failed to load configuration.");
    } finally {
      setIsLoading(false);
    }
  };

  const showTooltip = (field) =>
    setTooltipVisible((prev) => ({ ...prev, [field]: true }));

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
    n_features: "Number of input features used in the ESN model.",
  };

  const customLabels = {
    fourier_terms: "Fourier Order",
    reg_strength: "Regularization",
    ar_order: "AR Order",
    window: "Window Length",
    polyorder: "Polyorder",
    lags: "Lags",
    N_res: "Reservoir Neurons",
    rho: "Spectral Radius",
    alpha: "Leaking Rate",
    sparsity: "Sparsity",
    lambda_reg: "Lambda Regularization",
    n_features: "Number of Features",
  };

  const dhrParamOrder = [
    "fourier_terms",
    "window",
    "polyorder",
    "reg_strength",
    "ar_order",
  ];
  const esnParamOrder = [
    "N_res",
    "lambda_reg",
    "rho",
    "alpha",
    "sparsity",
    "lags",
    "n_features",
  ];

  const handleChange = (e, setParams) => {
    const { name, value } = e.target;
    setParams((prev) => {
      const parsed = parseFloat(value);
      return {
        ...prev,
        [name]: value === "" ? "" : isNaN(parsed) ? "" : parsed,
      };
    });
  };

  // CSS for the loading bar (already present, kept for reference)
  const loadingBarStyles = `
  .loading-bar {
    width: 100%;
    height: 4px;
    background-color: #e0e0e0;
    border-radius: 2px;
    overflow: hidden;
    position: relative;
    margin-top: 1rem;
  }
  .loading-bar::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 30%;
    height: 100%;
    background-color: #4caf50;
    animation: loading 1.5s infinite ease-in-out;
  }
  @keyframes loading {
    0% { transform: translateX(-100%); }
    50% { transform: translateX(300%); }
    100% { transform: translateX(300%); }
  }
`;

  const validateStep = () => {
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
      if (esnParams.n_features < 0) {
        newErrors.n_features = "Must be zero or positive.";
      }
    }
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleNextStep = () => {
    if (validateStep()) {
      setStep((prev) => prev + 1);
    }
  };

  const handleBack = () => {
    setStep((prev) => Math.max(prev - 1, 1));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!validateStep()) return;

    setIsLoading(true);
    setMessage("");

    try {
      if (isEditing) {
        await handleUpdate();
      } else {
        await handleCreate();
      }
    } catch (err) {
      console.error("Error submitting forecast:", err);
      setMessage(`${isEditing ? "Update" : "Forecast"} failed: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCreate = async () => {
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
    formData.append("n_features", esnParams.n_features);

    const normalizedModel =
      model.toLowerCase() === "dhr-esn" ? "hybrid" : model.toLowerCase();
    const endpoint = `http://localhost:8000/upload/${normalizedModel}/${granularity.toLowerCase()}`;

    const response = await axios.post(endpoint, formData);
    console.log("Response data:", response.data);

    const downloadUrls = response.data?.download_urls || [];
    const forecastId = response.data?.forecast_id || null;

    const csvUrl = downloadUrls.find((url) => url.endsWith(`_${steps}.csv`));
    const imageUrl = downloadUrls.find((url) => url.endsWith(`_${steps}.png`));

    console.log("CSV URL:", csvUrl);
    console.log("Image URL:", imageUrl);
    console.log("Forecast ID:", forecastId);

    let forecastData = [];
    if (csvUrl) {
      try {
        const csvResponse = await axios.get(csvUrl);
        const lines = csvResponse.data.split("\n").slice(1);
        forecastData = lines
          .filter((line) => line.trim() !== "")
          .map((line) => {
            const [timestamp, forecast] = line.split(",");
            return { timestamp, forecast: parseFloat(forecast) };
          });
      } catch (csvErr) {
        console.error("Error fetching CSV data:", csvErr);
      }
    }

    navigate("/result", {
      state: {
        imageUrl,
        forecastData: {
          id: forecastId,
          original_filename: originalFilename,
          model: model,
          steps: steps,
          forecastType: query.get("forecastType"),
          energyDemand: energyDemand,
          maxCapacity: maxCapacity,
          granularity: granularity,
          data: forecastData,
          tempFilename: tempFilename,
        },
      },
    });
  };

  const handleUpdate = async () => {
    try {
      console.log("Updating forecast with ID:", forecastId);

      const parsedForecastId = parseInt(forecastId);
      if (isNaN(parsedForecastId)) {
        throw new Error("Invalid forecast ID");
      }

      const configData = {
        forecast_id: parsedForecastId,
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
        n_features: parseFloat(esnParams.n_features),
      };

      console.log(
        "Sending configuration data:",
        JSON.stringify(configData, null, 2)
      );

      try {
        const endpoint = `http://localhost:8000/api/hybrid-configurations/${parsedForecastId}`;
        const configResponse = await axios.put(endpoint, configData, {
          headers: {
            "Content-Type": "application/json",
          },
        });
        console.log("Configuration update response:", configResponse.data);
        setMessage("");
      } catch (configErr) {
        console.error(
          "Configuration update error details:",
          configErr.response?.data || configErr.message
        );
        throw configErr;
      }

      const formData = new FormData();
      formData.append("tempFilename", tempFilename || "");
      formData.append("forecast_type", forecastType || "");
      formData.append("steps", steps || "");
      formData.append("forecast_id", parsedForecastId);
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
      formData.append("n_features", esnParams.n_features);

      console.log("FormData entries:");
      for (let pair of formData.entries()) {
        console.log(pair[0] + ": " + pair[1]);
      }

      const uploadEndpoint = `http://localhost:8000/upload/edit-hybrid/${
        granularity?.toLowerCase() || ""
      }`;
      console.log("Making request to:", uploadEndpoint);

      let response;
      try {
        response = await axios.post(uploadEndpoint, formData);
        console.log("Upload response:", response.data);
      } catch (uploadErr) {
        console.error(
          "Upload error details:",
          uploadErr.response?.data || uploadErr.message
        );
        setMessage("Configuration updated but forecast regeneration failed.");
      }

      let imageUrl = null;
      let forecastData = [];

      if (response && response.data) {
        if (
          response.data.download_urls &&
          Array.isArray(response.data.download_urls)
        ) {
          const downloadUrls = response.data.download_urls;
          const csvUrl = downloadUrls.find((url) =>
            url.endsWith(`_${steps}.csv`)
          );
          imageUrl = downloadUrls.find((url) => url.endsWith(`_${steps}.png`));

          if (csvUrl) {
            try {
              const csvResponse = await axios.get(csvUrl);
              const lines = csvResponse.data.split("\n").slice(1);
              forecastData = lines
                .filter((line) => line.trim() !== "")
                .map((line) => {
                  const [timestamp, forecast] = line.split(",");
                  return { timestamp, forecast: parseFloat(forecast) };
                });
            } catch (csvErr) {
              console.error("Error fetching CSV data:", csvErr);
            }
          }
        }
      }

      navigate("/result", {
        state: {
          imageUrl,
          forecastData: {
            id: parsedForecastId,
            original_filename: originalFilename,
            model: model,
            steps: steps,
            energyDemand: energyDemand,
            maxCapacity: maxCapacity,
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
      throw error;
    }
  };

  const getStepTitle = (stepNum) => {
    if (stepNum === 1) {
      return `Dynamic Harmonic Regression`;
    } else {
      return `Echo State Network`;
    }
  };

  const renderInput = (name, value, setParams) => (
    <div key={name} className="relative">
      <label className="block text-m font-medium mb-1 capitalize">
        {customLabels[name].replace(/_/g, " ")}
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
        className={`w-full p-2 border rounded-md`}
      />
      {errors[name] && (
        <p className="text-red-600 text-sm mt-1">{errors[name]}</p>
      )}
    </div>
  );

  const renderForm = (params, setParams) => {
    const paramOrder = step === 1 ? dhrParamOrder : esnParamOrder;
    return (
      <div className="grid grid-cols-2 gap-x-10 gap-y-4">
        {paramOrder.map((key) => renderInput(key, params[key], setParams))}
      </div>
    );
  };

  return (
    <div className="min-h-screen relative">
      <div className="fixed inset-0 bg-gray-100" style={{ zIndex: -1 }} />
      <div className="relative z-10 flex justify-center flex-1 min-h-screen">
        <div className="w-1/3 h-1/3 p-10 px-20 bg-white rounded-lg shadow-md mb-10">
          <style>{loadingBarStyles}</style>
          <h2 className="text-4xl text-left font-bold mb-10">
            {getStepTitle(step)}
          </h2>
          {message && (
            <p
              className={`mb-4 text-center ${
                message.includes("failed") ? "text-red-500" : "text-green-500"
              }`}>
              {message}
            </p>
          )}
          {isLoading && <div className="ml-2 mb-7 loading-bar"></div>}
          {step === 1 && (
            <form onSubmit={(e) => e.preventDefault()} className="space-y-4">
              {renderForm(dhrParams, setDhrParams)}
              <div className="flex justify-between mt-6">
                {isEditing && (
                  <button
                    onClick={() => navigate(-1)}
                    className="bg-red-700 border-none text-white py-2 px-4 rounded hover:bg-red-600 focus:outline-none focus:ring-2 focus:ring-red-500 cursor-pointer"
                    disabled={isLoading}>
                    Cancel
                  </button>
                )}
                <button
                  onClick={handleNextStep}
                  className="ml-auto bg-green-600 border-none text-white py-2 px-4 rounded hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 cursor-pointer"
                  disabled={isLoading}>
                  Next
                </button>
              </div>
            </form>
          )}
          {step === 2 && (
            <form onSubmit={handleSubmit} className="space-y-4">
              {renderForm(esnParams, setEsnParams)}
              <div className="flex justify-end space-x-4 mt-6">
                <button
                  onClick={handleBack}
                  className="bg-red-700 text-white py-2 px-4 rounded hover:bg-red-600 focus:outline-none focus:ring-2 focus:ring-red-500 cursor-pointer"
                  disabled={isLoading}>
                  Back
                </button>
                <button
                  type="submit"
                  className="bg-green-600 text-white py-2 px-4 rounded hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 cursor-pointer"
                  disabled={isLoading}>
                  {isEditing ? "Update" : "Submit"}
                </button>
              </div>
            </form>
          )}
        </div>
      </div>
    </div>
  );
};

export default ConfigureHybrid;

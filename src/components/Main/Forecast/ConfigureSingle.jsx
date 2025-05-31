import React, { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import axios from "axios";

function useQuery() {
  return new URLSearchParams(useLocation().search);
}

export default function ConfigureSingle() {
  const navigate = useNavigate();
  const location = useLocation();
  const query = useQuery();

  // Get URL parameters
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

  // Determine if we're in edit mode
  const isEditing = location.state?.isEditing || false;
  const existingConfig = location.state?.existingConfig || null;

  // Initialize state with default values based on forecastType and granularity
  const getDefaultDhrParams = () => {
    if (forecastType === "solar") {
      if (granularity === "Hourly") {
        return {
          fourier_terms: "3",
          reg_strength: "0.0001000100524",
          ar_order: "3",
          window: "23",
          polyorder: "3",
        };
      } else if (granularity === "Daily") {
        return {
          fourier_terms: "3",
          reg_strength: "0.006",
          ar_order: "3",
          window: "7",
          polyorder: "2",
        };
      }
    } else if (forecastType === "wind") {
      if (granularity === "Hourly") {
        return {
          fourier_terms: "3",
          reg_strength: "0.00010001005240259047",
          ar_order: "3",
          window: "23",
          polyorder: "3",
        };
      } else if (granularity === "Daily") {
        return {
          fourier_terms: "4",
          reg_strength: "0.0033872555658521508",
          ar_order: "7",
          window: "9",
          polyorder: "2",
        };
      }
    }
    return {};
  };

  const getDefaultEsnParams = () => {
    if (forecastType === "solar") {
      if (granularity === "Hourly") {
        return {
          lags: "24",
          N_res: "800",
          rho: "0.9308202574",
          alpha: "0.7191611348",
          sparsity: "0.1335175715",
          lambda_reg: "0.000000021",
        };
      } else if (granularity === "Daily") {
        return {
          lags: "2",
          N_res: "963",
          rho: "0.12455826",
          alpha: "0.2769944104",
          sparsity: "0.6855266625",
          lambda_reg: "0.0167",
        };
      }
    } else if (forecastType === "wind") {
      if (granularity === "Hourly") {
        return {
          lags: "24",
          N_res: "1999",
          rho: "0.08708791675",
          alpha: "0.494757914",
          sparsity: "0.1254732037",
          lambda_reg: "0.48",
        };
      } else if (granularity === "Daily") {
        return {
          lags: "4",
          N_res: "610",
          rho: "0.1032449387",
          alpha: "0.9637796974",
          sparsity: "0.8910025925",
          lambda_reg: "0.41",
        };
      }
    }
    return {};
  };

  const [dhrParams, setDhrParams] = useState(getDefaultDhrParams());
  const [esnParams, setEsnParams] = useState(getDefaultEsnParams());
  const [message, setMessage] = useState("");
  const [tooltipVisible, setTooltipVisible] = useState({});
  const [isLoading, setIsLoading] = useState(false);
  const [errors, setErrors] = useState({});

  // Load existing configuration if in edit mode
  useEffect(() => {
    if (isEditing && existingConfig) {
      console.log("Loading existing config:", existingConfig);
      if (model === "DHR") {
        setDhrParams({
          fourier_terms:
            existingConfig.fourier_terms?.toString() ||
            getDefaultDhrParams().fourier_terms,
          reg_strength:
            existingConfig.reg_strength?.toString() ||
            getDefaultDhrParams().reg_strength,
          ar_order:
            existingConfig.ar_order?.toString() ||
            getDefaultDhrParams().ar_order,
          window:
            existingConfig.window?.toString() || getDefaultDhrParams().window,
          polyorder:
            existingConfig.polyorder?.toString() ||
            getDefaultDhrParams().polyorder,
        });
      } else if (model === "ESN") {
        setEsnParams({
          lags: existingConfig.lags?.toString() || getDefaultEsnParams().lags,
          N_res:
            existingConfig.N_res?.toString() || getDefaultEsnParams().N_res,
          rho: existingConfig.rho?.toString() || getDefaultEsnParams().rho,
          alpha:
            existingConfig.alpha?.toString() || getDefaultEsnParams().alpha,
          sparsity:
            existingConfig.sparsity?.toString() ||
            getDefaultEsnParams().sparsity,
          lambda_reg:
            existingConfig.lambda_reg?.toString() ||
            getDefaultEsnParams().lambda_reg,
        });
      }
    } else if (isEditing && forecastId) {
      fetchExistingConfig();
    }
  }, [isEditing, existingConfig, model, forecastId, forecastType, granularity]);

  const fetchExistingConfig = async () => {
    if (!forecastId) return;

    setIsLoading(true);
    try {
      const endpoint = `http://localhost:8000/api/${model.toLowerCase()}-configurations/${forecastId}`;
      const response = await axios.get(endpoint);
      const config = response.data;

      if (model === "DHR") {
        setDhrParams({
          fourier_terms:
            config.fourier_terms?.toString() ||
            getDefaultDhrParams().fourier_terms,
          reg_strength:
            config.reg_strength?.toString() ||
            getDefaultDhrParams().reg_strength,
          ar_order:
            config.ar_order?.toString() || getDefaultDhrParams().ar_order,
          window: config.window?.toString() || getDefaultDhrParams().window,
          polyorder:
            config.polyorder?.toString() || getDefaultDhrParams().polyorder,
        });
      } else if (model === "ESN") {
        setEsnParams({
          lags: config.lags?.toString() || getDefaultEsnParams().lags,
          N_res: config.N_res?.toString() || getDefaultEsnParams().N_res,
          rho: config.rho?.toString() || getDefaultEsnParams().rho,
          alpha: config.alpha?.toString() || getDefaultEsnParams().alpha,
          sparsity:
            config.sparsity?.toString() || getDefaultEsnParams().sparsity,
          lambda_reg:
            config.lambda_reg?.toString() || getDefaultEsnParams().lambda_reg,
        });
      }
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

  const descriptions = {
    fourier_terms: "Number of Fourier terms for capturing seasonality.",
    reg_strength: "Ridge regression regularization parameter.",
    ar_order: "Number of past values to use for autoregression.",
    window: "Window size for smoothing.",
    polyorder: "Order of polynomial for Savitzky-Golay smoothing.",
    lags: "Number of time steps as input.",
    N_res: "Number of neurons in the reservoir.",
    rho: "Controls echo state property.",
    alpha: "Scaling factor applied to the input data.",
    sparsity: "Reservoir connectivity.",
    lambda_reg:
      "Regularization parameter for ESN training. Helps prevent overfitting.",
  };

  const customLabels = {
    fourier_terms: "Fourier Order",
    reg_strength: "Regularization",
    ar_order: "AR Order",
    window: "Window Length",
    polyorder: "Polyorder",
    lags: "Lags",
    N_res: "Reservoir Size",
    rho: "Spectral Radius",
    alpha: "Input Scaling",
    sparsity: "Sparsity",
    lambda_reg: "Regularization",
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
  ];

  const integerFields = [
    "fourier_terms",
    "ar_order",
    "window",
    "polyorder",
    "lags",
    "N_res",
  ];
  const decimalFields = [
    "reg_strength",
    "rho",
    "alpha",
    "sparsity",
    "lambda_reg",
  ];

  const handleChange = (e, modelType) => {
    const { name, value } = e.target;
    const setParams = modelType === "DHR" ? setDhrParams : setEsnParams;

    // Allow empty input or partial decimal for decimal fields
    if (value === "" || (decimalFields.includes(name) && value === ".")) {
      setParams((prev) => ({ ...prev, [name]: value }));
      return;
    }

    // For integer fields, only allow digits
    if (integerFields.includes(name)) {
      if (/^\d*$/.test(value)) {
        setParams((prev) => ({ ...prev, [name]: value }));
      }
      return;
    }

    // For decimal fields, allow valid decimal patterns
    if (decimalFields.includes(name)) {
      const isValidDecimal = /^-?\d*\.?\d*$/.test(value);
      if (isValidDecimal) {
        setParams((prev) => ({ ...prev, [name]: value }));
      }
      return;
    }
  };

  const validateForm = () => {
    const newErrors = {};

    const parseValue = (value, isDecimal) => {
      if (value === "" || value === ".") return NaN;
      return isDecimal ? parseFloat(value) : parseInt(value, 10);
    };

    if (model === "DHR") {
      const checks = [
        {
          key: "fourier_terms",
          value: parseValue(dhrParams.fourier_terms, false),
          check: (val) => Number.isInteger(val) && val > 0,
          error: "Must be a positive integer.",
        },
        {
          key: "reg_strength",
          value: parseValue(dhrParams.reg_strength, true),
          check: (val) => !isNaN(val) && val >= 0,
          error: "Must be zero or positive.",
        },
        {
          key: "ar_order",
          value: parseValue(dhrParams.ar_order, false),
          check: (val) => Number.isInteger(val) && val >= 0,
          error: "Must be zero or positive integer.",
        },
        {
          key: "window",
          value: parseValue(dhrParams.window, false),
          check: (val) => Number.isInteger(val) && val > 0,
          error: "Must be a positive integer.",
        },
        {
          key: "polyorder",
          value: parseValue(dhrParams.polyorder, false),
          check: (val) => Number.isInteger(val) && val >= 0,
          error: "Must be zero or positive integer.",
        },
      ];

      checks.forEach(({ key, value, check, error }) => {
        if (!check(value)) {
          newErrors[key] = error;
        }
      });
    } else if (model === "ESN") {
      const checks = [
        {
          key: "lags",
          value: parseValue(esnParams.lags, false),
          check: (val) => Number.isInteger(val) && val > 0,
          error: "Must be a positive integer.",
        },
        {
          key: "N_res",
          value: parseValue(esnParams.N_res, false),
          check: (val) => Number.isInteger(val) && val > 0,
          error: "Must be a positive integer.",
        },
        {
          key: "rho",
          value: parseValue(esnParams.rho, true),
          check: (val) => !isNaN(val) && val > 0 && val <= 1,
          error: "Must be > 0 and ≤ 1.",
        },
        {
          key: "alpha",
          value: parseValue(esnParams.alpha, true),
          check: (val) => !isNaN(val) && val >= 0 && val <= 1,
          error: "Must be between 0 and 1.",
        },
        {
          key: "sparsity",
          value: parseValue(esnParams.sparsity, true),
          check: (val) => !isNaN(val) && val >= 0 && val <= 1,
          error: "Must be between 0 and 1.",
        },
        {
          key: "lambda_reg",
          value: parseValue(esnParams.lambda_reg, true),
          check: (val) => !isNaN(val) && val >= 0,
          error: "Must be zero or positive.",
        },
      ];

      checks.forEach(({ key, value, check, error }) => {
        if (!check(value)) {
          newErrors[key] = error;
        }
      });
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!validateForm()) return;

    setIsLoading(true);
    setMessage("");

    try {
      if (isEditing) {
        await handleUpdate();
      } else {
        await handleCreate();
      }
    } catch (err) {
      console.error(err);
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
    formData.append("energy_demand", energyDemand);
    formData.append("max_capacity", maxCapacity);
    formData.append("temp_id", tempId);

    if (model === "DHR") {
      formData.append("fourier_terms", dhrParams.fourier_terms);
      formData.append("reg_strength", dhrParams.reg_strength);
      formData.append("ar_order", dhrParams.ar_order);
      formData.append("window", dhrParams.window);
      formData.append("polyorder", dhrParams.polyorder);
    } else if (model === "ESN") {
      formData.append("lags", esnParams.lags);
      formData.append("N_res", esnParams.N_res);
      formData.append("rho", esnParams.rho);
      formData.append("alpha", esnParams.alpha);
      formData.append("sparsity", esnParams.sparsity);
      formData.append("lambda_reg", esnParams.lambda_reg);
    }

    const endpoint = `http://localhost:8000/upload/${model.toLowerCase()}/${granularity.toLowerCase()}`;

    const response = await axios.post(endpoint, formData);
    console.log("Upload response:", response);
    const downloadUrls = response.data.download_urls || [];
    const forecastId = response.data.forecast_id;

    const csvUrl = downloadUrls.find((url) => url.endsWith(`_${steps}.csv`));
    const imageUrl = downloadUrls.find((url) => url.endsWith(`_${steps}.png`));

    let forecastData = [];
    if (csvUrl) {
      const csvResponse = await axios.get(csvUrl);
      const lines = csvResponse.data.split("\n").slice(1);
      forecastData = lines
        .filter((line) => line.trim() !== "")
        .map((line) => {
          const [timestamp, forecast] = line.split(",");
          return { timestamp, forecast: parseFloat(forecast) };
        });
    }

    navigate("/result", {
      state: {
        imageUrl,
        energyDemand: energyDemand,
        maxCapacity: maxCapacity,
        forecastData: {
          id: forecastId,
          original_filename: originalFilename,
          forecastType: forecastType,
          model: model,
          steps: steps,
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
    let configData;
    if (model === "DHR") {
      configData = {
        forecast_id: parseInt(forecastId),
        fourier_terms: parseInt(dhrParams.fourier_terms),
        reg_strength: parseFloat(dhrParams.reg_strength),
        ar_order: parseInt(dhrParams.ar_order),
        window: parseInt(dhrParams.window),
        polyorder: parseInt(dhrParams.polyorder),
      };
    } else if (model === "ESN") {
      configData = {
        forecast_id: parseInt(forecastId),
        lags: parseInt(esnParams.lags),
        N_res: parseInt(esnParams.N_res),
        rho: parseFloat(esnParams.rho),
        alpha: parseFloat(esnParams.alpha),
        sparsity: parseFloat(esnParams.sparsity),
        lambda_reg: parseFloat(esnParams.lambda_reg),
      };
    }

    try {
      const endpoint = `http://localhost:8000/api/${model.toLowerCase()}-configurations/${forecastId}`;
      await axios.put(endpoint, configData);
      console.log("Configuration updated successfully");

      const formData = new FormData();
      formData.append("tempFilename", tempFilename);
      formData.append("forecast_type", forecastType);
      formData.append("steps", steps);
      formData.append("forecast_id", forecastId);

      if (model === "DHR") {
        formData.append("fourier_terms", dhrParams.fourier_terms);
        formData.append("reg_strength", dhrParams.reg_strength);
        formData.append("ar_order", dhrParams.ar_order);
        formData.append("window", dhrParams.window);
        formData.append("polyorder", dhrParams.polyorder);
      } else if (model === "ESN") {
        formData.append("lags", esnParams.lags);
        formData.append("N_res", esnParams.N_res);
        formData.append("rho", esnParams.rho);
        formData.append("alpha", esnParams.alpha);
        formData.append("sparsity", esnParams.sparsity);
        formData.append("lambda_reg", esnParams.lambda_reg);
      }

      const endpoint1 = `http://localhost:8000/upload/edit-${model.toLowerCase()}/${granularity.toLowerCase()}`;
      const response = await axios.post(endpoint1, formData);
      console.log("Upload response:", response);

      let imageUrl = null;
      let forecastData = [];

      if (
        response.data &&
        response.data.download_urls &&
        Array.isArray(response.data.download_urls)
      ) {
        const downloadUrls = response.data.download_urls;
        const csvUrl = downloadUrls.find((url) =>
          url.endsWith(`_${steps}.csv`)
        );
        imageUrl = downloadUrls.find((url) => url.endsWith(`_${steps}.png`));

        if (csvUrl) {
          const csvResponse = await axios.get(csvUrl);
          const lines = csvResponse.data.split("\n").slice(1);
          forecastData = lines
            .filter((line) => line.trim() !== "")
            .map((line) => {
              const [timestamp, forecast] = line.split(",");
              return { timestamp, forecast: parseFloat(forecast) };
            });
        }
      } else {
        console.log(
          "No download_urls in response, attempting to fetch forecast directly"
        );
        try {
          const forecastResponse = await axios.get(
            `http://localhost:8000/api/forecasts/${forecastId}`
          );
          console.log("Direct forecast fetch response:", forecastResponse);
          if (forecastResponse.data) {
            if (forecastResponse.data.image_url) {
              imageUrl = forecastResponse.data.image_url;
            }
            if (forecastResponse.data.data) {
              forecastData = forecastResponse.data.data;
            }
          }
        } catch (forecastError) {
          console.error(
            "Failed to fetch forecast data directly:",
            forecastError
          );
        }
      }

      navigate("/result", {
        state: {
          imageUrl,
          energyDemand: energyDemand,
          maxCapacity: maxCapacity,
          forecastData: {
            id: forecastId,
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
      setMessage(`Update failed: ${error.message}`);
      throw error;
    }
  };

  const getModelTitle = () => {
    switch (model) {
      case "DHR":
        return `Dynamic Harmonic Regression`;
      case "ESN":
        return `Echo State Networks`;
      default:
        return `Model Configuration`;
    }
  };

  const renderInput = (name, value, modelType) => (
    <div key={name} className="relative">
      <label className="block text-m font-medium mb-1">
        {customLabels[name]}
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
        onChange={(e) => handleChange(e, modelType)}
        className={`w-full p-2 border rounded-md ${
          errors[name] ? "border-red-500" : ""
        }`}
        step={integerFields.includes(name) ? "1" : "any"}
        min={integerFields.includes(name) ? "0" : undefined}
      />
      {errors[name] && (
        <p className="text-red-600 text-sm mt-1">{errors[name]}</p>
      )}
    </div>
  );

  const renderDHRForm = () => (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="grid grid-cols-2 gap-x-10 gap-y-4">
        {dhrParamOrder.map((key) => renderInput(key, dhrParams[key], "DHR"))}
        <div></div> {/* Empty div for layout alignment */}
      </div>
      <div className="flex justify-end space-x-4 mt-10">
        <button
          type="button"
          onClick={() => navigate(-1)}
          className="px-4 py-2 bg-red-700 border-none text-white rounded-md hover:bg-red-600 focus:outline-none focus:ring-2 focus:ring-red-500 cursor-pointer"
          disabled={isLoading}>
          Cancel
        </button>
        <button
          type="submit"
          className="px-4 py-2 bg-green-500 border-none text-white rounded-md hover:bg-green-600 focus:outline-none focus:ring-2 focus:ring-green-500 cursor-pointer"
          disabled={isLoading}>
          {isEditing ? "Update" : "Submit"}
        </button>
      </div>
    </form>
  );

  const renderESNForm = () => (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="grid grid-cols-2 gap-x-10 gap-y-4">
        {esnParamOrder.map((key) => renderInput(key, esnParams[key], "ESN"))}
      </div>
      <div className="flex justify-end space-x-4 mt-6">
        <button
          type="button"
          onClick={() => navigate(-1)}
          className="px-4 py-2 bg-red-700 border-none text-white rounded-md hover:bg-red-600 focus:outline-none focus:ring-2 focus:ring-red-500 cursor-pointer"
          disabled={isLoading}>
          Cancel
        </button>
        <button
          type="submit"
          className="px-4 py-2 bg-green-500 border-none text-white rounded-md hover:bg-green-600 focus:outline-none focus:ring-2 focus:ring-green-500 cursor-pointer"
          disabled={isLoading}>
          {isEditing ? "Update" : "Submit"}
        </button>
      </div>
    </form>
  );

  const renderForm = () => {
    if (isLoading && !model) {
      return <div className="text-center py-6">Loading configuration...</div>;
    }
    return model === "ESN" ? renderESNForm() : renderDHRForm();
  };

  // CSS for the loading bar
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

  return (
    <div className="min-h-screen relative">
      <div className="fixed inset-0 bg-gray-100" style={{ zIndex: -1 }} />
      <div className="relative z-10 flex justify-center flex-1 min-h-screen">
        <div className="w-1/3 h-1/3 p-10 px-20 bg-white rounded-lg shadow-md mb-10">
          <style>{loadingBarStyles}</style>
          <h2 className="text-4xl text-left font-bold mb-10">
            {getModelTitle()}
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
          {renderForm()}
        </div>
      </div>
    </div>
  );
}

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
    // Default parameters for solar forecast
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
  };

  // Initialize state with default values based on forecastType and granularity
  const getDefaultEsnParams = () => {
    // Default parameters for solar forecast
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
  };

  const [dhrParams, setDhrParams] = useState(getDefaultDhrParams());

  const [esnParams, setEsnParams] = useState(getDefaultEsnParams());

  const [message, setMessage] = useState("");
  const [tooltipVisible, setTooltipVisible] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

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
          lags: existingConfig.lags?.toString(),
          N_res: existingConfig.N_res?.toString(),
          rho: existingConfig.rho?.toString(),
          alpha: existingConfig.alpha?.toString(),
          sparsity: existingConfig.sparsity?.toString(),
          lambda_reg: existingConfig.lambda_reg?.toString(),
        });
      }
    } else if (isEditing && forecastId) {
      // Fetch configuration if we have forecastId but no config
      fetchExistingConfig();
    }
  }, [isEditing, existingConfig, model, forecastId, forecastType, granularity]);

  // Function to fetch existing configuration
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
          lags: config.lags?.toString(),
          N_res: config.N_res?.toString(),
          rho: config.rho?.toString(),
          alpha: config.alpha?.toString(),
          sparsity: config.sparsity?.toString(),
          lambda_reg: config.lambda_reg?.toString(),
        });
      }
    } catch (error) {
      console.error("Error fetching configuration:", error);
      setMessage("Failed to load configuration.");
    } finally {
      setIsLoading(false);
    }
  };

  const showTooltip = (field) => setTooltipVisible(field);
  const hideTooltip = () => setTooltipVisible(null);

  const handleChange = (e, modelType) => {
    const { name, value } = e.target;
    if (modelType === "DHR") {
      setDhrParams((prev) => ({ ...prev, [name]: value }));
    } else if (modelType === "ESN") {
      setEsnParams((prev) => ({ ...prev, [name]: value }));
    }
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

  // Modified handleSubmit function
  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setMessage(""); // Clear any previous message

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
    formData.append("temp_id", tempId);

    const endpoint = `http://localhost:8000/upload/${model.toLowerCase()}/${granularity.toLowerCase()}`;

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
    // First, update the configuration
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
      // First API call to update configuration
      const endpoint = `http://localhost:8000/api/${model.toLowerCase()}-configurations/${forecastId}`;
      await axios.put(endpoint, configData);
      console.log("Configuration updated successfully");

      // Prepare form data for the second API call
      const formData = new FormData();
      formData.append("tempFilename", tempFilename);
      formData.append("forecast_type", forecastType);
      formData.append("steps", steps);
      formData.append("forecast_id", forecastId); // Add forecast_id here

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

      // Second API call to generate new forecast
      const endpoint1 = `http://localhost:8000/upload/edit-${model.toLowerCase()}/${granularity.toLowerCase()}`;
      const response = await axios.post(endpoint1, formData);
      console.log("Upload response:", response);

      let imageUrl = null;
      let forecastData = [];

      // Check if the response has download_urls
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

        // Fetch forecast data if CSV URL is available
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

        // Alternative approach: fetch the forecast data directly using the forecast ID
        try {
          const forecastResponse = await axios.get(
            `http://localhost:8000/api/forecasts/${forecastId}`
          );
          console.log("Direct forecast fetch response:", forecastResponse);

          if (forecastResponse.data) {
            // Try to get image URL from the forecast data if available
            if (forecastResponse.data.image_url) {
              imageUrl = forecastResponse.data.image_url;
            }

            // Try to get forecast data if available
            if (forecastResponse.data.data) {
              forecastData = forecastResponse.data.data;
            }
          }
        } catch (forecastError) {
          console.error(
            "Failed to fetch forecast data directly:",
            forecastError
          );
          // Continue with empty data since we at least updated the configuration
        }
      }

      // Navigate to result page with whatever data we have
      navigate("/result", {
        state: {
          imageUrl,
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
      throw error; // Re-throw to be caught by handleSubmit
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

  const renderDHRForm = () => (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="flex space-x-10">
        <div className="flex-1 relative">
          <label className="block text-m font-medium mb-1">
            Fourier Order
            <span
              className="text-gray-500 cursor-pointer ml-2"
              onMouseEnter={() => showTooltip("fourier_terms")}
              onMouseLeave={hideTooltip}>
              ⓘ
            </span>
            {tooltipVisible === "fourier_terms" && (
              <div className="absolute bg-gray-800 text-white p-2 rounded text-xs max-w-xs z-10">
                Number of Fourier terms for capturing seasonality
              </div>
            )}
          </label>
          <input
            type="text"
            name="fourier_terms"
            value={dhrParams.fourier_terms}
            onChange={(e) => handleChange(e, "DHR")}
            className="w-full p-2 border rounded-md"
          />
        </div>
        <div className="flex-1 relative">
          <label className="block text-m font-medium mb-1">
            Window Length
            <span
              className="text-gray-500 cursor-pointer ml-2"
              onMouseEnter={() => showTooltip("window")}
              onMouseLeave={hideTooltip}>
              ⓘ
            </span>
            {tooltipVisible === "window" && (
              <div className="absolute bg-gray-800 text-white p-2 rounded text-xs max-w-xs z-10">
                Window size for smoothing
              </div>
            )}
          </label>
          <input
            type="text"
            name="window"
            value={dhrParams.window}
            onChange={(e) => handleChange(e, "DHR")}
            className="w-full p-2 border rounded-md"
          />
        </div>
      </div>
      <div className="flex space-x-10">
        <div className="flex-1 relative">
          <label className="block text-m font-medium mb-1">
            Polyorder
            <span
              className="text-gray-500 cursor-pointer ml-2"
              onMouseEnter={() => showTooltip("polyorder")}
              onMouseLeave={hideTooltip}>
              ⓘ
            </span>
            {tooltipVisible === "polyorder" && (
              <div className="absolute bg-gray-800 text-white p-2 rounded text-xs max-w-xs z-10">
                Order of polynomial for Savitzky-Golay smoothing
              </div>
            )}
          </label>
          <input
            type="text"
            name="polyorder"
            value={dhrParams.polyorder}
            onChange={(e) => handleChange(e, "DHR")}
            className="w-full p-2 border rounded-md"
          />
        </div>
        <div className="flex-1">
          <label className="block text-m font-medium mb-1">
            Regularization
            <span
              className="text-gray-500 cursor-pointer ml-2"
              onMouseEnter={() => showTooltip("reg_strength")}
              onMouseLeave={hideTooltip}>
              ⓘ
            </span>
            {tooltipVisible === "reg_strength" && (
              <div className="absolute bg-gray-800 text-white p-2 rounded text-xs max-w-xs z-10">
                Ridge regression regularization parameter
              </div>
            )}
          </label>
          <input
            type="text"
            name="reg_strength"
            value={dhrParams.reg_strength}
            onChange={(e) => handleChange(e, "DHR")}
            className="w-full p-2 border rounded-md"
          />
        </div>
      </div>
      <div className="flex space-x-10">
        <div className="flex-1">
          <label className="block text-m font-medium mb-1">
            AR Order
            <span
              className="text-gray-500 cursor-pointer ml-2"
              onMouseEnter={() => showTooltip("trend_components")}
              onMouseLeave={hideTooltip}>
              ⓘ
            </span>
            {tooltipVisible === "trend_components" && (
              <div className="absolute bg-gray-800 text-white p-2 rounded text-xs max-w-xs z-10">
                Number of past values to use for autoregression
              </div>
            )}
          </label>
          <input
            type="text"
            name="ar_order"
            onChange={(e) => handleChange(e, "DHR")}
            value={dhrParams.ar_order}
            className="w-full p-2 border rounded-md"
          />
        </div>
        <div className="flex-1"></div>
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

  // Modified renderESNForm function
  const renderESNForm = () => (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="flex space-x-10">
        <div className="flex-1">
          <label className="block text-m font-medium mb-1">
            Reservoir Size
            <span
              className="text-gray-500 cursor-pointer ml-2"
              onMouseEnter={() => showTooltip("N_res")}
              onMouseLeave={hideTooltip}>
              ⓘ
            </span>
            {tooltipVisible === "N_res" && (
              <div className="absolute bg-gray-800 text-white p-2 rounded text-xs max-w-xs z-10">
                Number of neurons in the reservoir
              </div>
            )}
          </label>
          <input
            type="text"
            name="N_res"
            value={esnParams.N_res}
            onChange={(e) => handleChange(e, "ESN")}
            className="w-full p-2 border rounded-md"
          />
        </div>
        <div className="flex-1">
          <label className="block text-m font-medium mb-1">
            Regularization
            <span
              className="text-gray-500 cursor-pointer ml-2"
              onMouseEnter={() => showTooltip("lambda_reg")}
              onMouseLeave={hideTooltip}>
              ⓘ
            </span>
            {tooltipVisible === "lambda_reg" && (
              <div className="absolute bg-gray-800 text-white p-2 rounded text-xs max-w-xs z-10">
                Regularization parameter for ESN training. Helps prevent
                overfitting
              </div>
            )}
          </label>
          <input
            type="text"
            name="lambda_reg"
            value={esnParams.lambda_reg}
            onChange={(e) => handleChange(e, "ESN")}
            className="w-full p-2 border rounded-md"
          />
        </div>
      </div>
      <div className="flex space-x-10">
        <div className="flex-1">
          <label className="block text-m font-medium mb-1">
            Spectral Radius
            <span
              className="text-gray-500 cursor-pointer ml-2"
              onMouseEnter={() => showTooltip("rho")}
              onMouseLeave={hideTooltip}>
              ⓘ
            </span>
            {tooltipVisible === "rho" && (
              <div className="absolute bg-gray-800 text-white p-2 rounded text-xs max-w-xs z-10">
                Controls echo state property
              </div>
            )}
          </label>
          <input
            type="text"
            name="rho"
            value={esnParams.rho}
            onChange={(e) => handleChange(e, "ESN")}
            className="w-full p-2 border rounded-md"
          />
        </div>
        <div className="flex-1">
          <label className="block text-m font-medium mb-1">
            Input Scaling
            <span
              className="text-gray-500 cursor-pointer ml-2"
              onMouseEnter={() => showTooltip("alpha")}
              onMouseLeave={hideTooltip}>
              ⓘ
            </span>
            {tooltipVisible === "alpha" && (
              <div className="absolute bg-gray-800 text-white p-2 rounded text-xs max-w-xs z-10">
                Scaling factor applied to the input data
              </div>
            )}
          </label>
          <input
            type="text"
            name="alpha"
            value={esnParams.alpha}
            onChange={(e) => handleChange(e, "ESN")}
            className="w-full p-2 border rounded-md"
          />
        </div>
      </div>
      <div className="flex space-x-10">
        <div className="flex-1">
          <label className="block text-m font-medium mb-1">
            Sparsity
            <span
              className="text-gray-500 cursor-pointer ml-2"
              onMouseEnter={() => showTooltip("sparsity")}
              onMouseLeave={hideTooltip}>
              ⓘ
            </span>
            {tooltipVisible === "sparsity" && (
              <div className="absolute bg-gray-800 text-white p-2 rounded text-xs max-w-xs z-10">
                Reservoir connectivity
              </div>
            )}
          </label>
          <input
            type="text"
            name="sparsity"
            value={esnParams.sparsity}
            onChange={(e) => handleChange(e, "ESN")}
            className="w-full p-2 border rounded-md"
          />
        </div>
        <div className="flex-1">
          <label className="block text-m font-medium mb-1">
            Lags
            <span
              className="text-gray-500 cursor-pointer ml-2"
              onMouseEnter={() => showTooltip("lags")}
              onMouseLeave={hideTooltip}>
              ⓘ
            </span>
            {tooltipVisible === "lags" && (
              <div className="absolute bg-gray-800 text-white p-2 rounded text-xs max-w-xs z-10">
                Number of time steps as input
              </div>
            )}
          </label>
          <input
            type="text"
            name="lags"
            value={esnParams.lags}
            onChange={(e) => handleChange(e, "ESN")}
            className="w-full p-2 border rounded-md"
          />
        </div>
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

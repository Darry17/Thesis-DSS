import React, { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";

const ForecastResult = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { forecastId } = location.state || {};
  const [configuration, setConfiguration] = useState(null);
  const [forecastData, setForecastData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [recommendations, setRecommendations] = useState({
    batteryStorage: true,
    bidStrategically: true,
  });

  useEffect(() => {
    const fetchData = async () => {
      try {
        if (!forecastId) {
          throw new Error("Missing forecast ID");
        }

        // First fetch the forecast basic data
        const forecastResponse = await fetch(
          `http://localhost:8000/api/forecasts/${forecastId}`
        );
        if (!forecastResponse.ok) {
          throw new Error("Failed to fetch forecast data");
        }

        const forecastData = await forecastResponse.json();
        setForecastData(forecastData);

        // Use the model from forecast data
        const model = forecastData.model;

        // Then fetch the configuration based on model type
        let configEndpoint;
        switch (model) {
          case "DHR":
            configEndpoint = `/api/dhr-configurations/${forecastId}`;
            break;
          case "ESN":
            configEndpoint = `/api/esn-configurations/${forecastId}`;
            break;
          case "DHR-ESN":
            configEndpoint = `/api/hybrid-configurations/${forecastId}`;
            break;
          default:
            throw new Error(`Unknown model type: ${model}`);
        }

        const configResponse = await fetch(
          `http://localhost:8000${configEndpoint}`
        );

        if (!configResponse.ok) {
          throw new Error(`Failed to fetch ${model} configuration`);
        }

        const configData = await configResponse.json();

        setConfiguration(configData);
      } catch (error) {
        console.error("Error fetching data:", error);
        setError(error.message);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [forecastId]);

  // Function to get step label
  const getStepLabel = (steps, granularity) => {
    if (!steps || !granularity) return "No steps available";

    switch (granularity) {
      case "Hourly":
        switch (steps) {
          case "1-hour":
            return "1 Step (1-Hour Horizon)";
          case "24-hour":
            return "24 Steps (1-Day Horizon)";
          case "168-hour":
            return "168 Steps (1-Week Horizon)";
          default:
            return steps;
        }
      case "Daily":
        switch (steps) {
          case "1-day":
            return "1 Step (1-Day Horizon)";
          case "7-day":
            return "7 Steps (1-Week Horizon)";
          case "30-day":
            return "30 Steps (1-Month Horizon)";
          default:
            return steps;
        }
      case "Weekly":
        switch (steps) {
          case "1-week":
            return "1 Step (1-Week Horizon)";
          case "4-week":
            return "4 Steps (1-Month Horizon)";
          case "52-week":
            return "52 Steps (1-Year Horizon)";
          default:
            return steps;
        }
      default:
        return steps || "No steps available";
    }
  };

  // Updated renderConfigSection to handle hybrid model
  const renderConfigSection = () => {
    if (!configuration) return <div>Loading configuration...</div>;

    if (forecastData.model === "DHR-ESN") {
      return (
        <div className="space-y-8">
          {/* DHR Section */}
          <div>
            <h3 className="text-lg font-semibold mb-4">
              Dynamic Harmonic Regression
            </h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-gray-600">Fourier Order</p>
                <input
                  type="text"
                  value={configuration?.fourier_order || "-"}
                  disabled
                  className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
                />
              </div>
              <div>
                <p className="text-sm text-gray-600">Window Length</p>
                <input
                  type="text"
                  value={configuration?.window_length || "-"}
                  disabled
                  className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
                />
              </div>
              <div>
                <p className="text-sm text-gray-600">Seasonality Periods</p>
                <input
                  type="text"
                  value={configuration?.seasonality_periods || "-"}
                  disabled
                  className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
                />
              </div>
              <div>
                <p className="text-sm text-gray-600">Polyorder</p>
                <input
                  type="text"
                  value={configuration?.polyorder || "-"}
                  disabled
                  className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
                />
              </div>
              <div>
                <p className="text-sm text-gray-600">Regularization (DHR)</p>
                <input
                  type="text"
                  value={configuration?.regularization_dhr || "-"}
                  disabled
                  className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
                />
              </div>
              <div>
                <p className="text-sm text-gray-600">Trend Components</p>
                <input
                  type="text"
                  value={configuration?.trend_components || "-"}
                  disabled
                  className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
                />
              </div>
            </div>
          </div>

          {/* ESN Section */}
          <div>
            <h3 className="text-lg font-semibold mb-4">Echo State Networks</h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-gray-600">Reservoir Size</p>
                <input
                  type="text"
                  value={configuration?.reservoir_size || "-"}
                  disabled
                  className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
                />
              </div>
              <div>
                <p className="text-sm text-gray-600">Spectral Radius</p>
                <input
                  type="text"
                  value={configuration?.spectral_radius || "-"}
                  disabled
                  className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
                />
              </div>
              <div>
                <p className="text-sm text-gray-600">Sparsity</p>
                <input
                  type="text"
                  value={configuration?.sparsity || "-"}
                  disabled
                  className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
                />
              </div>
              <div>
                <p className="text-sm text-gray-600">Input Scaling</p>
                <input
                  type="text"
                  value={configuration?.input_scaling || "-"}
                  disabled
                  className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
                />
              </div>
              <div>
                <p className="text-sm text-gray-600">Dropout</p>
                <input
                  type="text"
                  value={configuration?.dropout || "-"}
                  disabled
                  className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
                />
              </div>
              <div>
                <p className="text-sm text-gray-600">Lags</p>
                <input
                  type="text"
                  value={configuration?.lags || "-"}
                  disabled
                  className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
                />
              </div>
              <div>
                <p className="text-sm text-gray-600">Regularization (ESN)</p>
                <input
                  type="text"
                  value={configuration?.regularization_esn || "-"}
                  disabled
                  className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
                />
              </div>
            </div>
          </div>
        </div>
      );
    } else if (forecastData.model === "DHR") {
      return (
        <div className="mb-8">
          <h3 className="text-lg font-semibold mb-4">
            Dynamic Harmonic Regression
          </h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-gray-600">Fourier Order</p>
              <input
                type="text"
                value={configuration?.fourier_order || "-"}
                disabled
                className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
              />
            </div>
            <div>
              <p className="text-sm text-gray-600">Trend Components</p>
              <input
                type="text"
                value={configuration?.trend_components || "-"}
                disabled
                className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
              />
            </div>
            <div>
              <p className="text-sm text-gray-600">Seasonality Periods</p>
              <input
                type="text"
                value={configuration?.seasonality_periods || "-"}
                disabled
                className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
              />
            </div>
            <div>
              <p className="text-sm text-gray-600">Window Length</p>
              <input
                type="text"
                value={configuration?.window_length || "-"}
                disabled
                className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
              />
            </div>
            <div>
              <p className="text-sm text-gray-600">Regularization</p>
              <input
                type="text"
                value={configuration?.regularization_dhr || "-"}
                disabled
                className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
              />
            </div>
            <div>
              <p className="text-sm text-gray-600">Polyorder</p>
              <input
                type="text"
                value={configuration?.polyorder || "-"}
                disabled
                className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
              />
            </div>
          </div>
        </div>
      );
    } else if (forecastData.model === "ESN") {
      return (
        <div>
          <h3 className="text-lg font-semibold mb-4">Echo State Networks</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-gray-600">Reservoir Size</p>
              <input
                type="text"
                value={configuration?.reservoir_size || "-"}
                disabled
                className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
              />
            </div>
            <div>
              <p className="text-sm text-gray-600">Input Scaling</p>
              <input
                type="text"
                value={configuration?.input_scaling || "-"}
                disabled
                className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
              />
            </div>
            <div>
              <p className="text-sm text-gray-600">Spectral Radius</p>
              <input
                type="text"
                value={configuration?.spectral_radius || "-"}
                disabled
                className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
              />
            </div>
            <div>
              <p className="text-sm text-gray-600">Regularization</p>
              <input
                type="text"
                value={configuration?.regularization_esn || "-"}
                disabled
                className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
              />
            </div>
            <div>
              <p className="text-sm text-gray-600">Sparsity</p>
              <input
                type="text"
                value={configuration?.sparsity || "-"}
                disabled
                className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
              />
            </div>
            <div>
              <p className="text-sm text-gray-600">Dropout</p>
              <input
                type="text"
                value={configuration?.dropout || "-"}
                disabled
                className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
              />
            </div>
            <div>
              <p className="text-sm text-gray-600">Lags</p>
              <input
                type="text"
                value={configuration?.lags || "-"}
                disabled
                className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
              />
            </div>
          </div>
        </div>
      );
    }
  };

  const handleEdit = () => {
    // Navigate to the appropriate configuration page based on model type
    if (forecastData.model === "DHR-ESN") {
      navigate("/HybridModelConfiguration", {
        state: {
          forecastId,
          isEditing: true,
          existingConfig: configuration,
        },
      });
    } else {
      navigate("/SingleModelConfiguration", {
        state: {
          forecastId,
          model: forecastData.model,
          isEditing: true,
          existingConfig: configuration,
        },
      });
    }
  };

  if (loading) {
    return <div className="p-6 text-center">Loading forecast data...</div>;
  }

  if (error) {
    return <div className="p-6 text-red-500">Error: {error}</div>;
  }

  return (
    <div className="p-6 max-w-5xl mx-auto">
      <div className="grid grid-cols-2 gap-6">
        {/* Left Column */}
        <div className="space-y-6">
          {/* Graph Card */}
          <div className="bg-white p-4 rounded-lg shadow">
            <h3 className="text-sm text-gray-500 mb-2">
              {forecastData?.filename || "No filename available"}
            </h3>
            <h2 className="font-bold mb-4">
              {forecastData?.model || "No model selected"}
            </h2>
            <div className="h-48 bg-gray-100 rounded mb-4">
              {/* Graph will go here */}
            </div>
            <div className="text-sm">View Results</div>
          </div>

          {/* Forecast Period */}
          <div className="bg-white p-4 rounded-lg shadow">
            <h3 className="text-gray-700 font-semibold mb-2">
              Generated Forecast
            </h3>
            <p className="text-xl font-bold">
              {getStepLabel(forecastData?.steps, forecastData?.granularity)}
            </p>
          </div>

          {/* Recommendations */}
          <div className="bg-white p-4 rounded-lg shadow">
            <h3 className="text-gray-700 font-semibold mb-4">
              Recommendations
            </h3>
            <p className="text-sm text-gray-600 mb-4">
              Production of solar energy will be 25% lower compared to the last
              24 hours
            </p>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-green-600 font-medium">A</span>
                  <p className="text-sm">
                    Pre-charge battery storage overnight
                  </p>
                </div>
                <input
                  type="checkbox"
                  checked={recommendations.batteryStorage}
                  onChange={(e) =>
                    setRecommendations((prev) => ({
                      ...prev,
                      batteryStorage: e.target.checked,
                    }))
                  }
                  className="h-4 w-4 text-green-600 rounded border-green-600 focus:ring-green-500 cursor-pointer"
                />
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-green-600 font-medium">A</span>
                  <p className="text-sm">
                    Bid strategically in the day-ahead market
                  </p>
                </div>
                <input
                  type="checkbox"
                  checked={recommendations.bidStrategically}
                  onChange={(e) =>
                    setRecommendations((prev) => ({
                      ...prev,
                      bidStrategically: e.target.checked,
                    }))
                  }
                  className="h-4 w-4 text-green-600 rounded border-green-600 focus:ring-green-500 cursor-pointer"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Right Column - Configurations */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-2xl font-bold mb-6">Configurations</h2>
          {configuration ? (
            renderConfigSection()
          ) : (
            <div>Loading configuration...</div>
          )}
          {/* Edit Button */}
          <button
            onClick={handleEdit}
            className="mt-8 bg-green-500 text-white px-6 py-2 rounded-md hover:bg-green-600 w-full">
            Edit
          </button>
        </div>
      </div>
    </div>
  );
};

export default ForecastResult;

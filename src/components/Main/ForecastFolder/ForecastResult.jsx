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

        // Fetch the forecast basic data
        const forecastResponse = await fetch(
          `http://localhost:8000/api/forecasts/${forecastId}`
        );
        if (!forecastResponse.ok) {
          throw new Error("Failed to fetch forecast data");
        }

        const forecastData = await forecastResponse.json();
        setForecastData(forecastData);

        // Determine configuration endpoint based on model type
        const model = forecastData.model;
        const configEndpoints = {
          DHR: `/api/dhr-configurations/${forecastId}`,
          ESN: `/api/esn-configurations/${forecastId}`,
          "DHR-ESN": `/api/hybrid-configurations/${forecastId}`,
        };

        const configEndpoint = configEndpoints[model];
        if (!configEndpoint) {
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

    const stepLabels = {
      Hourly: {
        "1-hour": "1 Step (1-Hour Horizon)",
        "24-hour": "24 Steps (1-Day Horizon)",
        "168-hour": "168 Steps (1-Week Horizon)",
      },
      Daily: {
        "1-day": "1 Step (1-Day Horizon)",
        "7-day": "7 Steps (1-Week Horizon)",
        "30-day": "30 Steps (1-Month Horizon)",
      },
      Weekly: {
        "1-week": "1 Step (1-Week Horizon)",
        "4-week": "4 Steps (1-Month Horizon)",
        "52-week": "52 Steps (1-Year Horizon)",
      },
    };

    return stepLabels[granularity]?.[steps] || steps || "No steps available";
  };

  // Render configuration section based on model type
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
      navigate("/hybrid-model-config", {
        state: {
          forecastId,
          isEditing: true,
          existingConfig: configuration,
        },
      });
    } else {
      navigate("/single-model-config", {
        state: {
          forecastId,
          model: forecastData.model,
          isEditing: true,
          existingConfig: configuration,
        },
      });
    }
  };

  const handleViewGraphs = () => {
    navigate("/view-graph", {
      state: { forecastId },
    });
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
              {forecastData?.original_filename || "No filename available"}
            </h3>
            <h2 className="font-bold mb-4">
              {forecastData?.model || "No model selected"}
            </h2>
            <div className="h-48 bg-gray-100 rounded mb-4">
              {/* Graph will go here */}
            </div>

            <button
              onClick={handleViewGraphs}
              className="w-30 mt-2 py-2 bg-white text-black rounded-md hover:bg-gray-50 border-b-2 border-green-500">
              View Graphs
            </button>
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
          <div className="flex space-x-4 mt-8">
            <button
              onClick={handleEdit}
              className="bg-green-500 text-white px-6 py-2 rounded-md hover:bg-green-600">
              Edit
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ForecastResult;

import React, { useState, useEffect } from "react";
import { useLocation } from "react-router-dom";

const ForecastResult = () => {
  const location = useLocation();
  const { filename, forecast_model, steps, granularity, forecastId } =
    location.state || {};
  const [configuration, setConfiguration] = useState(null);
  const [recommendations, setRecommendations] = useState({
    batteryStorage: true,
    bidStrategically: true,
  });

  // Fetch configuration data when component mounts
  useEffect(() => {
    const fetchConfiguration = async () => {
      try {
        if (!forecastId) return;

        const endpoint =
          forecast_model === "DHR"
            ? `/api/dhr-configurations/${forecastId}`
            : `/api/esn-configurations/${forecastId}`;

        const response = await fetch(`http://localhost:8000${endpoint}`);
        if (!response.ok) throw new Error("Failed to fetch configuration");

        const data = await response.json();
        setConfiguration(data);
      } catch (error) {
        console.error("Error fetching configuration:", error);
      }
    };

    fetchConfiguration();
  }, [forecastId, forecast_model]);

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

  // Function to determine which configuration section to show
  const renderConfigSection = () => {
    if (forecast_model === "DHR") {
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
    } else if (forecast_model === "ESN") {
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

  return (
    <div className="p-6 max-w-5xl mx-auto">
      <div className="grid grid-cols-2 gap-6">
        {/* Left Column */}
        <div className="space-y-6">
          {/* Graph Card */}
          <div className="bg-white p-4 rounded-lg shadow">
            <h3 className="text-sm text-gray-500 mb-2">
              {filename || "No filename available"}
            </h3>
            <h2 className="font-bold mb-4">
              {forecast_model || "No model selected"}
            </h2>
            {/* Replace with actual graph component */}
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
              {getStepLabel(steps, granularity)}
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
          {renderConfigSection()}
          {/* Edit Button */}
          <button className="mt-8 bg-green-500 text-white px-6 py-2 rounded-md hover:bg-green-600 w-full">
            Edit
          </button>
        </div>
      </div>
    </div>
  );
};

export default ForecastResult;

import React, { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";

const ViewLogs = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { forecastId } = location.state || {};
  const [forecastData, setForecastData] = useState(null);
  const [historyLog, setHistoryLog] = useState(null);
  const [esnConfig, setEsnConfig] = useState(null);
  const [dhrConfig, setDhrConfig] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (forecastId) {
      fetchForecastData();
      fetchHistoryLog();
      fetchModelConfigurations();
    } else {
      setError("No forecast ID provided");
      console.error("No forecast ID provided");
    }
  }, [forecastId]);

  const fetchForecastData = async () => {
    try {
      const response = await fetch(
        `http://localhost:8000/api/forecasts/${forecastId}`
      );

      if (!response.ok) {
        throw new Error(`Error fetching forecast: ${response.statusText}`);
      }

      const data = await response.json();
      setForecastData(data);

      if (data.filename) {
        await fetchFileContent(data.filename);
      }
    } catch (error) {
      console.error("Error fetching forecast data:", error);
      setError(error.message);
    }
  };

  const fetchFileContent = async (filename) => {
    try {
      const fileResponse = await fetch(
        `http://localhost:8000/storage/read/${filename}`
      );

      if (!fileResponse.ok) {
        throw new Error(
          `Failed to read file content: ${fileResponse.statusText}`
        );
      }

      const jsonContent = await fileResponse.json();
    } catch (error) {
      console.error("Error fetching file content:", error);
      setError(error.message);
    }
  };

  const fetchModelConfigurations = async () => {
    try {
      const forecastResponse = await fetch(
        `http://localhost:8000/api/forecasts/${forecastId}`
      );

      if (!forecastResponse.ok) {
        throw new Error(
          `Error fetching forecast: ${forecastResponse.statusText}`
        );
      }

      const forecastData = await forecastResponse.json();
      const modelType = forecastData.model?.toUpperCase();

      const configEndpoints = {
        DHR: `/api/dhr-configurations/${forecastId}`,
        ESN: `/api/esn-configurations/${forecastId}`,
        "DHR-ESN": `/api/hybrid-configurations/${forecastId}`,
        HYBRID: `/api/hybrid-configurations/${forecastId}`,
      };

      if (modelType && configEndpoints[modelType]) {
        const endpoint = configEndpoints[modelType];
        const response = await fetch(`http://localhost:8000${endpoint}`);

        if (response.ok) {
          const configData = await response.json();

          if (modelType === "ESN") {
            setEsnConfig(configData);
          } else if (modelType === "DHR") {
            setDhrConfig(configData);
          } else if (modelType === "DHR-ESN" || modelType === "HYBRID") {
            setEsnConfig(configData.esn_config);
            setDhrConfig(configData.dhr_config);
          }
        } else {
          throw new Error(`Error fetching config: ${response.statusText}`);
        }
      }
    } catch (error) {
      console.error("Error fetching model configurations:", error);
      setError(error.message);
    }
  };

  const fetchHistoryLog = async () => {
    try {
      const response = await fetch("http://localhost:8000/api/history-logs");

      if (!response.ok) {
        throw new Error(`Error fetching history logs: ${response.statusText}`);
      }

      const data = await response.json();
      const logs = Array.isArray(data.logs) ? data.logs : [];

      if (!Array.isArray(data.logs)) {
        console.warn("Expected an array in data.logs, received:", data);
      }

      const matchingLog = logs.find((log) => log.forecast_id === forecastId);

      if (matchingLog) {
        setHistoryLog(matchingLog);
      } else {
        console.warn(`No history log found for forecast ID: ${forecastId}`);
        setHistoryLog(null);
      }
    } catch (error) {
      console.error("Error fetching history logs:", error);
      setError(error.message);
    }
  };

  const handleBack = () => navigate(-1);

  const GraphPlaceholder = () => (
    <div className="bg-white p-4 rounded-lg border-gray-500 shadow mb-5">
      <h3 className="text-lg font-medium mb-2">Generated Power</h3>
      <div className="h-60 bg-gray-50 flex justify-center items-center">
        <div className="text-gray-500">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-16 w-16 mx-auto mb-4"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
            />
          </svg>
          <p className="text-xl font-medium">
            Graph Visualization Not Available
          </p>
          <p className="mt-2">
            Real-time graph data would be displayed here in a production
            environment.
          </p>
        </div>
      </div>
    </div>
  );

  const EsnConfiguration = () => (
    <div className="p-4 bg-white mb-6">
      <h3 className="text-2xl font-bold mb-6">
        Configurations - Echo State Networks
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Reservoir Size
          </label>
          <input
            type="text"
            value={esnConfig?.reservoir_size || "500"}
            readOnly
            className="w-full p-2 bg-gray-50 border border-gray-300 rounded-md text-gray-700 focus:outline-none cursor-default"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Regularization
          </label>
          <input
            type="text"
            value={esnConfig?.regularization || "0.2"}
            readOnly
            className="w-full p-2 bg-gray-50 border border-gray-300 rounded-md text-gray-700 focus:outline-none cursor-default"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Input Scaling
          </label>
          <input
            type="text"
            value={esnConfig?.input_scaling || "0.3"}
            readOnly
            className="w-full p-2 bg-gray-50 border border-gray-300 rounded-md text-gray-700 focus:outline-none cursor-default"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Dropouts
          </label>
          <input
            type="text"
            value={esnConfig?.dropouts || "1"}
            readOnly
            className="w-full p-2 bg-gray-50 border border-gray-300 rounded-md text-gray-700 focus:outline-none cursor-default"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Sparsity
          </label>
          <input
            type="text"
            value={esnConfig?.sparsity || "1.0"}
            readOnly
            className="w-full p-2 bg-gray-50 border border-gray-300 rounded-md text-gray-700 focus:outline-none cursor-default"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Spectral Radius
          </label>
          <input
            type="text"
            value={esnConfig?.spectral_radius || "0.9"}
            readOnly
            className="w-full p-2 bg-gray-50 border border-gray-300 rounded-md text-gray-700 focus:outline-none cursor-default"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Lags
          </label>
          <input
            type="text"
            value={esnConfig?.lags || "1"}
            readOnly
            className="w-full p-2 bg-gray-50 border border-gray-300 rounded-md text-gray-700 focus:outline-none cursor-default"
          />
        </div>
      </div>
    </div>
  );

  const DhrConfiguration = () => (
    <div className="p-4 bg-white mb-6">
      <h3 className="text-2xl font-bold mb-6">
        Configurations - Dynamic Harmonic Regression
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Fourier Order
          </label>
          <input
            type="text"
            value={dhrConfig?.fourier_order || "3"}
            readOnly
            className="w-full p-2 bg-gray-50 border border-gray-300 rounded-md text-gray-700 focus:outline-none cursor-default"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Seasonality Periods
          </label>
          <input
            type="text"
            value={dhrConfig?.seasonality_periods || "M"}
            readOnly
            className="w-full p-2 bg-gray-50 border border-gray-300 rounded-md text-gray-700 focus:outline-none cursor-default"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Regularization
          </label>
          <input
            type="text"
            value={dhrConfig?.regularization || "1e-4"}
            readOnly
            className="w-full p-2 bg-gray-50 border border-gray-300 rounded-md text-gray-700 focus:outline-none cursor-default"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Trend Components
          </label>
          <input
            type="text"
            value={dhrConfig?.trend_components || "2"}
            readOnly
            className="w-full p-2 bg-gray-50 border border-gray-300 rounded-md text-gray-700 focus:outline-none cursor-default"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Window Length
          </label>
          <input
            type="text"
            value={dhrConfig?.window_length || "1"}
            readOnly
            className="w-full p-2 bg-gray-50 border border-gray-300 rounded-md text-gray-700 focus:outline-none cursor-default"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Polyorder
          </label>
          <input
            type="text"
            value={dhrConfig?.polyorder || "0.1"}
            readOnly
            className="w-full p-2 bg-gray-50 border border-gray-300 rounded-md text-gray-700 focus:outline-none cursor-default"
          />
        </div>
      </div>
    </div>
  );

  const DatasetDetails = () => (
    <div className="p-4 bg-white">
      <h3 className="text-2xl font-bold mb-6">Dataset Details</h3>
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Dataset
          </label>
          <input
            type="text"
            value={forecastData?.original_filename || ""}
            readOnly
            className="w-full p-2 bg-gray-50 border border-gray-300 rounded-md text-gray-700 focus:outline-none cursor-default"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Granularity
          </label>
          <input
            type="text"
            value={forecastData?.granularity || ""}
            readOnly
            className="w-full p-2 bg-gray-50 border border-gray-300 rounded-md text-gray-700 focus:outline-none cursor-default"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Steps
          </label>
          <input
            type="text"
            value={forecastData?.steps || "24 (1-day Horizon)"}
            readOnly
            className="w-full p-2 bg-gray-50 border border-gray-300 rounded-md text-gray-700 focus:outline-none cursor-default"
          />
        </div>
      </div>
    </div>
  );

  const renderModelConfigurations = () => {
    const modelType = forecastData?.model?.toUpperCase() || "";

    if (modelType === "DHR-ESN" || modelType === "HYBRID") {
      return (
        <>
          <EsnConfiguration />
          <DhrConfiguration />
        </>
      );
    } else if (modelType === "ESN") {
      return <EsnConfiguration />;
    } else if (modelType === "DHR") {
      return <DhrConfiguration />;
    }

    return null;
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {error ? (
        <div className="p-6 bg-red-100 text-red-700 rounded">
          Error: {error}
        </div>
      ) : forecastId ? (
        <div>
          <div className="flex justify-between items-center mb-4">
            <h1 className="text-2xl font-bold">
              {historyLog?.file_name || "Loading..."}
            </h1>
            <button
              onClick={handleBack}
              className="px-3 py-1 bg-red-600 text-white rounded-md text-sm hover:bg-red-700">
              Back
            </button>
          </div>

          <GraphPlaceholder />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-4">
            <DatasetDetails />
          </div>

          {renderModelConfigurations()}
        </div>
      ) : (
        <div className="p-6 bg-red-100 text-red-700 rounded">
          No forecast ID provided. Please select a forecast from the history
          logs.
        </div>
      )}
    </div>
  );
};

export default ViewLogs;

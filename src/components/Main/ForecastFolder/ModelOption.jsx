import React, { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";

const ModelOption = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [filename, setFilename] = useState("No file selected");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Extract id from location state
  const fileId = location.state?.id;

  // If filename is available in location state, set it to state
  useEffect(() => {
    if (location.state?.filename) {
      setFilename(location.state.filename);
    }
    setLoading(false); // Once filename is set, stop loading
  }, [location.state?.filename]);

  return (
    <div className="p-6 bg-gray-100 min-h-screen flex flex-col items-center">
      <p>Selected File: {loading ? "Loading..." : filename}</p>
      {error && <p className="text-red-500">{error}</p>}
      <h1 className="text-2xl font-bold mb-6">Model Options</h1>

      <button
        onClick={() => navigate("/SolarForecastOptions")}
        className="mb-4 p-4 bg-blue-500 text-white rounded-md hover:bg-blue-600">
        ESN
      </button>

      <button
        onClick={() => navigate("/WindForecastOptions")}
        className="mb-4 p-4 bg-green-500 text-white rounded-md hover:bg-green-600">
        DHR
      </button>

      <button
        onClick={() => navigate("/WindForecastOptions")}
        className="mb-4 p-4 bg-green-500 text-white rounded-md hover:bg-green-600">
        ESN + DHR
      </button>

      <button
        onClick={() => navigate("/WindForecastOptions")}
        className="mb-4 p-4 bg-green-500 text-white rounded-md hover:bg-green-600">
        Sarima + LSTM
      </button>

      <button
        onClick={() => navigate("/WindForecastOptions")}
        className="mb-4 p-4 bg-green-500 text-white rounded-md hover:bg-green-600">
        Sarima + GRU
      </button>

      <button
        onClick={() => navigate("/WindForecastOptions")}
        className="mb-4 p-4 bg-green-500 text-white rounded-md hover:bg-green-600">
        Sarima + RNN
      </button>

      <button
        onClick={() => navigate("/WindForecastOptions")}
        className="mb-4 p-4 bg-green-500 text-white rounded-md hover:bg-green-600">
        Sarima + TRANSFORMER
      </button>
    </div>
  );
};

export default ModelOption;

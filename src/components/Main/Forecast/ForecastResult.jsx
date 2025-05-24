import { useLocation, useNavigate } from "react-router-dom";
import { useState, useEffect } from "react";
import Papa from "papaparse"; // Import PapaParse for CSV parsing

export default function ForecastResult() {
  const navigate = useNavigate();
  const { state } = useLocation();
  console.log("Received state:", state); // Debug log

  const imageUrl = state?.imageUrl;
  const [forecastData, setForecastData] = useState(state?.forecastData || {});
  const [config, setConfig] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [recommendations, setRecommendations] = useState({
    batteryStorage: false,
    bidStrategically: false,
  });
  const [fileContent, setFileContent] = useState("");
  const [parsedCsvData, setParsedCsvData] = useState([]); // Store forecast values only
  const [isZoomed, setIsZoomed] = useState(false);
  const [showModal, setShowModal] = useState(false);

  // Log fileContent, parsedCsvData, and calculate sum/mean
  useEffect(() => {
    if (fileContent) {
    }
    if (parsedCsvData.length > 0) {
      // Extract forecast values
      const forecastValues = parsedCsvData
        .map(
          (row) =>
            row.hybrid_forecast ??
            row.forecasted_solar_power ??
            row.forecasted_wind_power ??
            row.forecast ??
            null
        )
        .filter((value) => value !== null && !isNaN(value)); // Filter out null/invalid values

      // Calculate sum and mean
      const sum = forecastValues.reduce((acc, val) => acc + val, 0);
      const mean = forecastValues.length > 0 ? sum / forecastValues.length : 0;

      console.log("Mean of forecast values:", mean.toFixed(2)); // Round to 2 decimal places
    }
  }, [fileContent, parsedCsvData]);

  // Fetch forecast data including filename
  useEffect(() => {
    const fetchForecastData = async () => {
      if (!forecastData?.id) {
        setError("No forecast ID provided");
        setIsLoading(false);
        return;
      }

      try {
        setIsLoading(true);
        const forecastId = parseInt(forecastData.id, 10);

        if (isNaN(forecastId)) {
          throw new Error("Invalid forecast ID");
        }

        console.log("Fetching forecast with ID:", forecastId);

        const response = await fetch(
          `http://localhost:8000/api/forecasts/${forecastId}`,
          {
            headers: {
              Accept: "application/json",
            },
          }
        );

        if (!response.ok) {
          const errorText = await response.text();
          console.error("Response error:", response.status, errorText);
          throw new Error(`Error fetching forecast: ${response.statusText}`);
        }

        const data = await response.json();
        console.log("Received forecast data:", data);
        setForecastData((prev) => ({
          ...prev,
          ...data, // Merge fetched data, including filename
        }));
      } catch (err) {
        console.error("Error fetching forecast data:", err);
        setError(err.message);
      } finally {
        setIsLoading(false);
      }
    };

    fetchForecastData();
  }, [forecastData?.id]);

  // Fetch and parse CSV file content (exclude timestamp)
  useEffect(() => {
    const fetchFileContent = async () => {
      if (!forecastData?.filename) return;

      try {
        console.log("Fetching CSV file:", forecastData.filename);
        const response = await fetch(
          `http://localhost:8000/api/forecast-file/${forecastData.filename}`,
          {
            headers: {
              Accept: "text/csv",
            },
          }
        );
        if (!response.ok) {
          throw new Error(
            `Failed to fetch file content: ${response.statusText}`
          );
        }
        const text = await response.text();
        setFileContent(text);

        // Parse CSV data
        Papa.parse(text, {
          header: true,
          dynamicTyping: true,
          skipEmptyLines: true,
          complete: (results) => {
            console.log("Parsed CSV results:", results);

            // Check for specific forecast columns
            const firstRow = results.data[0] || {};
            const forecastColumn = [
              "hybrid_forecast",
              "forecasted_solar_power",
              "forecasted_wind_power",
            ].find((col) => col in firstRow);

            if (forecastColumn) {
              // Extract only the forecast column values
              const forecastData = results.data.map((row) => ({
                [forecastColumn]: row[forecastColumn] ?? null,
              }));
              setParsedCsvData(forecastData);
            } else {
              // Fallback to generic column names
              const forecastKey = Object.keys(firstRow).find(
                (key) =>
                  key.toLowerCase().includes("forecast") ||
                  key.toLowerCase().includes("value") ||
                  key.toLowerCase().includes("power")
              );

              if (forecastKey) {
                const forecastData = results.data.map((row) => ({
                  forecast: row[forecastKey] ?? null,
                }));
                setParsedCsvData(forecastData);
              } else {
                console.warn("No forecast column found in CSV");
                setParsedCsvData([]);
              }
            }
          },
          error: (error) => {
            console.error("CSV parsing error:", error);
            setError("Error parsing CSV data");
          },
        });
      } catch (err) {
        console.error("Error fetching file content:", err);
        setFileContent("Failed to load file content.");
        setError(err.message);
      }
    };

    fetchFileContent();
  }, [forecastData?.filename]);

  // Fetch configuration data
  useEffect(() => {
    const fetchConfiguration = async () => {
      if (!forecastData?.id || !forecastData?.model) {
        setIsLoading(false);
        return;
      }

      try {
        setIsLoading(true);
        setError(null);

        const modelKey =
          forecastData.model.toLowerCase() === "dhr-esn"
            ? "hybrid"
            : forecastData.model.toLowerCase();
        const endpoint = `http://localhost:8000/api/${modelKey}-configurations/${forecastData.id}`;
        console.log("Fetching from endpoint:", endpoint);

        const response = await fetch(endpoint);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log("Configuration data received:", data);
        setConfig(data);
      } catch (error) {
        console.error("Error fetching configuration:", error);
        setError(error.message);
      } finally {
        setIsLoading(false);
      }
    };

    fetchConfiguration();
  }, [forecastData?.id, forecastData?.model]);

  // Placeholder for handleViewGraphs
  const handleViewGraphs = () => {
    navigate("/view-graph", {
      state: {
        data: parsedCsvData, // Pass forecast values only
        forecastId: forecastData?.id,
      },
    });
  };

  // Placeholder for handleEdit
  const handleEdit = () => {
    console.log("handleEdit - forecastData:", forecastData);
    const queryParams = new URLSearchParams({
      model: forecastData.model || "",
      originalFileName:
        forecastData.filename || forecastData.original_filename || "",
      forecastType: forecastData.forecastType || "",
      steps: forecastData.steps || "",
      granularity: forecastData.granularity || "",
      forecastId: forecastData.id || "",
      tempFilename: forecastData.tempFilename || "",
    }).toString();

    const state = {
      forecastId: forecastData.id,
      model: forecastData.model,
      isEditing: true,
      existingConfig: config,
    };

    if (forecastData.model === "DHR-ESN") {
      navigate(`/configure-hybrid?${queryParams}`, { state });
    } else {
      navigate(`/configure-single?${queryParams}`, { state });
    }
  };

  // Get step label for forecast period
  const getStepLabel = (steps, granularity) => {
    if (!steps || !granularity) return "No steps available";

    const stepLabels = {
      Hourly: {
        1: "1 Step (1-Hour Horizon)",
        24: "24 Steps (1-Day Horizon)",
        168: "168 Steps (1-Week Horizon)",
      },
      Daily: {
        1: "1 Step (1-Day Horizon)",
        7: "7 Steps (1-Week Horizon)",
        30: "30 Steps (1-Month Horizon)",
      },
    };

    const stepsNum = Number(steps);
    return stepLabels[granularity]?.[stepsNum] || `${steps} Steps`;
  };

  // Render configuration section based on model type
  const renderConfigSection = () => {
    if (isLoading) {
      return <div className="text-gray-600">Loading configuration...</div>;
    }

    if (error) {
      return <div className="text-red-500">Error: {error}</div>;
    }

    if (!config) {
      return <div className="text-gray-600">No configuration available</div>;
    }

    if (forecastData.model === "DHR-ESN") {
      return (
        <div className="space-y-8">
          {/* DHR Section */}
          <div>
            <h3 className="text-lg font-semibold mb-4">
              Dynamic Harmonic Regression
            </h3>
            <div className="grid grid-cols-2 gap-x-7">
              <div>
                <p className="text-sm text-gray-600">Fourier Order</p>
                <input
                  type="text"
                  value={config?.fourier_terms || "-"}
                  disabled
                  className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold focus:outline-none cursor-default"
                />
              </div>
              <div>
                <p className="text-sm text-gray-600">Window Length</p>
                <input
                  type="text"
                  value={config?.window || "-"}
                  disabled
                  className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold focus:outline-none cursor-default"
                />
              </div>
              <div>
                <p className="text-sm text-gray-600">Polyorder</p>
                <input
                  type="text"
                  value={config?.polyorder || "-"}
                  disabled
                  className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold focus:outline-none cursor-default"
                />
              </div>
              <div>
                <p className="text-sm text-gray-600">Regularization (DHR)</p>
                <input
                  type="text"
                  value={config?.reg_strength || "-"}
                  disabled
                  className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold focus:outline-none cursor-default"
                />
              </div>
              <div>
                <p className="text-sm text-gray-600">AR Order</p>
                <input
                  type="text"
                  value={config?.ar_order || "-"}
                  disabled
                  className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
                />
              </div>
            </div>
          </div>

          {/* ESN Section */}
          <div>
            <h3 className="text-lg font-semibold mb-4">Echo State Networks</h3>
            <div className="grid grid-cols-2 gap-x-7">
              <div>
                <p className="text-sm text-gray-600">Reservoir Size</p>
                <input
                  type="text"
                  value={config?.N_res || "-"}
                  disabled
                  className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold focus:outline-none cursor-default"
                />
              </div>
              <div>
                <p className="text-sm text-gray-600">Spectral Radius</p>
                <input
                  type="text"
                  value={config?.rho || "-"}
                  disabled
                  className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold focus:outline-none cursor-default"
                />
              </div>
              <div>
                <p className="text-sm text-gray-600">Sparsity</p>
                <input
                  type="text"
                  value={config?.sparsity || "-"}
                  disabled
                  className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold focus:outline-none cursor-default"
                />
              </div>
              <div>
                <p className="text-sm text-gray-600">Input Scaling</p>
                <input
                  type="text"
                  value={config?.alpha || "-"}
                  disabled
                  className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold focus:outline-none cursor-default"
                />
              </div>
              <div>
                <p className="text-sm text-gray-600">Regularization</p>
                <input
                  type="text"
                  value={
                    config?.lambda_reg
                      ? Number(config.lambda_reg).toFixed(9)
                      : "-"
                  }
                  disabled
                  className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
                />
              </div>
              <div>
                <p className="text-sm text-gray-600">Lags</p>
                <input
                  type="text"
                  value={config?.lags || "-"}
                  disabled
                  className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
                />
              </div>
              <div>
                <p className="text-sm text-gray-600">N Features</p>
                <input
                  type="text"
                  value={config?.n_features || "-"}
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
          <div className="grid grid-cols-2 gap-x-7">
            <div>
              <p className="text-sm text-gray-600">Fourier Order</p>
              <input
                type="text"
                value={config?.fourier_terms || "-"}
                disabled
                className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
              />
            </div>
            <div>
              <p className="text-sm text-gray-600">Window Length</p>
              <input
                type="text"
                value={config?.window || "-"}
                disabled
                className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
              />
            </div>
            <div>
              <p className="text-sm text-gray-600">Polyorder</p>
              <input
                type="text"
                value={config?.polyorder || "-"}
                disabled
                className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
              />
            </div>
            <div>
              <p className="text-sm text-gray-600">Regularization</p>
              <input
                type="text"
                value={config?.reg_strength || "-"}
                disabled
                className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
              />
            </div>
            <div>
              <p className="text-sm text-gray-600">AR Order</p>
              <input
                type="text"
                value={config?.ar_order || "-"}
                disabled
                className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
              />
            </div>
          </div>
        </div>
      );
    } else if (forecastData.model === "ESN") {
      return (
        <div className="mb-8">
          <h3 className="text-lg font-semibold mb-4">Echo State Networks</h3>
          <div className="grid grid-cols-2 gap-x-7">
            <div>
              <p className="text-sm text-gray-600">Reservoir Size</p>
              <input
                type="text"
                value={config?.N_res || "-"}
                disabled
                className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
              />
            </div>
            <div>
              <p className="text-sm text-gray-600">Spectral Radius</p>
              <input
                type="text"
                value={config?.rho || "-"}
                disabled
                className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
              />
            </div>
            <div>
              <p className="text-sm text-gray-600">Sparsity</p>
              <input
                type="text"
                value={config?.sparsity || "-"}
                disabled
                className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
              />
            </div>
            <div>
              <p className="text-sm text-gray-600">Input Scaling</p>
              <input
                type="text"
                value={config?.alpha || "-"}
                disabled
                className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
              />
            </div>
            <div>
              <p className="text-sm text-gray-600">Regularization</p>
              <input
                type="text"
                value={
                  config?.lambda_reg
                    ? Number(config.lambda_reg).toFixed(9)
                    : "-"
                }
                disabled
                className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
              />
            </div>
            <div>
              <p className="text-sm text-gray-600">Lags</p>
              <input
                type="text"
                value={config?.lags || "-"}
                disabled
                className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
              />
            </div>
          </div>
        </div>
      );
    }
    return <div>No configuration available for this model.</div>;
  };

  return (
    <div className="max-w-5xl mx-auto">
      <div className="grid grid-cols-2 gap-6">
        {/* Left Column */}
        <div className="space-y-6">
          {/* Graph Card */}
          <div className="bg-white p-4 rounded-lg border-gray-500 shadow w-100">
            <h3 className="text-sm mb-2">
              {forecastData?.original_filename || "No filename available"}
            </h3>
            <h2 className="font-bold mb-4">
              {forecastData?.model || "No model selected"}
            </h2>
            <div className="h-48 rounded mb-4 overflow-hidden relative">
              {imageUrl ? (
                <>
                  <img
                    src={imageUrl}
                    alt="Forecast Plot"
                    className="w-full h-full object-contain rounded cursor-pointer"
                    onClick={() => setShowModal(true)}
                  />
                  {/* Modal Overlay */}
                  {showModal && (
                    <div
                      className="fixed inset-0 bg-black bg-opacity-75 z-50 flex items-center justify-center p-4"
                      onClick={() => setShowModal(false)}>
                      <div className="relative max-w-4xl max-h-[90vh]">
                        <button
                          className="absolute top-4 right-4 text-white hover:text-gray-300 text-xl"
                          onClick={() => setShowModal(false)}>
                          ×
                        </button>
                        <img
                          src={imageUrl}
                          alt="Forecast Plot"
                          className="max-w-full max-h-[85vh] object-contain rounded"
                        />
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div className="w-full h-full bg-gray-100 flex items-center justify-center">
                  <span className="text-gray-500">No graph available</span>
                </div>
              )}
            </div>
            {/* Button and Forecast Period in a flex container */}
            <div className="flex items-center justify-between">
              <button
                onClick={handleViewGraphs}
                className="py-2 px-4 bg-gray-200 text-black rounded-md hover:bg-gray-300 border-b-2 border-green-500 cursor-pointer">
                View Graphs
              </button>

              <div className="text-right mr-2">
                <h3 className="font-semibold mb-1">Generated Forecast</h3>
                <p className="text-sm font-semibold">
                  {getStepLabel(forecastData?.steps, forecastData?.granularity)}
                </p>
              </div>
            </div>
          </div>

          {/* Recommendations */}
          <div className="bg-white p-4 rounded-lg">
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
                  className="h-4 w-4 text-green-600 border-green-600 focus:ring-green-500 cursor-pointer rounded accent-green-700"
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
                  className="h-4 w-4 text-green-600 border-green-600 focus:ring-green-500 cursor-pointer rounded accent-green-700"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Right Column - Configurations */}
        <div className="">
          <h2 className="text-3xl font-bold mb-6">Configurations</h2>
          <div className="space-y-6">{renderConfigSection()}</div>
          <div className="flex space-x-4 mt-8 mb-4">
            <button
              onClick={handleEdit}
              className="bg-green-600 text-white px-6 py-2 rounded-md hover:bg-green-600 cursor-pointer">
              Edit
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

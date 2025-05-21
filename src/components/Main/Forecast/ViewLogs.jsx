import React, { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Line } from "react-chartjs-2";
import Papa from "papaparse";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  LineElement,
  PointElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  LineElement,
  PointElement,
  Title,
  Tooltip,
  Legend
);

const ViewLogs = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { forecastId } = location.state || {};
  const [forecastData, setForecastData] = useState(null);
  const [historyLog, setHistoryLog] = useState(null);
  const [esnConfig, setEsnConfig] = useState(null);
  const [dhrConfig, setDhrConfig] = useState(null);
  const [error, setError] = useState(null);
  const [chartData, setChartData] = useState(null);
  const [csvData, setCsvData] = useState(null);
  const [loading, setLoading] = useState(false);

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
      setLoading(true);
      const response = await fetch(
        `http://localhost:8000/api/forecasts/${forecastId}`
      );

      if (!response.ok) {
        throw new Error(`Error fetching forecast: ${response.statusText}`);
      }

      const data = await response.json();
      setForecastData(data);

      // If there's a filename, try to fetch CSV data
      if (data.filename) {
        await fetchCsvData(data.filename);
      } else {
        // Fallback to the old file content approach if no CSV filename
        if (data.filename) {
          await fetchFileContent(data.filename);
        }
      }
    } catch (error) {
      console.error("Error fetching forecast data:", error);
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchCsvData = async (filename) => {
    try {
      console.log("Fetching CSV file:", filename);
      const response = await fetch(
        `http://localhost:8000/api/forecast-file/${filename}`,
        {
          headers: {
            Accept: "text/csv",
          },
        }
      );

      if (!response.ok) {
        throw new Error(`Error fetching CSV: ${response.statusText}`);
      }

      const csvText = await response.text();
      console.log("CSV data fetched successfully");

      // Parse CSV data
      Papa.parse(csvText, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: (results) => {
          console.log("Parsed CSV data:", results);

          // Check for our specific forecast columns
          const firstRow = results.data[0] || {};
          const hasSpecificForecastColumns =
            "dhr_forecast" in firstRow ||
            "esn_forecast" in firstRow ||
            "hybrid_forecast" in firstRow;

          if (hasSpecificForecastColumns) {
            // Keep the data as is since it already has our expected columns
            setCsvData(results.data);

            // Generate chart data
            createChartFromCsvData(results.data);
          } else {
            // Handle possible column naming variations for generic forecast data
            const processedData = results.data.map((row) => {
              // Look for timestamp column (might be named timestamp, date, time, etc.)
              const timestampKey = Object.keys(row).find(
                (key) =>
                  key.toLowerCase().includes("time") ||
                  key.toLowerCase().includes("date")
              );

              // Look for forecast/value column
              const forecastKey = Object.keys(row).find(
                (key) =>
                  key.toLowerCase().includes("forecast") ||
                  key.toLowerCase().includes("value") ||
                  key.toLowerCase().includes("power")
              );

              return {
                timestamp: row[timestampKey] || "Unknown",
                forecast: row[forecastKey] || 0,
              };
            });

            setCsvData(processedData);

            // Generate simple chart data
            createSimpleChartFromCsvData(processedData);
          }
        },
        error: (error) => {
          console.error("CSV parsing error:", error);
          setError("Error parsing CSV data");
        },
      });
    } catch (err) {
      console.error("Error fetching CSV data:", err);
      setError(err.message);
    }
  };

  const createChartFromCsvData = (data) => {
    if (!data || data.length === 0) return;

    // Define color schemes
    const colorSchemes = [
      {
        borderColor: "rgba(75, 192, 192, 1)",
        backgroundColor: "rgba(75, 192, 192, 0.2)",
      }, // Teal
      {
        borderColor: "rgba(153, 102, 255, 1)",
        backgroundColor: "rgba(153, 102, 255, 0.2)",
      }, // Purple
      {
        borderColor: "rgba(255, 159, 64, 1)",
        backgroundColor: "rgba(255, 159, 64, 0.2)",
      }, // Orange
    ];

    const datasets = [];

    // Add datasets for each forecast type
    if ("dhr_forecast" in data[0]) {
      datasets.push({
        label: "DHR Forecast",
        data: data.map((item) => item.dhr_forecast),
        borderColor: colorSchemes[0].borderColor,
        backgroundColor: colorSchemes[0].backgroundColor,
        tension: 0.1,
        fill: false,
      });
    }

    if ("esn_forecast" in data[0]) {
      datasets.push({
        label: "ESN Forecast",
        data: data.map((item) => item.esn_forecast),
        borderColor: colorSchemes[1].borderColor,
        backgroundColor: colorSchemes[1].backgroundColor,
        tension: 0.1,
        fill: false,
      });
    }

    if ("hybrid_forecast" in data[0]) {
      datasets.push({
        label: "Hybrid Forecast",
        data: data.map((item) => item.hybrid_forecast),
        borderColor: colorSchemes[2].borderColor,
        backgroundColor: colorSchemes[2].backgroundColor,
        tension: 0.1,
        fill: false,
      });
    }

    setChartData({
      labels: data.map((item) => item.timestamp),
      datasets: datasets,
    });
  };

  const createSimpleChartFromCsvData = (data) => {
    if (!data || data.length === 0) return;

    setChartData({
      labels: data.map((item) => item.timestamp),
      datasets: [
        {
          label: "Forecast",
          data: data.map((item) => item.forecast),
          borderColor: "rgba(75, 192, 192, 1)",
          backgroundColor: "rgba(75, 192, 192, 0.2)",
          fill: false,
        },
      ],
    });
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
      // Assuming jsonContent contains the data for the chart
      // Example format: { timestamps: [...], values: [...] }
      if (jsonContent && jsonContent.timestamps && jsonContent.values) {
        setChartData({
          labels: jsonContent.timestamps,
          datasets: [
            {
              label: "Generated Power",
              data: jsonContent.values,
              borderColor: "rgba(75, 192, 192, 1)",
              backgroundColor: "rgba(75, 192, 192, 0.2)",
              fill: false,
            },
          ],
        });
      }
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
      setForecastData(forecastData);

      // Normalize the model type for consistent handling
      const modelType = (forecastData.model || "").toUpperCase();
      console.log("Model type:", modelType);

      // Handle different model types and their configuration endpoints
      if (modelType === "DHR") {
        const dhrResponse = await fetch(
          `http://localhost:8000/api/dhr-configurations/${forecastId}`
        );
        if (dhrResponse.ok) {
          const dhrData = await dhrResponse.json();
          setDhrConfig(dhrData);
          console.log("DHR config fetched:", dhrData);
        }
      } else if (modelType === "ESN") {
        const esnResponse = await fetch(
          `http://localhost:8000/api/esn-configurations/${forecastId}`
        );
        if (esnResponse.ok) {
          const esnData = await esnResponse.json();
          setEsnConfig(esnData);
          console.log("ESN config fetched:", esnData);
        }
      } else if (modelType === "HYBRID" || modelType === "DHR-ESN") {
        // For hybrid models, fetch both configurations

        // First, fetch hybrid config which might contain both configurations
        const hybridResponse = await fetch(
          `http://localhost:8000/api/hybrid-configurations/${forecastId}`
        );

        if (hybridResponse.ok) {
          const hybridData = await hybridResponse.json();
          console.log("Hybrid config fetched:", hybridData);

          // Check if the response has nested configs
          if (hybridData.dhr_config) {
            setDhrConfig(hybridData.dhr_config);
          } else if (hybridData.dhr) {
            setDhrConfig(hybridData.dhr);
          }

          if (hybridData.esn_config) {
            setEsnConfig(hybridData.esn_config);
          } else if (hybridData.esn) {
            setEsnConfig(hybridData.esn);
          }

          // If the hybrid data doesn't have nested configs, it might be a flat structure
          if (
            !hybridData.dhr_config &&
            !hybridData.esn_config &&
            !hybridData.dhr &&
            !hybridData.esn
          ) {
            // Try to determine which fields belong to which model
            const possibleDhrFields = [
              "fourier_terms",
              "window",
              "polyorder",
              "reg_strength",
              "ar_order",
            ];
            const possibleEsnFields = [
              "N_res",
              "rho",
              "sparsity",
              "alpha",
              "lambda_reg",
              "lags",
              "n_features",
            ];

            // Create config objects from the flat structure
            const extractedDhrConfig = {};
            const extractedEsnConfig = {};

            // Extract DHR fields
            possibleDhrFields.forEach((field) => {
              if (field in hybridData) {
                extractedDhrConfig[field] = hybridData[field];
              }
            });

            // Extract ESN fields
            possibleEsnFields.forEach((field) => {
              if (field in hybridData) {
                extractedEsnConfig[field] = hybridData[field];
              }
            });

            // Set the configs if any fields were found
            if (Object.keys(extractedDhrConfig).length > 0) {
              setDhrConfig(extractedDhrConfig);
            }

            if (Object.keys(extractedEsnConfig).length > 0) {
              setEsnConfig(extractedEsnConfig);
            }
          }
        }

        // If hybrid config doesn't have what we need, try individual endpoints as fallback
        if (!dhrConfig) {
          try {
            const dhrResponse = await fetch(
              `http://localhost:8000/api/dhr-configurations/${forecastId}`
            );
            if (dhrResponse.ok) {
              const dhrData = await dhrResponse.json();
              setDhrConfig(dhrData);
              console.log("DHR config fetched separately:", dhrData);
            }
          } catch (err) {
            console.warn("Could not fetch DHR config separately:", err);
          }
        }

        if (!esnConfig) {
          try {
            const esnResponse = await fetch(
              `http://localhost:8000/api/esn-configurations/${forecastId}`
            );
            if (esnResponse.ok) {
              const esnData = await esnResponse.json();
              setEsnConfig(esnData);
              console.log("ESN config fetched separately:", esnData);
            }
          } catch (err) {
            console.warn("Could not fetch ESN config separately:", err);
          }
        }
      }
    } catch (error) {
      console.error("Error fetching model configurations:", error);
      setError("Failed to fetch model configurations: " + error.message);
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

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "top",
      },
      title: {
        display: true,
        text: "Power Generation Forecast",
      },
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: "Time",
        },
      },
      y: {
        display: true,
        title: {
          display: true,
          text: "Power (kW)",
        },
      },
    },
  };

  // Check if CSV data has multiple forecasts
  const hasMultipleForecasts =
    csvData &&
    csvData.length > 0 &&
    ("dhr_forecast" in csvData[0] ||
      "esn_forecast" in csvData[0] ||
      "hybrid_forecast" in csvData[0]);

  const GraphPlaceholder = () => (
    <div className="mb-6">
      <h2 className="text-lg font-semibold mb-2">Generated Power</h2>
      {loading ? (
        <div className="h-48 flex items-center justify-center border border-gray-300 rounded">
          <p className="text-gray-500">Loading data...</p>
        </div>
      ) : chartData ? (
        <>
          <div className="h-[500px]">
            <Line data={chartData} options={options} />
          </div>
          {hasMultipleForecasts && (
            <div className="mt-2 p-2 bg-gray-50 rounded text-sm text-gray-600">
              Showing multiple forecast models for comparison: DHR, ESN, and
              Hybrid forecasts.
            </div>
          )}
        </>
      ) : (
        <div className="h-48 flex items-center justify-center border border-gray-300 rounded">
          <div className="text-gray-500">
            <p className="text-xl font-medium">No data available</p>
          </div>
        </div>
      )}
    </div>
  );

  const DatasetDetails = () => (
    <div className="mb-6">
      <h2 className="text-lg font-bold mb-2">Dataset Details</h2>
      <div className="space-y-2">
        <div>
          <label className="block font-medium">Dataset</label>
          <input
            type="text"
            value={forecastData?.original_filename || ""}
            readOnly
            className="w-1/4 border border-gray-300 rounded p-2"
          />
        </div>
        <div>
          <label className="block font-medium">Forecast File</label>
          <input
            type="text"
            value={forecastData?.filename || ""}
            readOnly
            className="w-1/4 border border-gray-300 rounded p-2"
          />
        </div>
        <div>
          <label className="block font-medium">Granularity</label>
          <input
            type="text"
            value={forecastData?.granularity || ""}
            readOnly
            className="w-1/4 border border-gray-300 rounded p-2"
          />
        </div>
        <div>
          <label className="block font-medium">Steps</label>
          <input
            type="text"
            value={forecastData?.steps || ""}
            readOnly
            className="w-1/4 border border-gray-300 rounded p-2"
          />
        </div>
      </div>
    </div>
  );

  const EsnConfiguration = () => {
    // Debug logging to help troubleshoot
    console.log("Rendering ESN Configuration with data:", esnConfig);

    return (
      <div>
        <h2 className="text-lg font-bold mb-2">
          Configurations - Echo State Networks
        </h2>
        {esnConfig ? (
          <div className="grid grid-cols-6 gap-x-10">
            <div>
              <label className="block font-medium">Reservoir Size</label>
              <input
                type="text"
                value={esnConfig?.N_res || ""}
                readOnly
                className="w-full border border-gray-300 rounded p-2"
              />
            </div>
            <div>
              <label className="block font-medium">Spectral Radius</label>
              <input
                type="text"
                value={esnConfig?.rho || ""}
                readOnly
                className="w-full border border-gray-300 rounded p-2"
              />
            </div>
            <div>
              <label className="block font-medium">Sparsity</label>
              <input
                type="text"
                value={esnConfig?.sparsity || ""}
                readOnly
                className="w-full border border-gray-300 rounded p-2"
              />
            </div>
            <div>
              <label className="block font-medium">Input Scaling</label>
              <input
                type="text"
                value={esnConfig?.alpha || ""}
                readOnly
                className="w-full border border-gray-300 rounded p-2"
              />
            </div>
            <div>
              <label className="block font-medium">Regularization</label>
              <input
                type="text"
                value={esnConfig?.lambda_reg || ""}
                readOnly
                className="w-full border border-gray-300 rounded p-2"
              />
            </div>
            <div>
              <label className="block font-medium">Lags</label>
              <input
                type="text"
                value={esnConfig?.lags || ""}
                readOnly
                className="w-full border border-gray-300 rounded p-2"
              />
            </div>
            {/* Add n_features field that appears in hybrid mode */}
            {esnConfig?.n_features !== undefined && (
              <div className="mt-5">
                <label className="block font-medium">N Features</label>
                <input
                  type="text"
                  value={esnConfig?.n_features || ""}
                  readOnly
                  className="w-full border border-gray-300 rounded p-2"
                />
              </div>
            )}
          </div>
        ) : (
          <div className="p-4 bg-yellow-100 text-yellow-700 rounded">
            No ESN configuration data available.
          </div>
        )}
      </div>
    );
  };

  const DhrConfiguration = () => {
    // Debug logging to help troubleshoot
    console.log("Rendering DHR Configuration with data:", dhrConfig);

    return (
      <div className="mt-6">
        <h2 className="text-lg font-bold mb-2">
          Configurations - Dynamic Harmonic Regression
        </h2>
        {dhrConfig ? (
          <div className="grid grid-cols-6 gap-x-10">
            <div>
              <label className="block font-medium">Fourier Order</label>
              <input
                type="text"
                value={dhrConfig?.fourier_terms || ""}
                readOnly
                className="w-full border border-gray-300 rounded p-2"
              />
            </div>
            <div>
              <label className="block font-medium">Window Length</label>
              <input
                type="text"
                value={dhrConfig?.window || ""}
                readOnly
                className="w-full border border-gray-300 rounded p-2"
              />
            </div>
            <div>
              <label className="block font-medium">Polyorder</label>
              <input
                type="text"
                value={dhrConfig?.polyorder || ""}
                readOnly
                className="w-full border border-gray-300 rounded p-2"
              />
            </div>
            <div>
              <label className="block font-medium">Regularization</label>
              <input
                type="text"
                value={dhrConfig?.reg_strength || ""}
                readOnly
                className="w-full border border-gray-300 rounded p-2"
              />
            </div>
            <div>
              <label className="block font-medium">AR Order</label>
              <input
                type="text"
                value={dhrConfig?.ar_order || ""}
                readOnly
                className="w-full border border-gray-300 rounded p-2"
              />
            </div>
          </div>
        ) : (
          <div className="p-4 bg-yellow-100 text-yellow-700 rounded">
            No DHR configuration data available.
          </div>
        )}
      </div>
    );
  };

  const renderModelConfigurations = () => {
    const modelType = (forecastData?.model || "").toUpperCase();
    console.log("Rendering model configurations for type:", modelType);
    console.log("DHR Config:", dhrConfig);
    console.log("ESN Config:", esnConfig);

    if (modelType === "HYBRID" || modelType === "DHR-ESN") {
      return (
        <>
          <DhrConfiguration />
          <EsnConfiguration />
        </>
      );
    } else if (modelType === "DHR") {
      return <DhrConfiguration />;
    } else if (modelType === "ESN") {
      return <EsnConfiguration />;
    }
    return (
      <div className="p-4 bg-gray-100 text-gray-700 rounded">
        No model configuration available for model type:{" "}
        {modelType || "Unknown"}
      </div>
    );
  };

  return (
    <div className="p-6">
      <div className="">
        {/* Header */}
        <div className="flex justify-between items-center mb-4">
          <h1 className="text-xl font-bold">
            {historyLog?.file_name ||
              forecastData?.original_filename ||
              "Forecast Details"}
          </h1>
          <button
            onClick={handleBack}
            className="bg-red-500 text-white px-3 py-1 rounded hover:bg-red-600">
            Back
          </button>
        </div>

        {error && <p className="text-red-600 mb-4">{error}</p>}

        {forecastId ? (
          <>
            <GraphPlaceholder />
            <DatasetDetails />
            {renderModelConfigurations()}
          </>
        ) : (
          <div className="p-4 bg-red-100 text-red-700 rounded">
            No forecast ID provided. Please select a forecast from the history
            logs.
          </div>
        )}
      </div>
    </div>
  );
};

export default ViewLogs;

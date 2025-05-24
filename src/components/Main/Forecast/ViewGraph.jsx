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

export default function ViewGraph() {
  const { state } = useLocation();
  const navigate = useNavigate();
  const [forecastData, setForecastData] = useState(null);
  const [csvData, setCsvData] = useState(null);
  const [chartData, setChartData] = useState({ labels: [], datasets: [] }); // New state for chart data
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Fetch forecast data on mount
  useEffect(() => {
    const fetchForecastData = async () => {
      try {
        // Check if state exists
        if (!state || !state.forecastId) {
          throw new Error("No forecast ID provided");
        }

        // Ensure forecastId is a number
        const forecastId = parseInt(state.forecastId, 10);

        if (isNaN(forecastId)) {
          throw new Error("Invalid forecast ID");
        }

        console.log("Fetching forecast with ID:", forecastId);

        setLoading(true);
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
        setForecastData(data);

        // After getting forecast data, fetch the CSV file if a filename exists
        if (data.filename) {
          await fetchCsvData(data.filename);
        }
      } catch (err) {
        console.error("Error fetching forecast data:", err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchForecastData();
  }, [state, navigate]);

  // Function to fetch CSV data
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
      console.log("Raw CSV content:", csvText);
      if (!csvText.trim()) {
        throw new Error("CSV file is empty");
      }

      // Parse CSV data
      Papa.parse(csvText, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: (results) => {
          console.log("Parsed CSV data:", results.data);
          console.log("Parsed CSV errors:", results.errors);
          if (results.data.length === 0) {
            setError("No data found in CSV file");
            return;
          }

          // Check for our specific forecast columns
          const firstRow = results.data[0] || {};
          console.log("First row of CSV:", firstRow);
          const hasSpecificForecastColumns =
            "dhr_forecast" in firstRow ||
            "esn_forecast" in firstRow ||
            "hybrid_forecast" in firstRow;

          if (hasSpecificForecastColumns) {
            setCsvData(results.data);
          } else {
            // Handle possible column naming variations for generic forecast data
            const timestampKey = Object.keys(firstRow).find((key) =>
              ["time", "date", "timestamp", "datetime"].some((term) =>
                key.toLowerCase().includes(term)
              )
            );

            const forecastKey = Object.keys(firstRow).find((key) =>
              ["forecast", "value", "power", "prediction", "hybrid"].some(
                (term) => key.toLowerCase().includes(term)
              )
            );

            if (!timestampKey || !forecastKey) {
              setError(
                "CSV file missing required timestamp or forecast columns"
              );
              return;
            }

            const processedData = results.data.map((row) => ({
              timestamp: row[timestampKey] || "Unknown",
              forecast: row[forecastKey] || 0,
            }));

            setCsvData(processedData);
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

  // Process chart data when csvData or state.data changes
  useEffect(() => {
    // Reset chart data and error
    setChartData({ labels: [], datasets: [] });
    setError(null);

    // Determine the date column key (datetime or timestamp)
    const dateKey =
      csvData && csvData.length > 0
        ? "datetime" in csvData[0]
          ? "datetime"
          : "timestamp" in csvData[0]
          ? "timestamp"
          : null
        : state?.data && state.data.length > 0
        ? "timestamp" in state.data[0]
          ? "timestamp"
          : "datetime" in state.data[0]
          ? "datetime"
          : null
        : null;

    if (!dateKey && csvData?.length > 0) {
      setError("CSV file missing required 'datetime' or 'timestamp' column");
      return;
    }

    const newChartData = { labels: [], datasets: [] };

    // Define color schemes for different forecast types
    const colorSchemes = [
      {
        borderColor: "rgba(75, 192, 192, 1)",
        backgroundColor: "rgba(75, 192, 192, 0.2)",
      },
      {
        borderColor: "rgba(153, 102, 255, 1)",
        backgroundColor: "rgba(153, 102, 255, 0.2)",
      },
      {
        borderColor: "rgba(255, 159, 64, 1)",
        backgroundColor: "rgba(255, 159, 64, 0.2)",
      },
    ];

    // If we have csvData
    if (csvData && csvData.length > 0 && dateKey) {
      // Filter rows to ensure they have a valid date and at least one forecast value
      const validData = csvData.filter((item) => {
        const hasDate =
          item[dateKey] !== undefined &&
          item[dateKey] !== null &&
          item[dateKey] !== "";
        const hasAtLeastOneForecast =
          (item.dhr_forecast !== undefined && !isNaN(item.dhr_forecast)) ||
          (item.esn_forecast !== undefined && !isNaN(item.esn_forecast)) ||
          (item.hybrid_forecast !== undefined &&
            !isNaN(item.hybrid_forecast)) ||
          (item.forecast !== undefined && !isNaN(item.forecast));
        return hasDate && hasAtLeastOneForecast;
      });

      if (validData.length === 0) {
        setError("No valid data with dates and forecast values found in CSV");
        return;
      }

      newChartData.labels = validData.map((item) => item[dateKey]);

      // Check for hybrid forecast columns
      const hasHybridForecasts =
        validData[0].hasOwnProperty("dhr_forecast") ||
        validData[0].hasOwnProperty("esn_forecast") ||
        validData[0].hasOwnProperty("hybrid_forecast");

      if (hasHybridForecasts) {
        // Add DHR forecast if it exists
        if (validData[0].hasOwnProperty("dhr_forecast")) {
          newChartData.datasets.push({
            label: "DHR Forecast",
            data: validData.map((item) =>
              isNaN(item.dhr_forecast) ? null : item.dhr_forecast
            ),
            borderColor: colorSchemes[0].borderColor,
            backgroundColor: colorSchemes[0].backgroundColor,
            tension: 0.1,
          });
        }

        // Add ESN forecast if it exists
        if (validData[0].hasOwnProperty("esn_forecast")) {
          newChartData.datasets.push({
            label: "ESN Forecast",
            data: validData.map((item) =>
              isNaN(item.esn_forecast) ? null : item.esn_forecast
            ),
            borderColor: colorSchemes[1].borderColor,
            backgroundColor: colorSchemes[1].backgroundColor,
            tension: 0.1,
          });
        }

        // Add Hybrid forecast if it exists
        if (validData[0].hasOwnProperty("hybrid_forecast")) {
          newChartData.datasets.push({
            label: "Hybrid Forecast",
            data: validData.map((item) =>
              isNaN(item.hybrid_forecast) ? null : item.hybrid_forecast
            ),
            borderColor: colorSchemes[2].borderColor,
            backgroundColor: colorSchemes[2].backgroundColor,
            tension: 0.1,
          });
        }
      } else {
        // Fallback to generic forecast if no specific columns
        newChartData.datasets.push({
          label: "Forecast",
          data: validData.map((item) =>
            isNaN(item.forecast) ? null : item.forecast
          ),
          borderColor: colorSchemes[0].borderColor,
          backgroundColor: colorSchemes[0].backgroundColor,
          tension: 0.1,
        });
      }
    } else if (state?.data && state.data.length > 0 && dateKey) {
      // Fallback to state data
      const validStateData = state.data.filter((item) => {
        const hasDate =
          item[dateKey] !== undefined &&
          item[dateKey] !== null &&
          item[dateKey] !== "";
        const hasForecast =
          item.forecast !== undefined && !isNaN(item.forecast);
        return hasDate && hasForecast;
      });

      if (validStateData.length === 0) {
        setError(
          "No valid data with dates and forecast values found in state data"
        );
        return;
      }

      newChartData.labels = validStateData.map((item) => item[dateKey]);
      newChartData.datasets.push({
        label: "Forecast",
        data: validStateData.map((item) =>
          isNaN(item.forecast) ? null : item.forecast
        ),
        borderColor: colorSchemes[0].borderColor,
        backgroundColor: colorSchemes[0].backgroundColor,
        tension: 0.1,
      });
    }

    console.log("Setting chart data:", newChartData);
    setChartData(newChartData);
  }, [csvData, state?.data]); // Dependencies: re-run when csvData or state.data changes

  const handleBack = () => navigate(-1);
  const handlePrint = () => window.print();

  // Add print stylesheet
  useEffect(() => {
    const style = document.createElement("style");
    style.id = "print-style";
    style.innerHTML = `
      @media print {
        body * {
          visibility: hidden;
        }
        .print-container, .print-container * {
          visibility: visible;
        }
        .no-print {
          display: none !important;
        }
        .print-container {
          position: absolute;
          left: 0;
          top: 0;
          width: 100%;
        }
      }
    `;
    document.head.appendChild(style);

    return () => {
      const styleElement = document.getElementById("print-style");
      if (styleElement) {
        document.head.removeChild(styleElement);
      }
    };
  }, []);

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      title: {
        display: true,
        text: "Forecast Over Time",
        font: {
          size: 16,
        },
      },
      legend: {
        position: "top",
      },
    },
    scales: {
      x: {
        title: { display: true, text: "Timestamp" },
        ticks: { maxRotation: 45 },
      },
      y: { title: { display: true, text: "Forecast" } },
    },
  };

  // Determine if we have actual data to display
  const hasData =
    (csvData && csvData.length > 0) || (state?.data && state.data.length > 0);

  // Check if we have multiple forecasts (hybrid case)
  const hasMultipleForecasts =
    csvData &&
    csvData.length > 0 &&
    ("dhr_forecast" in csvData[0] ||
      "esn_forecast" in csvData[0] ||
      "hybrid_forecast" in csvData[0]);

  // For single step forecasts
  const isSingleStep =
    forecastData?.steps === 1 &&
    !hasMultipleForecasts &&
    ((csvData && csvData.length === 1) ||
      (state?.data && state.data.length === 1));
  const singleStepValue =
    csvData?.[0]?.forecast || state?.data?.[0]?.forecast || "N/A";

  return (
    <div className="p-6 max-w-7xl mx-auto print-container">
      {loading ? (
        <div className="flex items-center justify-center h-screen">
          <p>Loading...</p>
        </div>
      ) : error ? (
        <div className="flex items-center justify-center h-screen">
          <p className="text-red-500">{error}</p>
        </div>
      ) : (
        <>
          <div className="flex justify-end mb-6 no-print">
            <div className="space-x-8">
              <button
                onClick={handleBack}
                className="px-4 py-2 bg-red-500 text-white rounded-xl hover:bg-red-600 cursor-pointer">
                Back
              </button>
              <button
                onClick={handlePrint}
                className="px-4 py-2 bg-green-600 text-white rounded-xl hover:bg-green-700 cursor-pointer">
                Print
              </button>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-white p-4 rounded-lg border-gray-500 shadow">
              <h3 className="text-lg font-medium mb-2">Generated Power</h3>
              {isSingleStep ? (
                <div className="h-[500px] bg-gray-50 flex items-center justify-center">
                  <div className="text-gray-700 text-xl font-semibold">
                    1-Step Forecast: {singleStepValue}
                  </div>
                </div>
              ) : hasMultipleForecasts ? (
                <div className="h-[500px]">
                  <Line data={chartData} options={options} />
                </div>
              ) : hasData ? (
                <div className="h-[500px]">
                  <Line data={chartData} options={options} />
                </div>
              ) : (
                <div className="h-[500px] bg-gray-50 flex items-center justify-center">
                  <div className="text-gray-500">
                    <p className="text-xl font-medium">No data available</p>
                  </div>
                </div>
              )}
            </div>

            <div className="p-4 bg-white rounded-lg border-gray-500 shadow">
              <h3 className="text-2xl font-bold mb-6">Dataset Details</h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Dataset
                  </label>
                  <input
                    type="text"
                    value={forecastData?.original_filename || "N/A"}
                    readOnly
                    className="w-80 p-2 bg-gray-50 border border-gray-300 rounded-md text-gray-700 focus:outline-none cursor-default"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Forecast File
                  </label>
                  <input
                    type="text"
                    value={forecastData?.filename || "N/A"}
                    readOnly
                    className="w-80 p-2 bg-gray-50 border border-gray-300 rounded-md text-gray-700 focus:outline-none cursor-default"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Granularity
                  </label>
                  <input
                    type="text"
                    value={forecastData?.granularity || "N/A"}
                    readOnly
                    className="w-80 p-2 bg-gray-50 border border-gray-300 rounded-md text-gray-700 focus:outline-none cursor-default"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Steps
                  </label>
                  <input
                    type="text"
                    value={forecastData?.steps || "N/A"}
                    readOnly
                    className="w-80 p-2 bg-gray-50 border border-gray-300 rounded-md text-gray-700 focus:outline-none cursor-default"
                  />
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

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
  const [recommendation, setRecommendation] = useState({
    title: "",
    text: "",
  });
  const [energyDemand, setEnergyDemand] = useState(null);
  const [maxCapacity, setMaxCapacity] = useState(null);

  useEffect(() => {
    if (forecastId) {
      fetchForecastData();
      fetchHistoryLog();
    } else {
      setError("No forecast ID provided");
      console.error("No forecast ID provided");
    }
  }, [forecastId]);

  useEffect(() => {
    if (forecastData) {
      fetchModelConfigurations();
    }
  }, [forecastData]);

  // Recommendation mapping based on granularity, steps, and title
  const getRecommendationActions = (title, granularity, steps) => {
    console.log("getRecommendationActions inputs:", {
      title,
      granularity,
      steps,
    });
    const actions = {
      Overgenerate: {
        Hourly: {
          1: [
            "Curtail excess solar/wind immediately",
            "Store surplus if possible",
          ],
          24: ["Pre-charge batteries", "Prepare for day-ahead overproduction"],
          168: ["Adjust weekly procurement", "Prepare long-term export plans"],
        },
        Daily: {
          1: ["Reduce planned imports", "Hold reserve from thermal units"],
          7: ["Optimize storage rotation", "Consider selling excess energy"],
          30: [
            "Plan investment in grid-scale storage or export infrastructure",
          ],
        },
      },
      Undergenerate: {
        Hourly: {
          1: ["Dispatch spinning reserves", "Reduce non-critical loads"],
          24: ["Alert load balancing systems", "Activate short-term contracts"],
          168: ["Schedule maintenance deferment", "Boost flexible supply"],
        },
        Daily: {
          1: ["Supplement with grid purchases", "Activate demand response"],
          7: ["Prepare contingency reserves", "Adjust weekly load plans"],
          30: [
            "Revise energy supply strategy",
            "Strengthen backup procurement",
          ],
        },
      },
      Balance: {
        Hourly: {
          1: ["Maintain real-time operations", "Continue monitoring"],
          24: ["Keep current schedule", "Adjust for minor variability"],
          168: ["Confirm forecast reliability", "Check for anomalies"],
        },
        Daily: {
          1: ["Continue scheduled dispatch", "Monitor short-term shifts"],
          7: ["Execute regular weekly plans", "Optimize distribution loads"],
          30: ["Maintain baseline strategy", "Review long-term forecasts"],
        },
      },
    };

    // Convert steps to a number for proper lookup
    const stepsNum = Number(steps);
    if (isNaN(stepsNum)) {
      console.warn("Steps is not a valid number, using default:", steps);
      return ["No valid steps provided", "No valid steps provided"];
    }

    // Check if inputs are valid
    if (!title || !granularity || !actions[title]?.[granularity]?.[stepsNum]) {
      console.warn("Invalid inputs for recommendation, using default:", {
        title,
        granularity,
        stepsNum,
      });
      return [
        "Invalid recommendation inputs. Please check the forecast data.",
        "Ensure title, granularity, and steps are correctly defined.",
      ];
    }

    return actions[title][granularity][stepsNum];
  };

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
      console.log("API Response:", data);
      setForecastData(data);

      const energyDemand = data.energyDemand || null;
      const maxCapacity = data.maxCapacity || null;
      setEnergyDemand(energyDemand);
      setMaxCapacity(maxCapacity);
      console.log("Energy Demand:", energyDemand);
      console.log("Max Capacity:", maxCapacity);

      let csvDataResult = null;
      if (data.filename) {
        if (data.filename.endsWith(".csv")) {
          csvDataResult = await fetchCsvData(data.filename);
        } else {
          await fetchFileContent(data.filename);
        }
      } else {
        setError("No filename provided in forecast data");
      }

      if (
        csvDataResult &&
        csvDataResult.length > 0 &&
        energyDemand !== null &&
        maxCapacity !== null &&
        maxCapacity !== 0
      ) {
        const forecastValues = csvDataResult
          .map(
            (row) =>
              row.hybrid_forecast ??
              row.forecasted_solar_power ??
              row.forecasted_wind_power ??
              row.forecast ??
              null
          )
          .filter((value) => value !== null && !isNaN(value));

        if (!forecastValues.length) {
          setRecommendation({
            title: "Invalid Data",
            text: "No valid forecast data available.",
          });
          return;
        }

        const sum = forecastValues.reduce((acc, val) => acc + val, 0);
        const mean =
          forecastValues.length > 0 ? sum / forecastValues.length : 0;

        if (mean < 0 || isNaN(mean)) {
          setRecommendation({
            title: "Invalid Forecast",
            text: "Forecast contains negative or invalid values, which are not suitable for power generation.",
          });
          return;
        }

        if (energyDemand <= 0 || maxCapacity <= 0) {
          setRecommendation({
            title: "Invalid Input",
            text: "Energy demand or max capacity must be positive.",
          });
          return;
        }

        const normDemand = energyDemand / maxCapacity;
        const lowerBound = normDemand * 0.9; // 90% of demand
        const upperBound = normDemand * 1.1; // 110% of demand

        console.log(
          "ForecastValues:",
          forecastValues,
          "Mean:",
          mean,
          "EnergyDemand:",
          energyDemand,
          "MaxCapacity:",
          maxCapacity,
          "NormDemand:",
          normDemand,
          "LowerBound:",
          lowerBound,
          "UpperBound:",
          upperBound,
          "In Balance Range:",
          lowerBound <= mean && mean <= upperBound,
          "Mean > UpperBound:",
          mean > upperBound,
          "Mean < LowerBound:",
          mean < lowerBound
        );

        const epsilon = 1e-10; // Handle floating-point precision
        if (mean > upperBound + epsilon) {
          setRecommendation({
            title: "Overgenerate",
            text: "Forecast analysis shows that generation is likely to exceed demand by more than 10%. Please begin charging battery energy storage systems, consider exporting excess power to the external grid if available, and initiate curtailment of solar or wind units to prevent grid overvoltage. You may also notify large consumers to increase their load through demand response programs.",
          });
        } else if (mean < lowerBound - epsilon) {
          setRecommendation({
            title: "Undergenerate",
            text: "The system anticipates a generation shortfall of over 10% compared to demand. Please dispatch backup generation units immediately, initiate energy imports if grid interconnection is available, and issue a demand response call to reduce load in non-critical sectors. Pre-charge energy storage systems during off-peak hours if time permits.",
          });
        } else {
          setRecommendation({
            title: "Balance",
            text: "Forecasts indicate that renewable generation and load demand are balanced within a ±10% range. Maintain current grid operations and monitor system frequency. You may optimize the charge/discharge cycle of storage units and schedule minor grid maintenance during this stable period.",
          });
        }
      } else {
        setRecommendation({
          title: "No Recommendation",
          text: "Insufficient data to generate a recommendation. Ensure forecast data, energy demand, and max capacity are available and valid.",
        });
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

      const results = await new Promise((resolve, reject) => {
        Papa.parse(csvText, {
          header: true,
          dynamicTyping: true,
          skipEmptyLines: true,
          complete: (results) => resolve(results),
          error: (error) => reject(error),
        });
      });

      const firstRow = results.data[0] || {};
      const hasSpecificForecastColumns =
        "dhr_forecast" in firstRow ||
        "esn_forecast" in firstRow ||
        "hybrid_forecast" in firstRow;

      if (hasSpecificForecastColumns) {
        setCsvData(results.data);
        createChartFromCsvData(results.data);
        return results.data;
      } else {
        const processedData = results.data.map((row) => {
          const timestampKey = Object.keys(row).find(
            (key) =>
              key.toLowerCase().includes("time") ||
              key.toLowerCase().includes("date")
          );
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
        createSimpleChartFromCsvData(processedData);
        return processedData;
      }
    } catch (err) {
      console.error("Error fetching CSV data:", err);
      setError(err.message);
      return null;
    }
  };

  const createChartFromCsvData = (data) => {
    if (!data || data.length === 0) {
      setError("No data available in CSV file");
      return;
    }

    const hasDatetime = "datetime" in data[0];
    const hasTimestamp = "timestamp" in data[0];
    if (!hasDatetime && !hasTimestamp) {
      setError("CSV file missing required 'datetime' or 'timestamp' column");
      return;
    }

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

    const validData = data.filter((item) => {
      const dateKey = hasDatetime ? "datetime" : "timestamp";
      const hasDate =
        item[dateKey] !== undefined &&
        item[dateKey] !== null &&
        item[dateKey] !== "";
      const hasAtLeastOneForecast =
        (item.dhr_forecast !== undefined && !isNaN(item.dhr_forecast)) ||
        (item.esn_forecast !== undefined && !isNaN(item.esn_forecast)) ||
        (item.hybrid_forecast !== undefined && !isNaN(item.hybrid_forecast));
      return hasDate && hasAtLeastOneForecast;
    });

    if (validData.length === 0) {
      setError("No valid data with dates and forecast values found in CSV");
      return;
    }

    const datasets = [];

    if ("dhr_forecast" in validData[0]) {
      datasets.push({
        label: "DHR Forecast",
        data: validData.map((item) =>
          isNaN(item.dhr_forecast) ? null : item.dhr_forecast
        ),
        borderColor: colorSchemes[0].borderColor,
        backgroundColor: colorSchemes[0].backgroundColor,
        tension: 0.1,
        fill: false,
      });
    }

    if ("esn_forecast" in validData[0]) {
      datasets.push({
        label: "ESN Forecast",
        data: validData.map((item) =>
          isNaN(item.esn_forecast) ? null : item.esn_forecast
        ),
        borderColor: colorSchemes[1].borderColor,
        backgroundColor: colorSchemes[1].backgroundColor,
        tension: 0.1,
        fill: false,
      });
    }

    if ("hybrid_forecast" in validData[0]) {
      datasets.push({
        label: "Hybrid Forecast",
        data: validData.map((item) =>
          isNaN(item.hybrid_forecast) ? null : item.hybrid_forecast
        ),
        borderColor: colorSchemes[2].borderColor,
        backgroundColor: colorSchemes[2].backgroundColor,
        tension: 0.1,
        fill: false,
      });
    }

    if (datasets.length === 0) {
      setError("No valid forecast columns found in CSV data");
      return;
    }

    const dateKey = hasDatetime ? "datetime" : "timestamp";
    setChartData({
      labels: validData.map((item) => item[dateKey]),
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
      let modelType;
      if (!forecastData) {
        const forecastResponse = await fetch(
          `http://localhost:8000/api/forecasts/${forecastId}`
        );
        if (!forecastResponse.ok) {
          throw new Error(
            `Error fetching forecast: ${forecastResponse.statusText}`
          );
        }
        const data = await forecastResponse.json();
        setForecastData(data);
        modelType = (data.model || "").toUpperCase();
      } else {
        modelType = (forecastData.model || "").toUpperCase();
      }

      const normalizedModelType =
        modelType === "DHR-ESN" ? "HYBRID" : modelType;
      console.log("Model type:", modelType);

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
      } else if (modelType === "DHR-ESN") {
        const hybridResponse = await fetch(
          `http://localhost:8000/api/hybrid-configurations/${forecastId}`
        );
        if (hybridResponse.ok) {
          const hybridData = await hybridResponse.json();
          console.log("Hybrid config fetched:", hybridData);

          if (hybridData.dhr_config || hybridData.dhr) {
            setDhrConfig(hybridData.dhr_config || hybridData.dhr);
          }
          if (hybridData.esn_config || hybridData.esn) {
            setEsnConfig(hybridData.esn_config || hybridData.esn);
          }

          if (
            !hybridData.dhr_config &&
            !hybridData.esn_config &&
            !hybridData.dhr &&
            !hybridData.esn
          ) {
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

            const extractedDhrConfig = {};
            const extractedEsnConfig = {};

            possibleDhrFields.forEach((field) => {
              if (field in hybridData) {
                extractedDhrConfig[field] = hybridData[field];
              }
            });

            possibleEsnFields.forEach((field) => {
              if (field in hybridData) {
                extractedEsnConfig[field] = hybridData[field];
              }
            });

            if (Object.keys(extractedDhrConfig).length > 0) {
              setDhrConfig(extractedDhrConfig);
            }
            if (Object.keys(extractedEsnConfig).length > 0) {
              setEsnConfig(extractedEsnConfig);
            }
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
      legend: { position: "top" },
      title: {
        display: true,
        text: "Power Generation Forecast",
      },
    },
    scales: {
      x: {
        display: true,
        title: { display: true, text: "Time" },
      },
      y: {
        display: true,
        title: { display: true, text: "Power (kW)" },
      },
    },
  };

  const hasMultipleForecasts =
    csvData &&
    csvData.length > 0 &&
    ("dhr_forecast" in csvData[0] ||
      "esn_forecast" in csvData[0] ||
      "hybrid_forecast" in csvData[0]);

  // Get recommendation actions based on current state
  const [actionA, actionB] = getRecommendationActions(
    recommendation.title,
    forecastData?.granularity,
    forecastData?.steps
  );

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
            className="w-full border border-gray-300 rounded p-2"
          />
        </div>
        <div>
          <label className="block font-medium">Forecast File</label>
          <input
            type="text"
            value={forecastData?.filename || ""}
            readOnly
            className="w-full border border-gray-300 rounded p-2"
          />
        </div>
        <div>
          <label className="block font-medium">Granularity</label>
          <input
            type="text"
            value={forecastData?.granularity || ""}
            readOnly
            className="w-full border border-gray-300 rounded p-2"
          />
        </div>
        <div>
          <label className="block font-medium">Steps</label>
          <input
            type="text"
            value={forecastData?.steps || ""}
            readOnly
            className="w-full border border-gray-300 rounded p-2"
          />
        </div>
        <div>
          <label className="block font-medium">Energy Demand (kW)</label>
          <input
            type="text"
            value={energyDemand ?? "N/A"}
            readOnly
            className="w-full border border-gray-300 rounded p-2"
          />
        </div>
        <div>
          <label className="block font-medium">Max Capacity (kW)</label>
          <input
            type="text"
            value={maxCapacity ?? "N/A"}
            readOnly
            className="w-full border border-gray-300 rounded p-2"
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
      <div>
        {/* Header */}
        <div className="flex justify-between items-center mb-4">
          <h1 className="text-xl font-bold">
            {historyLog?.file_name ||
              forecastData?.original_filename ||
              "Forecast Details"}
          </h1>
          <button
            onClick={handleBack}
            className="px-4 py-2 bg-red-500 border-none text-white rounded-xl hover:bg-red-600 cursor-pointer">
            Back
          </button>
        </div>

        {error && <p className="text-red-600 mb-4">{error}</p>}

        {forecastId ? (
          <>
            <GraphPlaceholder />
            <div className="flex flex-col md:flex-row gap-6">
              <div className="w-full md:w-1/2">
                <DatasetDetails />
              </div>
              <div className="w-full md:w-1/2">
                <div className="mb-6 bg-white p-4 rounded-lg">
                  <h3 className="text-gray-700 font-semibold mb-4">
                    Recommendation
                  </h3>
                  {recommendation.title ? (
                    <div>
                      <h3 className="font-semibold">{recommendation.title}</h3>
                      <p className="text-gray-600 text-justify leading-relaxed">
                        {recommendation.text}
                      </p>
                      <div className="space-y-2 mt-4">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <span className="text-green-600 font-medium">
                              A
                            </span>
                            <p className="text-sm">{actionA}</p>
                          </div>
                        </div>
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <span className="text-green-600 font-medium">
                              B
                            </span>
                            <p className="text-sm">{actionB}</p>
                          </div>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <p className="text-gray-600 text-justify leading-relaxed">
                      No recommendation available
                    </p>
                  )}
                </div>
              </div>
            </div>
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

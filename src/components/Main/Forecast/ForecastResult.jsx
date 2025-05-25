import { useLocation, useNavigate } from "react-router-dom";
import { useState, useEffect } from "react";
import Papa from "papaparse";

export default function ForecastResult() {
  const navigate = useNavigate();
  const { state } = useLocation();
  console.log("Received state:", state);

  const imageUrl = state?.imageUrl;
  const [forecastData, setForecastData] = useState(state?.forecastData || {});
  const [config, setConfig] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [recommendation, setRecommendation] = useState(""); // Moved to state
  const [parsedCsvData, setParsedCsvData] = useState([]);
  const [showModal, setShowModal] = useState(false);

  // Placeholder values for energyDemand and maxCapacity (replace with actual data source)
  const energyDemand = 1000; // Example value in kW, replace with fetched or passed data
  const maxCapacity = 1200; // Example value in kW, replace with fetched or passed data

  // Handle modal close with keyboard (accessibility)
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === "Escape") setShowModal(false);
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  // Calculate recommendation based on parsed CSV data
  useEffect(() => {
    if (parsedCsvData.length === 0) return;

    // Extract forecast values
    const forecastValues = parsedCsvData
      .map((row) => {
        const value =
          row.hybrid_forecast ??
          row.forecasted_solar_power ??
          row.forecasted_wind_power ??
          row.forecast ??
          null;
        return value !== null && !isNaN(value) ? Number(value) : null;
      })
      .filter((value) => value !== null);

    if (forecastValues.length === 0) {
      setRecommendation("No valid forecast data available.");
      return;
    }

    // Calculate mean
    const sum = forecastValues.reduce((acc, val) => acc + val, 0);
    const mean = sum / forecastValues.length;

    // Normalize demand
    const normDemand = energyDemand / maxCapacity;
    const upperbound = 0.9 * normDemand;
    const lowerbound = 1.1 * normDemand;

    // Set recommendation
    if (mean > upperbound) {
      setRecommendation(
        <div>
          <h3 class="semi-bold">Overgenerate</h3>
          <p>
            Forecast analysis shows that generation is likely to exceed demand
            by more than 10%. Please begin charging battery energy storage
            systems, consider exporting excess power to the external grid if
            available, and initiate curtailment of solar or wind units to
            prevent grid overvoltage. You may also notify large consumers to
            increase their load through demand response programs.
          </p>
        </div>
      );
    } else if (mean < lowerbound) {
      setRecommendation(
        <div>
          <h3 class="semi-bold">Undergenerated</h3>
          <p>
            The system anticipates a generation shortfall of over 10% compared
            to demand. Please dispatch backup generation units immediately,
            initiate energy imports if grid interconnection is available, and
            issue a demand response call to reduce load in non-critical sectors.
            Pre-charge energy storage systems during off-peak hours if time
            permits.
          </p>
        </div>
      );
    } else {
      setRecommendation(
        <div>
          <h3 class="semi-bold">Balance</h3>
          <p>
            Forecasts indicate that renewable generation and load demand are
            balanced within a ±10% range. Maintain current grid operations and
            monitor system frequency. You may optimize the charge/discharge
            cycle of storage units and schedule minor grid maintenance during
            this stable period.
          </p>
        </div>
      );
    }

    console.log("Mean of forecast values:", mean.toFixed(2));
  }, [parsedCsvData, energyDemand, maxCapacity]);

  // Fetch forecast data
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

        const response = await fetch(
          `http://localhost:8000/api/forecasts/${forecastId}`,
          {
            headers: { Accept: "application/json" },
          }
        );

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(
            `Error fetching forecast: ${response.status} - ${errorText}`
          );
        }

        const data = await response.json();
        setForecastData((prev) => ({ ...prev, ...data }));
      } catch (err) {
        console.error("Error fetching forecast data:", err);
        setError(err.message);
      } finally {
        setIsLoading(false);
      }
    };

    fetchForecastData();
  }, [forecastData?.id]);

  // Fetch and parse CSV file content
  useEffect(() => {
    const fetchFileContent = async () => {
      if (!forecastData?.filename) return;

      try {
        const response = await fetch(
          `http://localhost:8000/api/forecast-file/${forecastData.filename}`,
          { headers: { Accept: "text/csv" } }
        );

        if (!response.ok) {
          throw new Error(
            `Failed to fetch file content: ${response.statusText}`
          );
        }

        const text = await response.text();
        Papa.parse(text, {
          header: true,
          dynamicTyping: true,
          skipEmptyLines: true,
          complete: (results) => {
            const firstRow = results.data[0] || {};
            const forecastColumn = [
              "hybrid_forecast",
              "forecasted_solar_power",
              "forecasted_wind_power",
              "forecast",
              "value",
              "power",
            ].find((col) => col in firstRow && firstRow[col] !== null);

            if (forecastColumn) {
              const forecastData = results.data.map((row) => ({
                forecast: row[forecastColumn] ?? null,
              }));
              setParsedCsvData(forecastData);
            } else {
              console.warn("No forecast column found in CSV");
              setParsedCsvData([]);
              setError("No valid forecast data found in CSV");
            }
          },
          error: (error) => {
            console.error("CSV parsing error:", error);
            setError("Error parsing CSV data");
          },
        });
      } catch (err) {
        console.error("Error fetching file content:", err);
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
        const modelKey =
          forecastData.model.toLowerCase() === "dhr-esn"
            ? "hybrid"
            : forecastData.model.toLowerCase();
        const response = await fetch(
          `http://localhost:8000/api/${modelKey}-configurations/${forecastData.id}`
        );

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
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

  // Navigate to graph view
  const handleViewGraphs = () => {
    navigate("/view-graph", {
      state: { data: parsedCsvData, forecastId: forecastData?.id },
    });
  };

  // Navigate to edit page
  const handleEdit = () => {
    const queryParams = new URLSearchParams({
      model: forecastData.model || "",
      originalFileName:
        forecastData.filename || forecastData.original_filename || "",
      forecastType: forecastData.forecastType || "",
      steps: forecastData.steps || "",
      granularity: forecastData.granularity || "",
      forecastId: forecastData.id || "",
    }).toString();

    const state = {
      forecastId: forecastData.id,
      model: forecastData.model,
      isEditing: true,
      existingConfig: config,
    };

    navigate(
      forecastData.model === "DHR-ESN"
        ? `/configure-hybrid?${queryParams}`
        : `/configure-single?${queryParams}`,
      { state }
    );
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

  // Refactored configuration rendering
  const renderConfigSection = () => {
    if (isLoading) {
      return <div className="text-gray-600">Loading configuration...</div>;
    }

    if (error) {
      return (
        <div className="text-red-500 bg-red-50 p-4 rounded-md">
          Error: {error}
        </div>
      );
    }

    if (!config) {
      return <div className="text-gray-600">No configuration available</div>;
    }

    const renderInput = (label, value) => (
      <div>
        <p className="text-sm text-gray-600">{label}</p>
        <input
          type="text"
          value={value ?? "-"}
          disabled
          className="w-full p-2 bg-gray-100 border border-gray-300 rounded-md text-gray-700 font-semibold cursor-not-allowed"
        />
      </div>
    );

    const configSections = {
      "DHR-ESN": [
        {
          title: "Dynamic Harmonic Regression",
          fields: [
            { label: "Fourier Order", value: config.fourier_terms },
            { label: "Window Length", value: config.window },
            { label: "Polyorder", value: config.polyorder },
            { label: "Regularization (DHR)", value: config.reg_strength },
            { label: "AR Order", value: config.ar_order },
          ],
        },
        {
          title: "Echo State Networks",
          fields: [
            { label: "Reservoir Size", value: config.N_res },
            { label: "Spectral Radius", value: config.rho },
            { label: "Sparsity", value: config.sparsity },
            { label: "Input Scaling", value: config.alpha },
            {
              label: "Regularization",
              value: config.lambda_reg
                ? Number(config.lambda_reg).toFixed(9)
                : null,
            },
            { label: "Lags", value: config.lags },
            { label: "N Features", value: config.n_features },
          ],
        },
      ],
      DHR: [
        {
          title: "Dynamic Harmonic Regression",
          fields: [
            { label: "Fourier Order", value: config.fourier_terms },
            { label: "Window Length", value: config.window },
            { label: "Polyorder", value: config.polyorder },
            { label: "Regularization", value: config.reg_strength },
            { label: "AR Order", value: config.ar_order },
          ],
        },
      ],
      ESN: [
        {
          title: "Echo State Networks",
          fields: [
            { label: "Reservoir Size", value: config.N_res },
            { label: "Spectral Radius", value: config.rho },
            { label: "Sparsity", value: config.sparsity },
            { label: "Input Scaling", value: config.alpha },
            {
              label: "Regularization",
              value: config.lambda_reg
                ? Number(config.lambda_reg).toFixed(9)
                : null,
            },
            { label: "Lags", value: config.lags },
          ],
        },
      ],
    };

    const sections = configSections[forecastData.model] || [];

    return (
      <div className="space-y-8">
        {sections.map((section) => (
          <div key={section.title}>
            <h3 className="text-lg font-semibold mb-4">{section.title}</h3>
            <div className="grid grid-cols-2 gap-x-7">
              {section.fields.map((field) =>
                renderInput(field.label, field.value)
              )}
            </div>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="max-w-5xl mx-auto p-4">
      {error && (
        <div className="mb-4 p-4 bg-red-50 text-red-500 rounded-md">
          {error}
        </div>
      )}
      <div className="grid grid-cols-2 gap-6">
        {/* Left Column */}
        <div className="space-y-6">
          {/* Graph Card */}
          <div className="bg-white p-4 rounded-lg border-gray-200 shadow">
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
                  {showModal && (
                    <div
                      className="fixed inset-0 bg-black bg-opacity-75 z-50 flex items-center justify-center p-4"
                      onClick={() => setShowModal(false)}
                      role="dialog"
                      aria-label="Enlarged forecast plot">
                      <div className="relative max-w-4xl max-h-[90vh]">
                        <button
                          className="absolute top-4 right-4 text-white hover:text-gray-300 text-xl"
                          onClick={() => setShowModal(false)}
                          aria-label="Close modal">
                          ×
                        </button>
                        <img
                          src={imageUrl}
                          alt="Enlarged Forecast Plot"
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
            <div className="flex items-center justify-between">
              <button
                onClick={handleViewGraphs}
                className="py-2 px-4 bg-gray-200 text-black rounded-md hover:bg-gray-300 border-b-2 border-green-500 cursor-pointer"
                disabled={isLoading}>
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
            <h3 className="text-gray-700 font-semibold mb-4">Recommendation</h3>
            <p className="text-gray-600 text-justify leading-relaxed">
              {recommendation || "No recommendation available"}
            </p>
          </div>
        </div>

        {/* Right Column - Configurations */}
        <div className="space-y-6">
          <h2 className="text-3xl font-bold mb-6">Configurations</h2>
          {renderConfigSection()}
          <div className="flex space-x-4 mt-8 mb-4">
            <button
              onClick={handleEdit}
              className="bg-green-600 text-white px-6 py-2 rounded-md hover:bg-green-700 cursor-pointer"
              disabled={isLoading}>
              Edit
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

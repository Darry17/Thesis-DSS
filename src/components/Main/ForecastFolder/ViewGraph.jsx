import React, { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";

const ViewGraph = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { forecastId, filename } = location.state || {};
  const [metrics, setMetrics] = useState({
    rmse: "",
    cvrmse: "",
    mae: "",
  });

  // Add print stylesheet
  useEffect(() => {
    // Create a style element
    const style = document.createElement("style");
    style.id = "print-style";
    style.innerHTML = `
      @media print {
        /* Hide everything by default */
        body * {
          visibility: hidden;
        }
        
        /* Only show the print container and its contents */
        .print-container, .print-container * {
          visibility: visible;
        }
        
        /* Additional specificity for elements we want to hide */
        .no-print, 
        nav, 
        header, 
        footer, 
        aside, 
        button, 
        .nav-tabs, 
        .tabs,
        .navigation,
        [role="navigation"],
        a[href="/dashboard"],
        a[href="/forecast"],
        a[href="/history"],
        div:has(> a[href="/dashboard"]),
        div:has(> a[href="/forecast"]),
        div:has(> a[href="/history"]) {
          display: none !important;
        }
        
        /* Position the print container at the top of the page */
        .print-container {
          position: absolute;
          left: 0;
          top: 0;
          width: 100%;
        }
        
        /* Reset any margins or padding for clean print */
        body {
          margin: 0;
          padding: 0;
        }
        
        /* Remove background colors */
        body, .print-container {
          background-color: white !important;
        }
      }
    `;
    document.head.appendChild(style);

    // Cleanup function
    return () => {
      const styleElement = document.getElementById("print-style");
      if (styleElement) {
        document.head.removeChild(styleElement);
      }
    };
  }, []);

  // Action handlers
  const handleSaveForecast = async () => {
    try {
      console.log("Saving forecast with ID:", forecastId);

      if (!forecastId) {
        console.error("Missing forecastId - cannot save history log");
        alert("Error: Missing forecast ID. Cannot save to history log.");
        return;
      }

      // First, fetch the forecast details to get the model type
      const response = await fetch(
        `http://localhost:8000/api/forecasts/${forecastId}`
      );

      if (!response.ok) {
        throw new Error(`Error fetching forecast: ${response.statusText}`);
      }

      const forecastData = await response.json();
      let modelType = forecastData.model.toLowerCase(); // Get the model type (esn, dhr, hybrid)

      // For file naming purposes
      let fileModelType = modelType;

      // For the model column in history logs
      let displayModelType = modelType.toUpperCase();

      // If the model is dhr-esn, set file naming to use "hybrid" but keep display as DHR-ESN
      if (modelType === "dhr-esn") {
        fileModelType = "hybrid";
        displayModelType = "DHR-ESN";
      }

      // Format the current date as YYYY-MM-DD
      const today = new Date();
      const formattedDate = `${today.getFullYear()}-${String(
        today.getMonth() + 1
      ).padStart(2, "0")}-${String(today.getDate()).padStart(2, "0")}`;

      // Create the file_name in the required format: model-date
      const fileName = `${fileModelType}-${formattedDate}`;

      // Get metric values (either from state or use defaults)
      const rmseValue = parseFloat(metrics.rmse || "3");
      const cvrmseValue = parseFloat(metrics.cvrmse || "3");
      const maeValue = parseFloat(metrics.mae || "3");

      console.log("Creating history log with metrics:", {
        file_name: fileName,
        model: displayModelType,
        forecast_id: forecastId,
        rmse: rmseValue,
        cvrmse: cvrmseValue,
        mae: maeValue,
      });

      // Create the history log entry with metrics
      const historyLogResponse = await fetch(
        "http://localhost:8000/api/history-logs",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            file_name: fileName,
            model: displayModelType,
            forecast_id: forecastId,
            rmse: rmseValue,
            cvrmse: cvrmseValue,
            mae: maeValue,
          }),
        }
      );

      if (!historyLogResponse.ok) {
        const errorText = await historyLogResponse.text();
        console.error("Error response:", errorText);
        throw new Error(
          `Error creating history log: ${historyLogResponse.status} ${historyLogResponse.statusText} - ${errorText}`
        );
      }

      const responseData = await historyLogResponse.json();
      console.log("History log created successfully:", responseData);

      // Success message
      alert(`Forecast saved successfully as ${fileName}`);
    } catch (error) {
      console.error("Error saving forecast:", error);
      alert(`Error saving forecast: ${error.message}`);
    }
  };

  const handleBack = () => navigate(-1);
  const handlePrint = () => window.print();

  // Handle metrics change
  const handleMetricsChange = (e) => {
    const { name, value } = e.target;
    setMetrics((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  // Graph placeholder component
  const GraphPlaceholder = ({ title }) => (
    <div className="border rounded-md p-4 bg-white shadow mb-6">
      <h3 className="text-lg font-medium mb-2">{title}</h3>
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

  // Evaluation metrics component
  const EvaluationMetrics = () => (
    <div className="border rounded-md p-4 bg-white shadow">
      <h3 className="text-2xl font-bold mb-6">Evaluation Metrics</h3>
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Root Mean Squared Error (RMSE)
          </label>
          <input
            type="text"
            name="rmse"
            value={metrics.rmse || "3"}
            readOnly
            className="w-full p-2 bg-gray-50 border border-gray-300 rounded-md text-gray-700 focus:outline-none cursor-default"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Coefficient Variation of the RMSE (CV(RMSE))
          </label>
          <input
            type="text"
            name="cvrmse"
            value={metrics.cvrmse || "3"}
            readOnly
            className="w-full p-2 bg-gray-50 border border-gray-300 rounded-md text-gray-700 focus:outline-none cursor-default"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Mean Absolute Error (MAE)
          </label>
          <input
            type="text"
            name="mae"
            value={metrics.mae || "3"}
            readOnly
            className="w-full p-2 bg-gray-50 border border-gray-300 rounded-md text-gray-700 focus:outline-none cursor-default"
          />
        </div>
      </div>
    </div>
  );

  // Graph types
  const graphTypes = ["Generated Power"];

  return (
    <div className="p-6 max-w-7xl mx-auto print-container">
      {/* Action buttons - will be hidden when printing */}
      <div className="flex justify-between mb-6 no-print">
        <button
          onClick={handleSaveForecast}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">
          Save Forecast
        </button>
        <div className="space-x-2">
          <button
            onClick={handleBack}
            className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600">
            Back
          </button>
          <button
            onClick={handlePrint}
            className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700">
            Print
          </button>
        </div>
      </div>

      {/* Main content layout */}
      <div className="space-y-6">
        {/* Graphs section */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {graphTypes.map((type, index) => (
            <GraphPlaceholder key={index} title={type} />
          ))}
        </div>

        {/* Metrics section - positioned at the bottom left */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="md:col-span-1">
            <EvaluationMetrics />
          </div>
          <div className="md:col-span-2 no-print">
            {/* Empty space to ensure metrics are on the left */}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ViewGraph;

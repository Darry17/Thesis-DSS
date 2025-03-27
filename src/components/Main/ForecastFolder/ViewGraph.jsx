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
      console.log("Saving forecast:", forecastId);

      // First, fetch the forecast details to get the model type
      const response = await fetch(
        `http://localhost:8000/api/forecasts/${forecastId}`
      );

      if (!response.ok) {
        throw new Error(`Error fetching forecast: ${response.statusText}`);
      }

      const forecastData = await response.json();
      const modelType = forecastData.model.toLowerCase(); // Get the model type (esn, dhr, hybrid)

      // Format the current date as YYYY-MM-DD
      const today = new Date();
      const formattedDate = `${today.getFullYear()}-${String(
        today.getMonth() + 1
      ).padStart(2, "0")}-${String(today.getDate()).padStart(2, "0")}`;

      // Create the file_name in the required format: model-date
      const fileName = `${modelType}-${formattedDate}`;

      // Create the history log entry
      const historyLogResponse = await fetch(
        "http://localhost:8000/api/history-logs",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            file_name: fileName,
            model: modelType.toUpperCase(),
            forecast_id: forecastId,
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
    <div className="border rounded-md p-4 bg-white shadow">
      <h3 className="text-lg font-medium mb-2">{title}</h3>
      <div className="aspect-square w-full bg-gray-100 relative">
        {/* Graph axes */}
        <div className="absolute left-8 top-0 h-full border-l border-gray-300"></div>
        <div className="absolute left-0 bottom-8 w-full border-b border-gray-300"></div>

        {/* Y-axis labels */}
        <div className="absolute left-2 top-2 text-xs text-gray-500">100</div>
        <div className="absolute left-2 top-1/2 -translate-y-1/2 text-xs text-gray-500">
          50
        </div>
        <div className="absolute left-2 bottom-4 text-xs text-gray-500">0</div>

        {/* X-axis labels */}
        <div className="absolute left-8 bottom-2 text-xs text-gray-500">50</div>
        <div className="absolute right-2 bottom-2 text-xs text-gray-500">
          100
        </div>

        {/* Example SVG line graph */}
        <svg
          className="absolute inset-0 p-8"
          viewBox="0 0 100 100"
          preserveAspectRatio="none">
          <path
            d="M0,20 C10,15 20,35 30,25 C40,15 50,25 60,20 C70,15 80,30 90,40 L90,100 L0,100 Z"
            fill="none"
            stroke="green"
            strokeWidth="2"
          />
        </svg>
      </div>
    </div>
  );

  // Evaluation metrics component
  const EvaluationMetrics = () => (
    <div className="border rounded-md p-4 bg-white shadow">
      <h3 className="text-lg font-medium mb-3">Evaluation Metrics</h3>
      <div className="space-y-4">
        <div>
          <label className="block text-sm text-gray-700 mb-1">
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
          <label className="block text-sm text-gray-700 mb-1">
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
          <label className="block text-sm text-gray-700 mb-1">
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

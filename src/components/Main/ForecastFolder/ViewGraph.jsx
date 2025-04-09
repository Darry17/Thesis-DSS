import React, { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";

const ViewGraph = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { forecastId, filename } = location.state || {};
  const [forecastData, setForecastData] = useState(null); // State for forecast data
  const [loading, setLoading] = useState(true); // Optional: Add loading state
  const [error, setError] = useState(null); // Optional: Add error state

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
        .print-container {
          position: absolute;
          left: 0;
          top: 0;
          width: 100%;
        }
        body {
          margin: 0;
          padding: 0;
        }
        body, .print-container {
          background-color: white !important;
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

  // Fetch forecast data on mount
  useEffect(() => {
    const fetchForecastData = async () => {
      try {
        if (!forecastId) {
          throw new Error("No forecast ID provided");
        }

        const token = localStorage.getItem("token");
        if (!token) {
          navigate("/login");
          return;
        }

        setLoading(true);
        const response = await fetch(
          `http://localhost:8000/api/forecasts/${forecastId}`,
          {
            headers: {
              Authorization: `Bearer ${token}`,
            },
          }
        );

        if (!response.ok) {
          throw new Error(`Error fetching forecast: ${response.statusText}`);
        }

        const data = await response.json();
        setForecastData(data);
      } catch (err) {
        console.error("Error fetching forecast data:", err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchForecastData();
  }, [forecastId, navigate]);

  const handleBack = () => navigate(-1);
  const handlePrint = () => window.print();

  // Graph placeholder component
  const GraphPlaceholder = ({ title }) => (
    <div className="bg-white p-4 rounded-lg border-gray-500 shadow">
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

  // DatasetDetails component
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
            value={forecastData?.original_filename || "N/A"}
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
            value={forecastData?.steps || "24 (1-day Horizon)"}
            readOnly
            className="w-80 p-2 bg-gray-50 border border-gray-300 rounded-md text-gray-700 focus:outline-none cursor-default"
          />
        </div>
      </div>
    </div>
  );

  // Graph types
  const graphTypes = ["Generated Power"];

  if (loading) {
    return <div>Loading...</div>;
  }

  if (error) {
    return <div>Error: {error}</div>;
  }

  return (
    <div className="p-6 max-w-7xl mx-auto print-container">
      <div className="flex justify-end mb-6 no-print">
        <div className="space-x-8">
          <button
            onClick={handleBack}
            className="px-4 py-2 bg-red-500 text-white rounded-xl hover:bg-red-600">
            Back
          </button>
          <button
            onClick={handlePrint}
            className="px-4 py-2 bg-green-600 text-white rounded-xl hover:bg-green-700">
            Print
          </button>
        </div>
      </div>

      <div className="space-y-6">
        {graphTypes.map((type, index) => (
          <GraphPlaceholder key={index} title={type} />
        ))}
        {/* Add DatasetDetails after the graph */}
        <DatasetDetails />
      </div>
    </div>
  );
};

export default ViewGraph;

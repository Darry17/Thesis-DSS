import React, { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";

const ViewGraph = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { forecastId } = location.state || {};
  const [loading, setLoading] = useState(false);

  // Handle save forecast
  const handleSaveForecast = () => {
    // Implement save functionality here
    console.log("Saving forecast:", forecastId);
    // You could trigger a download or save to database
  };

  // Handle go back
  const handleBack = () => {
    navigate(-1);
  };

  // Handle print
  const handlePrint = () => {
    window.print();
  };

  // Array of graph types to display
  const graphTypes = [
    "Generated Power",
    "DNI",
    "Zenith Angle",
    "DHI",
    "GHI",
    "Temperature",
  ];

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Button row */}
      <div className="flex justify-between mb-6">
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

      {/* Grid of graphs */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {graphTypes.map((type, index) => (
          <div key={index} className="border rounded-md p-4 bg-white shadow">
            <h3 className="text-lg font-medium mb-2">{type}</h3>
            {/* Placeholder for graph - square shape */}
            <div className="aspect-square w-full bg-gray-100 flex items-center justify-center relative">
              {/* Create a simple placeholder graph with green line */}
              <div className="w-full h-full p-4 relative">
                {/* Graph axes */}
                <div className="absolute left-8 top-0 h-full border-l border-gray-300"></div>
                <div className="absolute left-0 bottom-8 w-full border-b border-gray-300"></div>

                {/* Y-axis labels */}
                <div className="absolute left-2 top-2 text-xs text-gray-500">
                  100
                </div>
                <div className="absolute left-2 top-1/2 -translate-y-1/2 text-xs text-gray-500">
                  50
                </div>
                <div className="absolute left-2 bottom-4 text-xs text-gray-500">
                  0
                </div>

                {/* X-axis labels */}
                <div className="absolute left-8 bottom-2 text-xs text-gray-500">
                  50
                </div>
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
          </div>
        ))}
      </div>
    </div>
  );
};

export default ViewGraph;

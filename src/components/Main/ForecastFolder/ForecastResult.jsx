import React from "react";

const ForecastResult = () => {
  return (
    <div className="p-6 max-w-5xl mx-auto">
      <div className="grid grid-cols-2 gap-6">
        {/* Left Column */}
        <div className="space-y-6">
          {/* Graph Card */}
          <div className="bg-white p-4 rounded-lg shadow">
            <h3 className="text-sm text-gray-500 mb-2">
              forecast-wind-data.csv
            </h3>
            <h2 className="font-bold mb-4">DHR-ESN</h2>
            {/* Replace with actual graph component */}
            <div className="h-48 bg-gray-100 rounded mb-4">
              {/* Graph will go here */}
            </div>
            <div className="text-sm">View Results</div>
          </div>

          {/* Forecast Period */}
          <div className="bg-white p-4 rounded-lg shadow">
            <h3 className="text-gray-700 font-semibold mb-2">
              Generated Forecast
            </h3>
            <p className="text-xl font-bold">24 Hours</p>
          </div>

          {/* Recommendations */}
          <div className="bg-white p-4 rounded-lg shadow">
            <h3 className="text-gray-700 font-semibold mb-4">
              Recommendations
            </h3>
            <p className="text-sm text-gray-600 mb-4">
              Production of solar energy will be 25% lower compared to the last
              24 hours
            </p>
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <div className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center text-white">
                  ✓
                </div>
                <p className="text-sm">Pre-charge battery storage overnight</p>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center text-white">
                  ✓
                </div>
                <p className="text-sm">
                  Bid strategically in the day-ahead market
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column - Configurations */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-2xl font-bold mb-6">Configurations</h2>

          {/* DHR Section */}
          <div className="mb-8">
            <h3 className="text-lg font-semibold mb-4">
              Dynamic Harmonic Regression
            </h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-gray-600">Fourier Order</p>
                <p className="font-semibold">3</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Trend Components</p>
                <p className="font-semibold">2</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Seasonality Periods</p>
                <p className="font-semibold">M</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Window Length</p>
                <p className="font-semibold">1</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Regularization</p>
                <p className="font-semibold">1e-4</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Polyorder</p>
                <p className="font-semibold">0.1</p>
              </div>
            </div>
          </div>

          {/* ESN Section */}
          <div>
            <h3 className="text-lg font-semibold mb-4">Echo State Networks</h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-gray-600">Reservoir Size</p>
                <p className="font-semibold">500</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Input Scaling</p>
                <p className="font-semibold">0.3</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Spectral Radius</p>
                <p className="font-semibold">0.9</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Regularization</p>
                <p className="font-semibold">0.2</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Sparsity</p>
                <p className="font-semibold">1.0</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Dropout</p>
                <p className="font-semibold">0.2</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Lags</p>
                <p className="font-semibold">1</p>
              </div>
            </div>
          </div>

          {/* Edit Button */}
          <button className="mt-8 bg-green-500 text-white px-6 py-2 rounded-md hover:bg-green-600 w-full">
            Edit
          </button>
        </div>
      </div>
    </div>
  );
};

export default ForecastResult;

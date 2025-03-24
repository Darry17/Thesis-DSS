import React, { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";

const SingleModelConfiguration = () => {
  const location = useLocation();
  const navigate = useNavigate();

  // Get the model directly from location.state.model
  const selectedModel = location.state?.model;

  const getModelTitle = () => {
    if (!selectedModel) return "MODEL CONFIGURATION";

    switch (selectedModel) {
      case "DHR":
        return "DYNAMIC HARMONIC REGRESSION";
      case "ESN":
        return "ECHO STATE NETWORK";
      default:
        return "MODEL CONFIGURATION";
    }
  };

  // State to manage form inputs
  const [formData, setFormData] = useState({
    // ESN fields
    reservoirSize: "",
    spectralRadius: "",
    sparsity: "",
    inputScaling: "",
    regularization: "",
    dropout: "",
    lags: "",
    // DHR fields
    fourierOrder: "",
    windowLength: "",
    seasonalityPeriods: "",
    polyorder: "",
    trendComponents: "",
  });

  // Handle input changes
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: value,
    }));
  };

  // Handle form submission
  const handleSubmit = (e) => {
    e.preventDefault();
    console.log("Form submitted:", formData);
  };

  // Handle cancel button
  const handleCancel = () => {
    navigate(-1);
  };

  const renderForm = () => {
    if (selectedModel === "ESN") {
      return (
        <div className="p-6 flex justify-center items-center">
          <form onSubmit={handleSubmit} className="space-y-4">
            {/* First Row */}
            <div className="flex space-x-4">
              <div className="flex-1">
                <label className="block text-sm font-medium mb-1">
                  Reservoir Size{" "}
                  <span className="text-gray-500 cursor-pointer">ⓘ</span>
                </label>
                <input
                  type="text"
                  name="reservoirSize"
                  value={formData.reservoirSize}
                  onChange={handleChange}
                  placeholder="500"
                  className="w-full p-2 border border-gray-300 rounded-md"
                />
              </div>
              <div className="flex-1">
                <label className="block text-sm font-medium mb-1">
                  Regularization{" "}
                  <span className="text-gray-500 cursor-pointer">ⓘ</span>
                </label>
                <input
                  type="text"
                  name="regularization"
                  value={formData.regularization}
                  onChange={handleChange}
                  placeholder="0.2"
                  className="w-full p-2 border border-gray-300 rounded-md"
                />
              </div>
            </div>

            {/* Second Row */}
            <div className="flex space-x-4">
              <div className="flex-1">
                <label className="block text-sm font-medium mb-1">
                  Spectral Radius{" "}
                  <span className="text-gray-500 cursor-pointer">ⓘ</span>
                </label>
                <input
                  type="text"
                  name="spectralRadius"
                  value={formData.spectralRadius}
                  onChange={handleChange}
                  placeholder="0.9"
                  className="w-full p-2 border border-gray-300 rounded-md"
                />
              </div>
              <div className="flex-1">
                <label className="block text-sm font-medium mb-1">
                  Dropout{" "}
                  <span className="text-gray-500 cursor-pointer">ⓘ</span>
                </label>
                <input
                  type="text"
                  name="dropout"
                  value={formData.dropout}
                  onChange={handleChange}
                  placeholder="1"
                  className="w-full p-2 border border-gray-300 rounded-md"
                />
              </div>
            </div>

            {/* Third Row */}
            <div className="flex space-x-4">
              <div className="flex-1">
                <label className="block text-sm font-medium mb-1">
                  Sparsity{" "}
                  <span className="text-gray-500 cursor-pointer">ⓘ</span>
                </label>
                <input
                  type="text"
                  name="sparsity"
                  value={formData.sparsity}
                  onChange={handleChange}
                  placeholder="1.0"
                  className="w-full p-2 border border-gray-300 rounded-md"
                />
              </div>
              <div className="flex-1">
                <label className="block text-sm font-medium mb-1">
                  Lags <span className="text-gray-500 cursor-pointer">ⓘ</span>
                </label>
                <input
                  type="text"
                  name="lags"
                  value={formData.lags}
                  onChange={handleChange}
                  placeholder="1"
                  className="w-full p-2 border border-gray-300 rounded-md"
                />
              </div>
            </div>

            {/* Fourth Row */}
            <div className="flex space-x-4">
              <div className="flex-1">
                <label className="block text-sm font-medium mb-1">
                  Input Scaling{" "}
                  <span className="text-gray-500 cursor-pointer">ⓘ</span>
                </label>
                <input
                  type="text"
                  name="inputScaling"
                  value={formData.inputScaling}
                  onChange={handleChange}
                  placeholder="0.3"
                  className="w-full p-2 border border-gray-300 rounded-md"
                />
              </div>
              <div className="flex-1">{/* Empty div for alignment */}</div>
            </div>

            {/* Buttons */}
            <div className="flex justify-end space-x-4 mt-6">
              <button
                type="button"
                onClick={handleCancel}
                className="px-4 py-2 bg-red-500 text-white rounded-md hover:bg-red-600">
                Back
              </button>
              <button
                type="submit"
                className="px-4 py-2 bg-green-500 text-white rounded-md hover:bg-green-600">
                Submit
              </button>
            </div>
          </form>
        </div>
      );
    }

    // Return DHR form (existing form)
    return (
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* First Row: Fourier Order and Window Length */}
        <div className="flex space-x-4">
          <div className="flex-1">
            <label className="block text-sm font-medium mb-1">
              Fourier Order{" "}
              <span className="text-gray-500 cursor-pointer">ⓘ</span>
            </label>
            <input
              type="text"
              name="fourierOrder"
              value={formData.fourierOrder}
              onChange={handleChange}
              placeholder="3"
              className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div className="flex-1">
            <label className="block text-sm font-medium mb-1">
              Window Length{" "}
              <span className="text-gray-500 cursor-pointer">ⓘ</span>
            </label>
            <input
              type="text"
              name="windowLength"
              value={formData.windowLength}
              onChange={handleChange}
              placeholder="1"
              className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>

        {/* Second Row: Seasonality Periods and Polyorder */}
        <div className="flex space-x-4">
          <div className="flex-1">
            <label className="block text-sm font-medium mb-1">
              Seasonality Periods{" "}
              <span className="text-gray-500 cursor-pointer">ⓘ</span>
            </label>
            <input
              type="text"
              name="seasonalityPeriods"
              value={formData.seasonalityPeriods}
              onChange={handleChange}
              placeholder="M"
              className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div className="flex-1">
            <label className="block text-sm font-medium mb-1">
              Polyorder <span className="text-gray-500 cursor-pointer">ⓘ</span>
            </label>
            <input
              type="text"
              name="polyorder"
              value={formData.polyorder}
              onChange={handleChange}
              placeholder="0.1"
              className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>

        {/* Third Row: Regularization and Trend Components */}
        <div className="flex space-x-4">
          <div className="flex-1">
            <label className="block text-sm font-medium mb-1">
              Regularization{" "}
              <span className="text-gray-500 cursor-pointer">ⓘ</span>
            </label>
            <input
              type="text"
              name="regularization"
              value={formData.regularization}
              onChange={handleChange}
              placeholder="1e-4"
              className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div className="flex-1">
            <label className="block text-sm font-medium mb-1">
              Trend Components{" "}
              <span className="text-gray-500 cursor-pointer">ⓘ</span>
            </label>
            <input
              type="text"
              name="trendComponents"
              value={formData.trendComponents}
              onChange={handleChange}
              placeholder="2"
              className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>

        {/* Buttons */}
        <div className="flex justify-end space-x-4 mt-6">
          <button
            type="button"
            onClick={handleCancel}
            className="px-4 py-2 bg-red-500 text-white rounded-md hover:bg-red-600 focus:outline-none focus:ring-2 focus:ring-red-500">
            Cancel
          </button>
          <button
            type="submit"
            className="px-4 py-2 bg-green-500 text-white rounded-md hover:bg-green-600 focus:outline-none focus:ring-2 focus:ring-green-500">
            Submit
          </button>
        </div>
      </form>
    );
  };

  return (
    <div className="max-w-md mx-auto p-6 bg-gray-50 rounded-lg shadow-md">
      <h2 className="text-xl font-bold text-center mb-6">{getModelTitle()}</h2>
      {renderForm()}
    </div>
  );
};

export default SingleModelConfiguration;

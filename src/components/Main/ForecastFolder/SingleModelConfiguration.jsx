import React, { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";

const SingleModelConfiguration = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const selectedModel = location.state?.model;
  const forecastId = location.state?.forecastId;
  const isEditing = location.state?.isEditing;
  const existingConfig = location.state?.existingConfig;

  // Redirect if no model is selected
  React.useEffect(() => {
    if (!location.state || !selectedModel) {
      navigate("/generate");
    }
  }, [location.state, selectedModel, navigate]);

  // Add validation for forecastId
  useEffect(() => {
    if (!forecastId) {
      console.error("No forecast ID provided");
      navigate(-1);
    }
  }, [forecastId, navigate]);

  // Initialize form data based on selected model and existing config
  const getInitialFormData = () => {
    if (selectedModel === "DHR") {
      return isEditing
        ? {
            fourierOrder: existingConfig.fourier_order.toString(),
            windowLength: existingConfig.window_length.toString(),
            seasonalityPeriods: existingConfig.seasonality_periods,
            polyorder: existingConfig.polyorder.toString(),
            regularization: existingConfig.regularization_dhr.toString(),
            trendComponents: existingConfig.trend_components.toString(),
          }
        : {
            fourierOrder: "",
            windowLength: "",
            seasonalityPeriods: "",
            polyorder: "",
            regularization: "",
            trendComponents: "",
          };
    } else if (selectedModel === "ESN") {
      return isEditing
        ? {
            reservoirSize: existingConfig.reservoir_size.toString(),
            spectralRadius: existingConfig.spectral_radius.toString(),
            sparsity: existingConfig.sparsity.toString(),
            inputScaling: existingConfig.input_scaling.toString(),
            regularization: existingConfig.regularization_esn.toString(),
            dropout: existingConfig.dropout.toString(),
            lags: existingConfig.lags.toString(),
          }
        : {
            reservoirSize: "",
            spectralRadius: "",
            sparsity: "",
            inputScaling: "",
            regularization: "",
            dropout: "",
            lags: "",
          };
    }
    return {};
  };

  // State to manage form inputs with only relevant fields
  const [formData, setFormData] = useState(getInitialFormData());

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

  // Handle input changes
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: value,
    }));
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      if (selectedModel === "DHR") {
        const dhrConfig = {
          forecast_id: parseInt(forecastId),
          fourier_order: parseInt(formData.fourierOrder),
          window_length: parseInt(formData.windowLength),
          seasonality_periods: formData.seasonalityPeriods,
          polyorder: parseFloat(formData.polyorder),
          regularization_dhr: parseFloat(formData.regularization),
          trend_components: parseInt(formData.trendComponents),
        };

        // Remove the forecast_id from PUT URL path if isEditing
        const endpoint = isEditing
          ? `http://localhost:8000/api/dhr-configurations/${forecastId}`
          : "http://localhost:8000/api/dhr-configurations";

        const response = await fetch(endpoint, {
          method: isEditing ? "PUT" : "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(dhrConfig),
        });

        if (!response.ok) {
          // Get the error details from the response
          const errorData = await response.json();
          console.error("Server error details:", errorData);
          throw new Error(
            `Failed to save DHR configuration: ${errorData.detail}`
          );
        }

        const result = await response.json();

        // Navigate to forecast result page instead of going back
        navigate("/ForecastResult", {
          state: {
            forecastId,
          },
        });
      } else if (selectedModel === "ESN") {
        const esnConfig = {
          forecast_id: parseInt(forecastId),
          reservoir_size: parseInt(formData.reservoirSize),
          spectral_radius: parseFloat(formData.spectralRadius),
          sparsity: parseFloat(formData.sparsity),
          input_scaling: parseFloat(formData.inputScaling),
          dropout: parseFloat(formData.dropout),
          lags: parseInt(formData.lags),
          regularization_esn: parseFloat(formData.regularization),
        };

        // Remove the forecast_id from PUT URL path if isEditing
        const endpoint = isEditing
          ? `http://localhost:8000/api/esn-configurations/${forecastId}`
          : "http://localhost:8000/api/esn-configurations";

        const response = await fetch(endpoint, {
          method: isEditing ? "PUT" : "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(esnConfig),
        });

        if (!response.ok) {
          throw new Error("Failed to save ESN configuration");
        }

        // Navigate to forecast result page instead of going back
        navigate("/ForecastResult", {
          state: {
            forecastId,
          },
        });
      }
    } catch (error) {
      console.error("Error saving configuration:", error);
      // You might want to show an error message to the user
    }
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

import React, { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";

const SingleModelConfiguration = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const {
    model: selectedModel,
    forecastId,
    isEditing,
    existingConfig,
  } = location.state || {};

  const [errors, setErrors] = useState({});
  const [tooltipVisible, setTooltipVisible] = useState(null);

  // Redirect if no model is selected
  useEffect(() => {
    if (!selectedModel) {
      navigate("/generate");
    }
  }, [selectedModel, navigate]);

  // Validate forecastId
  useEffect(() => {
    if (!forecastId) {
      console.error("No forecast ID provided");
      navigate(-1);
    }
  }, [forecastId, navigate]);

  // Initialize form data based on selected model and existing config
  const getInitialFormData = () => {
    if (!selectedModel) return {};

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

  const [formData, setFormData] = useState(getInitialFormData());

  const getModelTitle = () => {
    switch (selectedModel) {
      case "DHR":
        return "DYNAMIC HARMONIC REGRESSION";
      case "ESN":
        return "ECHO STATE NETWORK";
      default:
        return "MODEL CONFIGURATION";
    }
  };

  const validateField = (name, value) => {
    if (!value.trim()) return null; // Skip validation for empty fields

    if (name === "sparsity") {
      const numValue = parseFloat(value);
      if (isNaN(numValue) || numValue < 0 || numValue > 1) {
        return "Sparsity must be between 0 and 1";
      }
    }

    // Validate numeric fields
    if (
      [
        "reservoirSize",
        "spectralRadius",
        "inputScaling",
        "regularization",
        "dropout",
        "lags",
        "fourierOrder",
        "windowLength",
        "polyorder",
        "trendComponents",
      ].includes(name)
    ) {
      const numValue = parseFloat(value);
      if (isNaN(numValue)) {
        return `${
          name.charAt(0).toUpperCase() + name.slice(1)
        } must be a number`;
      }

      // Additional validation for specific fields
      if (
        name === "reservoirSize" &&
        (numValue <= 0 || !Number.isInteger(Number(value)))
      ) {
        return "Reservoir Size must be a positive integer";
      }
      if (
        name === "lags" &&
        (numValue <= 0 || !Number.isInteger(Number(value)))
      ) {
        return "Lags must be a positive integer";
      }
      if (
        name === "fourierOrder" &&
        (numValue <= 0 || !Number.isInteger(Number(value)))
      ) {
        return "Fourier Order must be a positive integer";
      }
      if (
        name === "windowLength" &&
        (numValue <= 0 || !Number.isInteger(Number(value)))
      ) {
        return "Window Length must be a positive integer";
      }
      if (
        name === "trendComponents" &&
        (numValue <= 0 || !Number.isInteger(Number(value)))
      ) {
        return "Trend Components must be a positive integer";
      }
      if (name === "spectralRadius" && numValue <= 0) {
        return "Spectral Radius must be positive";
      }
      if (name === "inputScaling" && numValue <= 0) {
        return "Input Scaling must be positive";
      }
      if (name === "dropout" && (numValue < 0 || numValue > 1)) {
        return "Dropout must be between 0 and 1";
      }
      if ((name === "regularization" || name === "polyorder") && numValue < 0) {
        return `${
          name.charAt(0).toUpperCase() + name.slice(1)
        } must be non-negative`;
      }
    }

    return null;
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: value,
    }));

    // Validate field
    const error = validateField(name, value);
    setErrors((prev) => ({
      ...prev,
      [name]: error,
    }));
  };

  const showTooltip = (field) => {
    setTooltipVisible(field);
  };

  const hideTooltip = () => {
    setTooltipVisible(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    // Validate all fields before submitting
    const newErrors = {};
    let hasErrors = false;

    // Determine which fields to validate based on the selected model
    const fieldsToValidate =
      selectedModel === "DHR"
        ? [
            "fourierOrder",
            "windowLength",
            "seasonalityPeriods",
            "polyorder",
            "regularization",
            "trendComponents",
          ]
        : [
            "reservoirSize",
            "spectralRadius",
            "sparsity",
            "inputScaling",
            "regularization",
            "dropout",
            "lags",
          ];

    // Validate each field
    fieldsToValidate.forEach((field) => {
      const error = validateField(field, formData[field] || "");
      if (error) {
        newErrors[field] = error;
        hasErrors = true;
      }
    });

    setErrors(newErrors);
    if (hasErrors) {
      return;
    }

    try {
      let config, endpoint;

      if (selectedModel === "DHR") {
        config = {
          forecast_id: parseInt(forecastId),
          fourier_order: parseInt(formData.fourierOrder),
          window_length: parseInt(formData.windowLength),
          seasonality_periods: formData.seasonalityPeriods,
          polyorder: parseFloat(formData.polyorder),
          regularization_dhr: parseFloat(formData.regularization),
          trend_components: parseInt(formData.trendComponents),
        };

        endpoint = isEditing
          ? `http://localhost:8000/api/dhr-configurations/${forecastId}`
          : "http://localhost:8000/api/dhr-configurations";
      } else if (selectedModel === "ESN") {
        config = {
          forecast_id: parseInt(forecastId),
          reservoir_size: parseInt(formData.reservoirSize),
          spectral_radius: parseFloat(formData.spectralRadius),
          sparsity: parseFloat(formData.sparsity),
          input_scaling: parseFloat(formData.inputScaling),
          dropout: parseFloat(formData.dropout),
          lags: parseInt(formData.lags),
          regularization_esn: parseFloat(formData.regularization),
        };

        endpoint = isEditing
          ? `http://localhost:8000/api/esn-configurations/${forecastId}`
          : "http://localhost:8000/api/esn-configurations";
      }

      const response = await fetch(endpoint, {
        method: isEditing ? "PUT" : "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(config),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          `Failed to save configuration: ${errorData.detail || "Unknown error"}`
        );
      }

      navigate("/result", {
        state: { forecastId },
      });
    } catch (error) {
      console.error("Error saving configuration:", error);
      // Could add user-facing error message here
    }
  };

  const handleCancel = () => {
    navigate(-1);
  };

  const renderESNForm = () => (
    <div className="p-6 flex justify-center items-center">
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* First Row */}
        <div className="flex space-x-4">
          <div className="flex-1">
            <label className="block text-sm font-medium mb-1">
              Reservoir Size{" "}
              <span
                className="text-gray-500 cursor-pointer"
                onMouseEnter={() => showTooltip("reservoirSize")}
                onMouseLeave={hideTooltip}>
                ⓘ
              </span>
              {tooltipVisible === "reservoirSize" && (
                <div className="absolute bg-gray-800 text-white p-2 rounded text-xs max-w-xs">
                  Number of neurons in the reservoir. Typically between
                  100-1000.
                </div>
              )}
            </label>
            <input
              type="text"
              name="reservoirSize"
              value={formData.reservoirSize}
              onChange={handleChange}
              placeholder="500"
              className={`w-full p-2 border rounded-md ${
                errors.reservoirSize ? "border-red-500" : "border-gray-300"
              }`}
            />
            {errors.reservoirSize && (
              <p className="text-red-500 text-xs mt-1">
                {errors.reservoirSize}
              </p>
            )}
          </div>
          <div className="flex-1">
            <label className="block text-sm font-medium mb-1">
              Regularization{" "}
              <span
                className="text-gray-500 cursor-pointer"
                onMouseEnter={() => showTooltip("regularization")}
                onMouseLeave={hideTooltip}>
                ⓘ
              </span>
              {tooltipVisible === "regularization" && (
                <div className="absolute bg-gray-800 text-white p-2 rounded text-xs max-w-xs">
                  Regularization parameter for ESN training. Helps prevent
                  overfitting.
                </div>
              )}
            </label>
            <input
              type="text"
              name="regularization"
              value={formData.regularization}
              onChange={handleChange}
              placeholder="0.2"
              className={`w-full p-2 border rounded-md ${
                errors.regularization ? "border-red-500" : "border-gray-300"
              }`}
            />
            {errors.regularization && (
              <p className="text-red-500 text-xs mt-1">
                {errors.regularization}
              </p>
            )}
          </div>
        </div>

        {/* Second Row */}
        <div className="flex space-x-4">
          <div className="flex-1">
            <label className="block text-sm font-medium mb-1">
              Spectral Radius{" "}
              <span
                className="text-gray-500 cursor-pointer"
                onMouseEnter={() => showTooltip("spectralRadius")}
                onMouseLeave={hideTooltip}>
                ⓘ
              </span>
              {tooltipVisible === "spectralRadius" && (
                <div className="absolute bg-gray-800 text-white p-2 rounded text-xs max-w-xs">
                  Controls the echo state property. Usually set below 1.
                </div>
              )}
            </label>
            <input
              type="text"
              name="spectralRadius"
              value={formData.spectralRadius}
              onChange={handleChange}
              placeholder="0.9"
              className={`w-full p-2 border rounded-md ${
                errors.spectralRadius ? "border-red-500" : "border-gray-300"
              }`}
            />
            {errors.spectralRadius && (
              <p className="text-red-500 text-xs mt-1">
                {errors.spectralRadius}
              </p>
            )}
          </div>
          <div className="flex-1">
            <label className="block text-sm font-medium mb-1">
              Dropout{" "}
              <span
                className="text-gray-500 cursor-pointer"
                onMouseEnter={() => showTooltip("dropout")}
                onMouseLeave={hideTooltip}>
                ⓘ
              </span>
              {tooltipVisible === "dropout" && (
                <div className="absolute bg-gray-800 text-white p-2 rounded text-xs max-w-xs">
                  Dropout rate for improving robustness.
                </div>
              )}
            </label>
            <input
              type="text"
              name="dropout"
              value={formData.dropout}
              onChange={handleChange}
              placeholder="1"
              className={`w-full p-2 border rounded-md ${
                errors.dropout ? "border-red-500" : "border-gray-300"
              }`}
            />
            {errors.dropout && (
              <p className="text-red-500 text-xs mt-1">{errors.dropout}</p>
            )}
          </div>
        </div>

        {/* Third Row */}
        <div className="flex space-x-4">
          <div className="flex-1 relative">
            <label className="block text-sm font-medium mb-1">
              Sparsity{" "}
              <span
                className="text-gray-500 cursor-pointer"
                onMouseEnter={() => showTooltip("sparsity")}
                onMouseLeave={hideTooltip}>
                ⓘ
              </span>
              {tooltipVisible === "sparsity" && (
                <div className="absolute bg-gray-800 text-white p-2 rounded text-xs max-w-xs z-10">
                  Connectivity of the reservoir. Must be between 0 and 1.
                </div>
              )}
            </label>
            <input
              type="text"
              name="sparsity"
              value={formData.sparsity}
              onChange={handleChange}
              placeholder="0.1"
              className={`w-full p-2 border rounded-md ${
                errors.sparsity ? "border-red-500" : "border-gray-300"
              }`}
            />
            {errors.sparsity && (
              <p className="text-red-500 text-xs mt-1">{errors.sparsity}</p>
            )}
          </div>
          <div className="flex-1">
            <label className="block text-sm font-medium mb-1">
              Lags{" "}
              <span
                className="text-gray-500 cursor-pointer"
                onMouseEnter={() => showTooltip("lags")}
                onMouseLeave={hideTooltip}>
                ⓘ
              </span>
              {tooltipVisible === "lags" && (
                <div className="absolute bg-gray-800 text-white p-2 rounded text-xs max-w-xs">
                  Number of time steps to use as input.
                </div>
              )}
            </label>
            <input
              type="text"
              name="lags"
              value={formData.lags}
              onChange={handleChange}
              placeholder="1"
              className={`w-full p-2 border rounded-md ${
                errors.lags ? "border-red-500" : "border-gray-300"
              }`}
            />
            {errors.lags && (
              <p className="text-red-500 text-xs mt-1">{errors.lags}</p>
            )}
          </div>
        </div>

        {/* Fourth Row */}
        <div className="flex space-x-4">
          <div className="flex-1 relative">
            <label className="block text-sm font-medium mb-1">
              Input Scaling{" "}
              <span
                className="text-gray-500 cursor-pointer"
                onMouseEnter={() => showTooltip("inputScaling")}
                onMouseLeave={hideTooltip}>
                ⓘ
              </span>
              {tooltipVisible === "inputScaling" && (
                <div className="absolute bg-gray-800 text-white p-2 rounded text-xs max-w-xs">
                  Scaling factor applied to the input data.
                </div>
              )}
            </label>
            <input
              type="text"
              name="inputScaling"
              value={formData.inputScaling}
              onChange={handleChange}
              placeholder="0.3"
              className={`w-full p-2 border rounded-md ${
                errors.inputScaling ? "border-red-500" : "border-gray-300"
              }`}
            />
            {errors.inputScaling && (
              <p className="text-red-500 text-xs mt-1">{errors.inputScaling}</p>
            )}
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

  const renderDHRForm = () => (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* First Row: Fourier Order and Window Length */}
      <div className="flex space-x-4">
        <div className="flex-1 relative">
          <label className="block text-sm font-medium mb-1">
            Fourier Order{" "}
            <span
              className="text-gray-500 cursor-pointer"
              onMouseEnter={() => showTooltip("fourierOrder")}
              onMouseLeave={hideTooltip}>
              ⓘ
            </span>
            {tooltipVisible === "fourierOrder" && (
              <div className="absolute bg-gray-800 text-white p-2 rounded text-xs max-w-xs">
                Number of sine and cosine terms to include in the model.
              </div>
            )}
          </label>
          <input
            type="text"
            name="fourierOrder"
            value={formData.fourierOrder}
            onChange={handleChange}
            placeholder="3"
            className={`w-full p-2 border rounded-md ${
              errors.fourierOrder ? "border-red-500" : "border-gray-300"
            } focus:outline-none focus:ring-2 focus:ring-blue-500`}
          />
          {errors.fourierOrder && (
            <p className="text-red-500 text-xs mt-1">{errors.fourierOrder}</p>
          )}
        </div>
        <div className="flex-1 relative">
          <label className="block text-sm font-medium mb-1">
            Window Length{" "}
            <span
              className="text-gray-500 cursor-pointer"
              onMouseEnter={() => showTooltip("windowLength")}
              onMouseLeave={hideTooltip}>
              ⓘ
            </span>
            {tooltipVisible === "windowLength" && (
              <div className="absolute bg-gray-800 text-white p-2 rounded text-xs max-w-xs">
                Window size for the Savitzky-Golay filter.
              </div>
            )}
          </label>
          <input
            type="text"
            name="windowLength"
            value={formData.windowLength}
            onChange={handleChange}
            placeholder="1"
            className={`w-full p-2 border rounded-md ${
              errors.windowLength ? "border-red-500" : "border-gray-300"
            } focus:outline-none focus:ring-2 focus:ring-blue-500`}
          />
          {errors.windowLength && (
            <p className="text-red-500 text-xs mt-1">{errors.windowLength}</p>
          )}
        </div>
      </div>

      {/* Second Row: Seasonality Periods and Polyorder */}
      <div className="flex space-x-4">
        <div className="flex-1 relative">
          <label className="block text-sm font-medium mb-1">
            Seasonality Periods{" "}
            <span
              className="text-gray-500 cursor-pointer"
              onMouseEnter={() => showTooltip("seasonalityPeriods")}
              onMouseLeave={hideTooltip}>
              ⓘ
            </span>
            {tooltipVisible === "seasonalityPeriods" && (
              <div className="absolute bg-gray-800 text-white p-2 rounded text-xs max-w-xs">
                Seasonality components to include (e.g., D=daily, W=weekly,
                M=monthly, Q=quarterly, Y=yearly).
              </div>
            )}
          </label>
          <input
            type="text"
            name="seasonalityPeriods"
            value={formData.seasonalityPeriods}
            onChange={handleChange}
            placeholder="M"
            className={`w-full p-2 border rounded-md ${
              errors.seasonalityPeriods ? "border-red-500" : "border-gray-300"
            } focus:outline-none focus:ring-2 focus:ring-blue-500`}
          />
          {errors.seasonalityPeriods && (
            <p className="text-red-500 text-xs mt-1">
              {errors.seasonalityPeriods}
            </p>
          )}
        </div>
        <div className="flex-1 relative">
          <label className="block text-sm font-medium mb-1">
            Polyorder{" "}
            <span
              className="text-gray-500 cursor-pointer"
              onMouseEnter={() => showTooltip("polyorder")}
              onMouseLeave={hideTooltip}>
              ⓘ
            </span>
            {tooltipVisible === "polyorder" && (
              <div className="absolute bg-gray-800 text-white p-2 rounded text-xs max-w-xs">
                Polynomial order for the Savitzky-Golay filter.
              </div>
            )}
          </label>
          <input
            type="text"
            name="polyorder"
            value={formData.polyorder}
            onChange={handleChange}
            placeholder="0.1"
            className={`w-full p-2 border rounded-md ${
              errors.polyorder ? "border-red-500" : "border-gray-300"
            } focus:outline-none focus:ring-2 focus:ring-blue-500`}
          />
          {errors.polyorder && (
            <p className="text-red-500 text-xs mt-1">{errors.polyorder}</p>
          )}
        </div>
      </div>

      {/* Third Row: Regularization and Trend Components */}
      <div className="flex space-x-4">
        <div className="flex-1 relative">
          <label className="block text-sm font-medium mb-1">
            Regularization{" "}
            <span
              className="text-gray-500 cursor-pointer"
              onMouseEnter={() => showTooltip("regularization")}
              onMouseLeave={hideTooltip}>
              ⓘ
            </span>
            {tooltipVisible === "regularization" && (
              <div className="absolute bg-gray-800 text-white p-2 rounded text-xs max-w-xs">
                Regularization parameter for the DHR model to prevent
                overfitting.
              </div>
            )}
          </label>
          <input
            type="text"
            name="regularization"
            value={formData.regularization}
            onChange={handleChange}
            placeholder="1e-4"
            className={`w-full p-2 border rounded-md ${
              errors.regularization ? "border-red-500" : "border-gray-300"
            } focus:outline-none focus:ring-2 focus:ring-blue-500`}
          />
          {errors.regularization && (
            <p className="text-red-500 text-xs mt-1">{errors.regularization}</p>
          )}
        </div>
        <div className="flex-1 relative">
          <label className="block text-sm font-medium mb-1">
            Trend Components{" "}
            <span
              className="text-gray-500 cursor-pointer"
              onMouseEnter={() => showTooltip("trendComponents")}
              onMouseLeave={hideTooltip}>
              ⓘ
            </span>
            {tooltipVisible === "trendComponents" && (
              <div className="absolute bg-gray-800 text-white p-2 rounded text-xs max-w-xs">
                Number of trend components to include in the model.
              </div>
            )}
          </label>
          <input
            type="text"
            name="trendComponents"
            value={formData.trendComponents}
            onChange={handleChange}
            placeholder="2"
            className={`w-full p-2 border rounded-md ${
              errors.trendComponents ? "border-red-500" : "border-gray-300"
            } focus:outline-none focus:ring-2 focus:ring-blue-500`}
          />
          {errors.trendComponents && (
            <p className="text-red-500 text-xs mt-1">
              {errors.trendComponents}
            </p>
          )}
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

  const renderForm = () => {
    if (selectedModel === "ESN") {
      return renderESNForm();
    }
    return renderDHRForm();
  };

  return (
    <div className="max-w-md mx-auto p-6 bg-gray-50 rounded-lg shadow-md">
      <h2 className="text-xl font-bold text-center mb-6">{getModelTitle()}</h2>
      {renderForm()}
    </div>
  );
};

export default SingleModelConfiguration;

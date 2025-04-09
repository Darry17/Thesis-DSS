import React, { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";

const HybridModelConfiguration = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { forecastId, isEditing, existingConfig, originalFileName } =
    location.state || {};
  const [step, setStep] = useState(1); // 1 for DHR, 2 for ESN
  const [errors, setErrors] = useState({});
  const [tooltipVisible, setTooltipVisible] = useState(null);

  const initialFormData = isEditing
    ? {
        // DHR fields
        fourierOrder: existingConfig.fourier_order.toString(),
        windowLength: existingConfig.window_length.toString(),
        seasonalityPeriods: existingConfig.seasonality_periods,
        polyorder: existingConfig.polyorder.toString(),
        regularizationDHR: existingConfig.regularization_dhr.toString(),
        trendComponents: existingConfig.trend_components.toString(),
        // ESN fields
        reservoirSize: existingConfig.reservoir_size.toString(),
        spectralRadius: existingConfig.spectral_radius.toString(),
        sparsity: existingConfig.sparsity.toString(),
        inputScaling: existingConfig.input_scaling.toString(),
        dropout: existingConfig.dropout.toString(),
        lags: existingConfig.lags.toString(),
        regularizationESN: existingConfig.regularization_esn.toString(),
      }
    : {
        // DHR fields
        fourierOrder: "",
        windowLength: "",
        seasonalityPeriods: "",
        polyorder: "",
        regularizationDHR: "",
        trendComponents: "",
        // ESN fields
        reservoirSize: "",
        spectralRadius: "",
        sparsity: "",
        inputScaling: "",
        dropout: "",
        lags: "",
        regularizationESN: "",
      };

  const [formData, setFormData] = useState(initialFormData);

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
        "regularizationDHR",
        "regularizationESN",
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
      if (
        (name === "regularizationDHR" ||
          name === "regularizationESN" ||
          name === "polyorder") &&
        numValue < 0
      ) {
        return `${
          name.charAt(0).toUpperCase() + name.slice(1)
        } must be non-negative`;
      }
    }

    return null;
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
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

  const handleSaveForecast = async () => {
    try {
      console.log("Saving forecast with ID:", forecastId);

      if (!forecastId) {
        console.error("Missing forecastId - cannot save history log");
        alert("Error: Missing forecast ID. Cannot save to history log.");
        return;
      }

      const token = localStorage.getItem("token")?.trim();
      if (!token) {
        console.error("No authentication token found. Redirecting to login.");
        navigate("/login");
        return;
      }

      const selectedModel = "DHR-ESN"; // Since this is the hybrid model configuration
      let modelType = selectedModel.toLowerCase();
      let fileModelType = modelType;
      let displayModelType = modelType.toUpperCase();

      if (modelType === "dhr-esn") {
        fileModelType = "hybrid";
        displayModelType = "DHR-ESN";
      }

      const today = new Date();
      const formattedDate = `${today.getFullYear()}-${String(
        today.getMonth() + 1
      ).padStart(2, "0")}-${String(today.getDate()).padStart(2, "0")}`;

      // Filename format: [original-filename]-[model]-[date]-[forecastId]
      const fileName = originalFileName
        ? `${originalFileName}-${fileModelType}-${formattedDate}-${forecastId}`
        : `${fileModelType}-${formattedDate}-${forecastId}`; // Fallback if originalFileName is missing

      console.log("Creating history log with metrics:", {
        file_name: fileName,
        model: displayModelType,
        forecast_id: forecastId,
      });

      const historyLogResponse = await fetch(
        "http://localhost:8000/api/history-logs",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({
            file_name: fileName,
            model: displayModelType,
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

      const responseData = await historyLogResponse.json();
      console.log("History log created successfully:", responseData);
    } catch (error) {
      console.error("Error saving forecast:", error);
      alert(`Error saving forecast: ${error.message}`);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (step === 1) {
      // Validate DHR fields before moving to step 2
      const newErrors = {};
      let hasErrors = false;

      const dhrFields = [
        "fourierOrder",
        "windowLength",
        "seasonalityPeriods",
        "polyorder",
        "regularizationDHR",
        "trendComponents",
      ];

      dhrFields.forEach((field) => {
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

      setStep(2); // Move to ESN configuration
      return;
    }

    // Validate all ESN fields before submitting
    const newErrors = {};
    let hasErrors = false;

    const esnFields = [
      "reservoirSize",
      "spectralRadius",
      "sparsity",
      "inputScaling",
      "dropout",
      "lags",
      "regularizationESN",
    ];

    esnFields.forEach((field) => {
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
      // Fetch the token from localStorage
      const token = localStorage.getItem("token")?.trim();
      if (!token) {
        console.error("No authentication token found. Redirecting to login.");
        navigate("/login");
        return;
      }

      const hybridConfig = {
        forecast_id: parseInt(forecastId),
        // DHR part
        fourier_order: parseInt(formData.fourierOrder),
        window_length: parseInt(formData.windowLength),
        seasonality_periods: formData.seasonalityPeriods,
        polyorder: parseFloat(formData.polyorder),
        regularization_dhr: parseFloat(formData.regularizationDHR),
        trend_components: parseInt(formData.trendComponents),
        // ESN part
        reservoir_size: parseInt(formData.reservoirSize),
        spectral_radius: parseFloat(formData.spectralRadius),
        sparsity: parseFloat(formData.sparsity),
        input_scaling: parseFloat(formData.inputScaling),
        dropout: parseFloat(formData.dropout),
        lags: parseInt(formData.lags),
        regularization_esn: parseFloat(formData.regularizationESN),
      };

      const endpoint = isEditing
        ? `http://localhost:8000/api/hybrid-configurations/${forecastId}`
        : "http://localhost:8000/api/hybrid-configurations";

      const response = await fetch(endpoint, {
        method: isEditing ? "PUT" : "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`, // Add the Authorization header
        },
        body: JSON.stringify(hybridConfig),
      });

      if (!response.ok) {
        if (response.status === 401) {
          console.error("Session expired. Redirecting to login.");
          localStorage.removeItem("token");
          navigate("/login");
          return;
        }
        const errorData = await response.json();
        throw new Error(
          `Failed to ${isEditing ? "update" : "save"} hybrid configuration: ${
            errorData.detail
          }`
        );
      }

      // Save the forecast to history logs
      await handleSaveForecast();

      navigate("/result", {
        state: { forecastId },
      });
    } catch (error) {
      console.error("Error saving configuration:", error);
      alert(`Error: ${error.message}`);
    }
  };

  const handleBack = () => {
    if (step === 2) {
      setStep(1); // Go back to DHR configuration
    } else {
      navigate(-1); // Go back to previous page
    }
  };

  const renderFormField = (name, label, placeholder, tooltip = null) => {
    const hasError = errors[name];

    return (
      <div className="flex-1 relative">
        <label className="block text-sm font-medium mb-1">
          {label}{" "}
          <span
            className="text-gray-500 cursor-pointer"
            onMouseEnter={() => showTooltip(name)}
            onMouseLeave={hideTooltip}>
            ⓘ
          </span>
          {tooltipVisible === name && tooltip && (
            <div className="absolute bg-gray-800 text-white p-2 rounded text-xs max-w-xs z-10">
              {tooltip}
            </div>
          )}
        </label>
        <input
          type="text"
          name={name}
          value={formData[name]}
          onChange={handleChange}
          placeholder={placeholder}
          className={`w-full p-2 border rounded-md ${
            hasError ? "border-red-500" : "border-gray-300"
          }`}
        />
        {hasError && <p className="text-red-500 text-xs mt-1">{hasError}</p>}
      </div>
    );
  };

  const renderDHRForm = () => (
    <form onSubmit={handleSubmit} className="space-y-4">
      <h2 className="text-4xl font-bold text-left mb-10">
        Dynamic Harmonic Regression
      </h2>

      <div className="flex space-x-4">
        {renderFormField(
          "fourierOrder",
          "Fourier Order",
          "3",
          "Number of sine and cosine terms to include in the model."
        )}
        {renderFormField(
          "windowLength",
          "Window Length",
          "1",
          "Window size for the Savitzky-Golay filter."
        )}
      </div>

      <div className="flex space-x-4">
        {renderFormField(
          "seasonalityPeriods",
          "Seasonality Periods",
          "M",
          "Seasonality components to include (e.g., D=daily, W=weekly, M=monthly, Q=quarterly, Y=yearly)."
        )}
        {renderFormField(
          "polyorder",
          "Polyorder",
          "0.1",
          "Polynomial order for the Savitzky-Golay filter."
        )}
      </div>

      <div className="flex space-x-4 w-58">
        {renderFormField(
          "regularizationDHR",
          "Regularization (DHR)",
          "1e-4",
          "Regularization parameter for the DHR model to prevent overfitting."
        )}
      </div>

      <div className="flex space-x-4 w-58">
        {renderFormField(
          "trendComponents",
          "Trend Components",
          "2",
          "Number of trend components to include in the model."
        )}
      </div>

      <div className="flex justify-end space-x-4 mt-10">
        <button
          type="button"
          onClick={handleBack}
          className="px-4 py-2 bg-red-500 text-white rounded-md hover:bg-red-600">
          Back
        </button>
        <button
          type="submit"
          className="px-4 py-2 bg-green-700 text-white rounded-md hover:bg-green-600">
          Next
        </button>
      </div>
    </form>
  );

  const renderESNForm = () => (
    <form onSubmit={handleSubmit} className="space-y-4">
      <h2 className="text-4xl font-bold text-left mb-10">Echo State Network</h2>

      <div className="flex space-x-4">
        {renderFormField(
          "reservoirSize",
          "Reservoir Size",
          "500",
          "Number of neurons in the reservoir. Typically between 100-1000."
        )}
        {renderFormField(
          "regularizationESN",
          "Regularization (ESN)",
          "0.2",
          "Regularization parameter for ESN training. Helps prevent overfitting."
        )}
      </div>

      <div className="flex space-x-4">
        {renderFormField(
          "spectralRadius",
          "Spectral Radius",
          "0.9",
          "Controls the echo state property. Usually set below 1."
        )}
        {renderFormField(
          "dropout",
          "Dropout",
          "1",
          "Dropout rate for improving robustness."
        )}
      </div>

      <div className="flex space-x-4">
        {renderFormField(
          "sparsity",
          "Sparsity",
          "0.1",
          "Connectivity of the reservoir. Must be between 0 and 1."
        )}
        {renderFormField(
          "lags",
          "Lags",
          "1",
          "Number of time steps to use as input."
        )}
      </div>

      <div className="flex space-x-4">
        {renderFormField(
          "inputScaling",
          "Input Scaling",
          "0.3",
          "Scaling factor applied to the input data."
        )}
        <div className="flex-1">{/* Empty div for alignment */}</div>
      </div>

      <div className="flex justify-end space-x-4 mt-10">
        <button
          type="button"
          onClick={handleBack}
          className="px-4 py-2 bg-red-500 text-white rounded-md hover:bg-red-600">
          Back
        </button>
        <button
          type="submit"
          className="px-4 py-2 bg-green-700 text-white rounded-md hover:bg-green-600">
          Submit
        </button>
      </div>
    </form>
  );

  return (
    <div className="min-h-screen relative">
      {/* Background Layer */}
      <div className="fixed inset-0 bg-gray-100" style={{ zIndex: -1 }} />
      {/* Content */}
      <div className="relative z-10 flex justify-center items-center flex-1 min-h-screen">
        <div className="w-150 h-150 p-10 px-15 bg-white rounded-lg shadow-md">
          {step === 1 ? renderDHRForm() : renderESNForm()}
        </div>
      </div>
    </div>
  );
};

export default HybridModelConfiguration;

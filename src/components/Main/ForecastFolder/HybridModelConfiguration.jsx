import React, { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";

const HybridModelConfiguration = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { forecastId, isEditing, existingConfig } = location.state || {};
  const [step, setStep] = useState(1); // 1 for DHR, 2 for ESN

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

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (step === 1) {
      setStep(2); // Move to ESN configuration
      return;
    }

    try {
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
        },
        body: JSON.stringify(hybridConfig),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          `Failed to ${isEditing ? "update" : "save"} hybrid configuration: ${
            errorData.detail
          }`
        );
      }

      navigate("/result", {
        state: { forecastId },
      });
    } catch (error) {
      console.error("Error saving configuration:", error);
    }
  };

  const handleBack = () => {
    if (step === 2) {
      setStep(1); // Go back to DHR configuration
    } else {
      navigate(-1); // Go back to previous page
    }
  };

  const renderFormField = (name, label, placeholder) => (
    <div className="flex-1">
      <label className="block text-sm font-medium mb-1">
        {label} <span className="text-gray-500 cursor-pointer">ⓘ</span>
      </label>
      <input
        type="text"
        name={name}
        value={formData[name]}
        onChange={handleChange}
        placeholder={placeholder}
        className="w-full p-2 border border-gray-300 rounded-md"
      />
    </div>
  );

  const renderDHRForm = () => (
    <form onSubmit={handleSubmit} className="space-y-4">
      <h2 className="text-xl font-bold text-center mb-6">
        DYNAMIC HARMONIC REGRESSION
      </h2>

      <div className="flex space-x-4">
        {renderFormField("fourierOrder", "Fourier Order", "3")}
        {renderFormField("windowLength", "Window Length", "1")}
      </div>

      <div className="flex space-x-4">
        {renderFormField("seasonalityPeriods", "Seasonality Periods", "M")}
        {renderFormField("polyorder", "Polyorder", "0.1")}
      </div>

      <div className="flex space-x-4">
        {renderFormField("regularizationDHR", "Regularization (DHR)", "1e-4")}
        {renderFormField("trendComponents", "Trend Components", "2")}
      </div>

      <div className="flex justify-end space-x-4 mt-6">
        <button
          type="button"
          onClick={handleBack}
          className="px-4 py-2 bg-red-500 text-white rounded-md hover:bg-red-600">
          Back
        </button>
        <button
          type="submit"
          className="px-4 py-2 bg-green-500 text-white rounded-md hover:bg-green-600">
          Next
        </button>
      </div>
    </form>
  );

  const renderESNForm = () => (
    <form onSubmit={handleSubmit} className="space-y-4">
      <h2 className="text-xl font-bold text-center mb-6">ECHO STATE NETWORK</h2>

      <div className="flex space-x-4">
        {renderFormField("reservoirSize", "Reservoir Size", "500")}
        {renderFormField("regularizationESN", "Regularization (ESN)", "0.2")}
      </div>

      <div className="flex space-x-4">
        {renderFormField("spectralRadius", "Spectral Radius", "0.9")}
        {renderFormField("dropout", "Dropout", "1")}
      </div>

      <div className="flex space-x-4">
        {renderFormField("sparsity", "Sparsity", "1.0")}
        {renderFormField("lags", "Lags", "1")}
      </div>

      <div className="flex space-x-4">
        {renderFormField("inputScaling", "Input Scaling", "0.3")}
        <div className="flex-1">{/* Empty div for alignment */}</div>
      </div>

      <div className="flex justify-end space-x-4 mt-6">
        <button
          type="button"
          onClick={handleBack}
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
  );

  return (
    <div className="max-w-md mx-auto p-6 bg-gray-50 rounded-lg shadow-md">
      {step === 1 ? renderDHRForm() : renderESNForm()}
    </div>
  );
};

export default HybridModelConfiguration;

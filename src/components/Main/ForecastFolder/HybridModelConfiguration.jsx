import React, { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";

const HybridConfiguration = () => {
  const navigate = useNavigate();
  const [step, setStep] = useState(1); // 1 for DHR, 2 for ESN
  const [formData, setFormData] = useState({
    // DHR fields
    fourierOrder: "",
    windowLength: "",
    seasonalityPeriods: "",
    polyorder: "",
    regularization: "",
    trendComponents: "",
    // ESN fields
    reservoirSize: "",
    spectralRadius: "",
    sparsity: "",
    inputScaling: "",
    dropout: "",
    lags: "",
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (step === 1) {
      setStep(2); // Move to ESN configuration
    } else {
      console.log("Final form submitted:", formData);
      // Add your submission logic here
    }
  };

  const handleBack = () => {
    if (step === 2) {
      setStep(1); // Go back to DHR configuration
    } else {
      navigate(-1); // Go back to previous page
    }
  };

  const renderDHRForm = () => (
    <form onSubmit={handleSubmit} className="space-y-4">
      <h2 className="text-xl font-bold text-center mb-6">
        DYNAMIC HARMONIC REGRESSION
      </h2>
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
            className="w-full p-2 border border-gray-300 rounded-md"
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
            className="w-full p-2 border border-gray-300 rounded-md"
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
            className="w-full p-2 border border-gray-300 rounded-md"
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
            className="w-full p-2 border border-gray-300 rounded-md"
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
            className="w-full p-2 border border-gray-300 rounded-md"
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
            className="w-full p-2 border border-gray-300 rounded-md"
          />
        </div>
      </div>

      {/* Buttons */}
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
            Dropout <span className="text-gray-500 cursor-pointer">ⓘ</span>
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
            Sparsity <span className="text-gray-500 cursor-pointer">ⓘ</span>
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

export default HybridConfiguration;

import React, { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";

const GenerateForecast = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [fileData, setFileData] = useState(null);
  const [tooltipVisible, setTooltipVisible] = useState(null);

  const showTooltip = (field) => setTooltipVisible(field);
  const hideTooltip = () => setTooltipVisible(null);

  const modelType = location.state?.modelType || ""; // Get modelType from navigation state

  const getGranularityFromFilename = (filename) => {
    if (!filename) return "Hourly";
    if (filename.includes("hourly")) return "Hourly";
    if (filename.includes("daily")) return "Daily";
    if (filename.includes("weekly")) return "Weekly";
    return "Hourly";
  };

  const getStepOptions = (granularity) => {
    switch (granularity) {
      case "Hourly":
        return [
          { value: "1-hour", label: "1 Step (1-Hour Horizon)" },
          { value: "24-hour", label: "24 Steps (1-Day Horizon)" },
          { value: "168-hour", label: "168 Steps (1-Week Horizon)" },
        ];
      case "Daily":
        return [
          { value: "1-day", label: "1 Step (1-Day Horizon)" },
          { value: "7-day", label: "7 Steps (1-Week Horizon)" },
          { value: "30-day", label: "30 Steps (1-Month Horizon)" },
        ];
      case "Weekly":
        return [
          { value: "1-week", label: "1 Step (1-Week Horizon)" },
          { value: "4-week", label: "4 Steps (1-Month Horizon)" },
          { value: "52-week", label: "52 Steps (1-Year Horizon)" },
        ];
      default:
        return [];
    }
  };

  const initialGranularity = getGranularityFromFilename();
  const initialStepOptions = getStepOptions(initialGranularity);

  const [formData, setFormData] = useState({
    filename: null,
    original_filename: null,
    granularity: initialGranularity,
    steps: initialStepOptions.length ? initialStepOptions[0].value : "",
    modelType: "",
    model: "",
  });

  useEffect(() => {
    const stepOptions = getStepOptions(formData.granularity);
    if (stepOptions.length) {
      setFormData((prev) => ({
        ...prev,
        steps: stepOptions[0].value,
      }));
    }
  }, [formData.granularity]);

  useEffect(() => {
    const fetchFileData = async () => {
      try {
        setLoading(true);
        const urlParams = new URLSearchParams(location.search);
        const dataType = urlParams.get("type") || "hourly";

        const response = await fetch(
          `http://localhost:8000/storage/latest-file/?data_type=${dataType}`
        );

        if (!response.ok) {
          throw new Error(`Failed to fetch ${dataType} file`);
        }

        const latestFile = await response.json();

        setFileData({
          filename: latestFile.filename,
          original_filename: latestFile.original_filename,
          upload_date: latestFile.upload_date,
        });

        setFormData((prev) => ({
          ...prev,
          filename: latestFile.filename,
          original_filename: latestFile.original_filename,
          granularity: getGranularityFromFilename(latestFile.filename),
        }));
      } catch (err) {
        console.error("Error:", err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchFileData();
  }, [location.search]);

  const isFormValid = () => {
    return fileData && formData.modelType && formData.model;
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
      ...(name === "modelType" && { model: "" }),
    }));
  };

  const handleGenerate = async (e) => {
    e.preventDefault();
    if (!fileData) return;

    try {
      const fileResponse = await fetch(
        `http://localhost:8000/storage/read/${fileData.filename}`
      );

      if (!fileResponse.ok) {
        throw new Error("Failed to read file data");
      }

      const data = await fileResponse.json();

      const forecastData = {
        filename: fileData.filename,
        original_filename: fileData.original_filename,
        forecast_model: formData.model,
        steps: formData.steps,
        granularity: formData.granularity,
      };

      const createForecastResponse = await fetch(
        "http://localhost:8000/api/forecasts",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(forecastData),
        }
      );

      if (!createForecastResponse.ok) {
        const errorText = await createForecastResponse.text();
        throw new Error(`Failed to create forecast record: ${errorText}`);
      }

      const createdForecast = await createForecastResponse.json();

      navigate(
        formData.modelType === "Hybrid"
          ? "/hybrid-model-config"
          : "/single-model-config",
        {
          state: {
            ...formData,
            forecastData: data,
            forecastId: createdForecast.id,
            modelType,
            originalFileName: formData.original_filename,
          },
        }
      );
    } catch (err) {
      setError(err.message);
      console.error("Error:", err);
    }
  };

  if (loading) {
    return <div className="p-6">Loading...</div>;
  }

  if (error) {
    return <div className="p-6 text-red-500">Error: {error}</div>;
  }

  // Set background based on modelType
  const backgroundImage = modelType
    ? `url(/${modelType.toLowerCase()}-bg.png)`
    : "none";

  return (
    <div className="relative min-h-screen flex">
      <div
        className="fixed inset-0 overflow-hidden"
        style={{
          backgroundImage: backgroundImage,
          backgroundSize: "cover",
          backgroundPosition: "center",
          zIndex: -1,
        }}
      />
      <div className="fixed inset-0 bg-black/60" style={{ zIndex: -1 }} />
      <form
        onSubmit={handleGenerate}
        className="bg-gray-100 p-15 py-7 rounded-lg shadow-md w-150 m-auto">
        <h1 className="text-4xl font-bold mb-6">Generate</h1>

        {/* File Information */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mbIts-1">
            Current File
          </label>
          <div className="p-3 bg-gray-50 rounded-md">
            <p className="text-sm text-gray-600">
              {fileData?.original_filename ||
                fileData?.filename ||
                "No file selected"}
            </p>
            {fileData?.upload_date && (
              <p className="text-xs text-gray-500">
                Uploaded: {new Date(fileData.upload_date).toLocaleString()}
              </p>
            )}
          </div>
        </div>

        {/* Granularity - Read Only */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Granularity
            <span
              className="text-gray-500 cursor-pointer ml-2"
              onMouseEnter={() => showTooltip("granularity")}
              onMouseLeave={hideTooltip}>
              ⓘ
            </span>
            {tooltipVisible === "granularity" && (
              <div className="absolute bg-gray-800 text-white p-2 rounded text-xs max-w-xs">
                The resolution of the dataset and forecast, determines the
                smallest unit of time considered in the data
              </div>
            )}
          </label>
          <input
            type="text"
            value={formData.granularity}
            disabled
            className="block w-full p-2 border border-gray-300 rounded-md bg-gray-100 cursor-not-allowed"
          />
        </div>

        {/* Steps */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Steps
            <span
              className="text-gray-500 cursor-pointer ml-2"
              onMouseEnter={() => showTooltip("steps")}
              onMouseLeave={hideTooltip}>
              ⓘ
            </span>
            {tooltipVisible === "steps" && (
              <div className="absolute bg-gray-800 text-white p-2 rounded text-xs max-w-xs">
                The number of time periods into the future for which forecasts
                are made.
              </div>
            )}
          </label>
          <select
            name="steps"
            value={formData.steps}
            onChange={handleChange}
            className="block w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
            {getStepOptions(formData.granularity).map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>

        {/* Model Type (Single/Hybrid) */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Model Type
          </label>
          <div className="flex gap-4">
            <label className="flex items-center">
              <input
                type="radio"
                name="modelType"
                value="Single"
                checked={formData.modelType === "Single"}
                onChange={handleChange}
                className="mr-2"
              />
              Single
            </label>
            <label className="flex items-center">
              <input
                type="radio"
                name="modelType"
                value="Hybrid"
                checked={formData.modelType === "Hybrid"}
                onChange={handleChange}
                className="mr-2"
              />
              Hybrid
            </label>
          </div>
          {!formData.modelType && (
            <p className="text-sm text-red-500 mt-1">
              Please select a model type
            </p>
          )}
        </div>

        {/* Model Selection */}
        {formData.modelType && (
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Model
            </label>
            <select
              name="model"
              value={formData.model}
              onChange={handleChange}
              className="block w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
              <option value="">Select a model</option>
              {formData.modelType === "Single" ? (
                <>
                  <option value="DHR">DHR</option>
                  <option value="ESN">ESN</option>
                </>
              ) : (
                <option value="DHR-ESN">DHR-ESN</option>
              )}
            </select>
            {!formData.model && (
              <p className="text-sm text-red-500 mt-1">Please select a model</p>
            )}
          </div>
        )}

        {/* Buttons */}
        <div className="flex justify-end gap-3">
          <button
            type="button"
            onClick={() => navigate(-1)}
            className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600 transition-colors cursor-pointer">
            Cancel
          </button>
          <button
            type="submit"
            disabled={!isFormValid()}
            className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer">
            Submit
          </button>
        </div>
      </form>
    </div>
  );
};

export default GenerateForecast;

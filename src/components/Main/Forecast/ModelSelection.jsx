import React, { useState, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";

function useQuery() {
  return new URLSearchParams(useLocation().search);
}

const ModelSelection = () => {
  const navigate = useNavigate();
  const query = useQuery();

  const tempId = query.get("tempId");
  const tempFilename = query.get("tempFilename");
  const originalFileName = query.get("originalFileName");
  const forecastType = query.get("forecastType");

  const [energyDemand, setEnergyDemand] = useState("");
  const [maxCapacity, setMaxCapacity] = useState("");
  const [granularity, setGranularity] = useState("Hourly");
  const [steps, setSteps] = useState("1");
  const [modelType, setModelType] = useState("");
  const [model, setModel] = useState("");
  const [error, setError] = useState(null);

  const getStepOptions = (granularity) => {
    switch (granularity) {
      case "Hourly":
        return [
          { value: "1", label: "1 Step (1-Hour Horizon)" },
          { value: "24", label: "24 Steps (1-Day Horizon)" },
          { value: "168", label: "168 Steps (1-Week Horizon)" },
        ];
      case "Daily":
        return [
          { value: "1", label: "1 Step (1-Day Horizon)" },
          { value: "7", label: "7 Steps (1-Week Horizon)" },
          { value: "30", label: "30 Steps (1-Month Horizon)" },
        ];
      default:
        return [];
    }
  };

  useEffect(() => {
    const stepOptions = getStepOptions(granularity);
    if (stepOptions.length) {
      setSteps(stepOptions[0].value);
    }
  }, [granularity]);

  const handleSubmit = (e) => {
    e.preventDefault();

    if (!tempId || !tempFilename || !forecastType) {
      setError("Missing file or forecast type.");
      return;
    }

    setError(null);

    const baseRoute =
      model === "DHR-ESN" ? "/configure-hybrid" : "/configure-single";

    navigate(
      `${baseRoute}?tempId=${encodeURIComponent(
        tempId
      )}&tempFilename=${encodeURIComponent(
        tempFilename
      )}&originalFileName=${encodeURIComponent(
        originalFileName
      )}&forecastType=${encodeURIComponent(
        forecastType
      )}&granularity=${granularity}&steps=${steps}&modelType=${modelType}&model=${model}&energyDemand=${encodeURIComponent(
        energyDemand
      )}&maxCapacity=${encodeURIComponent(maxCapacity)}`
    );
  };
  const backgroundImage = forecastType
    ? `url(/${forecastType.toLowerCase()}-bg.png)`
    : "none";

  return (
    <div className="relative min-h-screen items-center justify-center">
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
        onSubmit={handleSubmit}
        className="bg-gray-100 p-8 rounded-lg shadow-md w-full max-w-md m-auto">
        <h2 className="text-2xl font-semibold mb-6">Model Selection</h2>

        {error && <p className="text-red-500 mb-4">{error}</p>}

        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Current File
          </label>
          <div className="p-3 bg-gray-50 rounded-md">
            <p className="text-sm text-gray-600">
              {originalFileName || "No file selected"}
            </p>
          </div>
        </div>

        <div className="mb-4">
          <label
            htmlFor="energyDemand"
            className="block text-sm font-medium mb-1">
            Energy Demand (kWh)
          </label>
          <input
            type="text"
            id="energyDemand"
            name="energyDemand"
            value={energyDemand}
            onChange={(e) => {
              const value = e.target.value;
              if (/^\d*\.?\d*$/.test(value)) {
                setEnergyDemand(value);
              }
            }}
            className="block w-full p-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            required
          />
        </div>

        <div className="mb-4">
          <label
            htmlFor="energyDemand"
            className="block text-sm font-medium mb-1">
            Max Energy Capacity (kWh)
          </label>
          <input
            type="text"
            id="maxCapacity"
            name="maxCapacity"
            value={maxCapacity}
            onChange={(e) => {
              const value = e.target.value;
              if (/^\d*\.?\d*$/.test(value)) {
                setMaxCapacity(e.target.value);
              }
            }}
            className="block w-full p-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            required
          />
        </div>

        <div className="mb-4">
          <label className="block text-sm font-medium mb-1">Granularity</label>
          <select
            value={granularity}
            onChange={(e) => setGranularity(e.target.value)}
            className="block w-full p-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
            <option value="Hourly">Hourly</option>
            <option value="Daily">Daily</option>
          </select>
        </div>

        <div className="mb-4">
          <label className="block text-sm font-medium mb-1">Steps</label>
          <select
            value={steps}
            onChange={(e) => setSteps(e.target.value)}
            className="block w-full p-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
            {getStepOptions(granularity).map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        <div className="mb-4">
          <label className="block text-sm font-medium mb-1">Model Type</label>
          <div className="flex gap-4">
            <label className="flex items-center">
              <input
                type="radio"
                value="Single"
                checked={modelType === "Single"}
                onChange={(e) => setModelType(e.target.value)}
                className="mr-2"
              />
              Single
            </label>
            <label className="flex items-center">
              <input
                type="radio"
                value="Hybrid"
                checked={modelType === "Hybrid"}
                onChange={(e) => setModelType(e.target.value)}
                className="mr-2"
              />
              Hybrid
            </label>
          </div>
        </div>

        {modelType && (
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1">Model</label>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="block w-full p-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
              <option value="">Select a model</option>
              {modelType === "Single" ? (
                <>
                  <option value="DHR">DHR</option>
                  <option value="ESN">ESN</option>
                </>
              ) : (
                <option value="DHR-ESN">DHR-ESN</option>
              )}
            </select>
          </div>
        )}

        <div className="flex justify-end gap-3">
          <button
            type="button"
            onClick={() => navigate(-1)}
            className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600 transition-colors">
            Cancel
          </button>
          <button
            type="submit"
            disabled={!granularity || !steps || !modelType || !model}
            className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed">
            Submit
          </button>
        </div>
      </form>
    </div>
  );
};

export default ModelSelection;

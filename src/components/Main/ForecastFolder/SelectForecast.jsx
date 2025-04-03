import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

const SelectForecast = () => {
  const navigate = useNavigate();
  const [fileData, setFileData] = useState({
    filename: "No file selected",
    original_filename: "",
    upload_date: null,
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchLatestFile();
  }, []);

  const fetchLatestFile = async () => {
    try {
      const response = await fetch(
        "http://localhost:8000/storage/latest-file/?data_type=json"
      );
      if (!response.ok) {
        throw new Error("Failed to fetch latest file");
      }
      const data = await response.json();

      const jsonDataResponse = await fetch(
        `http://localhost:8000/storage/json-data/${data.id}`
      );

      if (!jsonDataResponse.ok) {
        throw new Error("Failed to fetch original filename");
      }

      const jsonData = await jsonDataResponse.json();

      setFileData({
        filename: data.filename,
        original_filename:
          jsonData.original_filename || "Unknown original file",
        upload_date: new Date(data.upload_date).toLocaleString(),
      });
    } catch (err) {
      console.error("Error fetching latest file:", err);
      setError("Error fetching latest file");
    } finally {
      setLoading(false);
    }
  };

  const handleModelSelect = async (modelType) => {
    try {
      setLoading(true);
      const response = await fetch(
        `http://localhost:8000/storage/read/${fileData.filename}`
      );
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const data = await response.json();

      if (!Array.isArray(data) || data.length === 0) {
        throw new Error("Data is empty or not an array");
      }

      const { filteredData, folderPrefix } = processData(data, modelType);
      await uploadProcessedData(filteredData, folderPrefix, modelType);

      navigate(`/generate?type=${folderPrefix}`);
    } catch (err) {
      console.error("Error details:", err);
      setError(`Error processing data: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const processData = (data, modelType) => {
    const timePatterns = {
      date: /^\d{4}-\d{2}-\d{2}$/,
      time: /^\d{4}-\d{2}-\d{2}T\d{2}:00:00$/,
      week: /^\d{4}-W\d{2}$/,
    };

    const requiredFields = {
      Solar: [
        "solar_power",
        "dhi",
        "dni",
        "ghi",
        "temperature",
        "relative_humidity",
        "solar_zenith_angle",
      ],
      Wind: ["wind_power", "wind_speed", "dew_point"],
    };

    const firstRow = data[0];
    let selectedTimeField = null;
    let folderPrefix = null;

    for (const [field, regex] of Object.entries(timePatterns)) {
      if (firstRow[field] && regex.test(firstRow[field])) {
        selectedTimeField = field;
        folderPrefix =
          field === "time" ? "hourly" : field === "date" ? "daily" : "weekly";
        break;
      }
    }

    if (!selectedTimeField) {
      throw new Error("No valid time-related field found in data");
    }

    const filteredData = data.map((row) => ({
      [selectedTimeField]: row[selectedTimeField],
      ...Object.fromEntries(
        requiredFields[modelType]
          .filter((field) => row[field] !== undefined)
          .map((field) => [field, row[field]])
      ),
    }));

    if (filteredData.every((row) => Object.keys(row).length <= 1)) {
      throw new Error("Filtered data lacks required fields");
    }

    return { filteredData, folderPrefix };
  };

  const uploadProcessedData = async (filteredData, folderPrefix, modelType) => {
    const latestFileResponse = await fetch(
      `http://localhost:8000/storage/latest-file/?data_type=${folderPrefix}`
    );

    let nextId = 1;

    if (latestFileResponse.ok) {
      const latestFile = await latestFileResponse.json();
      if (latestFile && typeof latestFile.id === "number") {
        nextId = latestFile.id + 1;
      }
    }

    const newFilename = `${nextId}_${folderPrefix}_${modelType.toLowerCase()}_data.json`;

    const formData = new FormData();
    formData.append(
      "file",
      new Blob([JSON.stringify(filteredData, null, 2)], {
        type: "application/json",
      }),
      newFilename
    );
    formData.append("original_filename", fileData.original_filename);

    const response = await fetch(
      "http://localhost:8000/storage/process_model_data/",
      {
        method: "POST",
        body: formData,
      }
    );

    if (!response.ok) {
      throw new Error(await response.text());
    }

    return newFilename;
  };

  return (
    <div className="relative min-h-screen flex items-center justify-center">
      <div
        className="fixed inset-0 overflow-hidden"
        style={{
          backgroundImage: `url(/wind-img.png), url(/solar-img.png)`,
          backgroundSize: "cover",
          backgroundPosition: "center",
          backgroundBlendMode: "overlay",
          zIndex: -1,
        }}
      />
      <div className="p-6 flex flex-col items-center w-full">
        {error && <p className="text-red-500 mb-4">{error}</p>}

        <div className="flex justify-between w-full max-w-2xl">
          <div className="flex flex-col items-center">
            <span className="text-white text-lg font-semibold mb-2">
              Solar Energy
            </span>
            <button
              onClick={() => handleModelSelect("Solar")}
              disabled={loading}
              className="px-6 py-3 bg-blue-500 text-white rounded-md 
                hover:bg-blue-600 transition-colors
                disabled:opacity-50 disabled:cursor-not-allowed">
              Solar
            </button>
          </div>
          <div className="flex flex-col items-center">
            <span className="text-white text-lg font-semibold mb-2">
              Wind Energy
            </span>
            <button
              onClick={() => handleModelSelect("Wind")}
              disabled={loading}
              className="px-6 py-3 bg-green-500 text-white rounded-md 
                hover:bg-green-600 transition-colors
                disabled:opacity-50 disabled:cursor-not-allowed">
              Wind
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SelectForecast;

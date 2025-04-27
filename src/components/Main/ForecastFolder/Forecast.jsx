import React, { useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { useDropzone } from "react-dropzone";

const Forecast = () => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState("");
  const [isUploadDisabled, setIsUploadDisabled] = useState(true);
  const navigate = useNavigate();

  const validateFile = useCallback((file) => {
    if (!file) {
      setMessage("Please select a valid file.");
      return false;
    }

    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target.result;
        const lines = text.split("\n");

        if (lines.length < 2) {
          setMessage("Invalid CSV format.");
          resolve(false);
          return;
        }

        const headers = lines[0].split(",").map((h) => h.trim());
        const rows = lines
          .slice(1)
          .map((line) => line.split(",").map((v) => v.trim()));

        const timeColumns = ["date", "week", "time"];
        const foundTimeColumns = timeColumns.filter((col) =>
          headers.includes(col)
        );

        if (foundTimeColumns.length !== 1) {
          setMessage(
            "CSV must contain exactly one of the following columns: date, week, or time."
          );
          resolve(false);
          return;
        }

        const timeIndex = headers.indexOf(foundTimeColumns[0]);
        const timeColumn = foundTimeColumns[0];

        const formats = {
          date: { regex: /^\d{4}-\d{2}-\d{2}$/, example: "2024-02-05" },
          time: {
            regex: /^\d{4}-\d{2}-\d{2}T\d{2}:00:00$/,
            example: "2024-02-05T14:00:00",
          },
          week: { regex: /^\d{4}-W\d{2}$/, example: "2018-W01" },
        };

        for (let i = 0; i < rows.length; i++) {
          if (!rows[i][timeIndex]) continue;

          const timestamp = rows[i][timeIndex];
          const format = formats[timeColumn];

          if (!format.regex.test(timestamp)) {
            setMessage(
              `Timestamp on row ${i + 1} is not in the correct format (e.g., ${
                format.example
              }).`
            );
            resolve(false);
            return;
          }
        }

        const requiredColumns = [
          "solar_power",
          "dhi",
          "dni",
          "ghi",
          "temperature",
          "relative_humidity",
          "solar_zenith_angle",
          "wind_power",
          "wind_speed",
          "dew_point",
        ];

        const missingColumns = requiredColumns.filter(
          (col) => !headers.includes(col)
        );

        if (missingColumns.length > 0) {
          setMessage(`Missing required columns: ${missingColumns.join(", ")}`);
          resolve(false);
          return;
        }

        setFile(file);
        setMessage("");
        setIsUploadDisabled(false);
        resolve(true);
      };

      reader.readAsText(file);
    });
  }, []);

  const onDrop = useCallback(
    async (acceptedFiles) => {
      if (acceptedFiles.length === 0) {
        setMessage("Please drop a valid CSV file.");
        return;
      }

      // Clear previous file state
      if (file) {
        setFile(null);
        setIsUploadDisabled(true);
      }

      const selectedFile = acceptedFiles[0];
      const isValid = await validateFile(selectedFile);
      if (isValid) {
        setFile(selectedFile);
        setMessage("");
        setIsUploadDisabled(false);
      }
    },
    [validateFile, file]
  );

  const handleDropzoneClick = useCallback(() => {
    // If a file is already selected, we need to manually clear it first
    if (file) {
      setFile(null);
      setIsUploadDisabled(true);
      setMessage("");
    }
  }, [file]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "text/csv": [".csv"] },
    disabled: isProcessing,
    maxFiles: 1,
    onClick: handleDropzoneClick,
  });

  const handleUpload = (event) => {
    event.preventDefault();
    event.stopPropagation();

    if (!file) {
      setMessage("Please select a file before uploading.");
      return;
    }

    setIsProcessing(true);
    setIsUploadDisabled(true);

    const worker = new Worker(
      new URL("../../../../src/config/workerCSV.js", import.meta.url),
      { type: "module" }
    );

    worker.onmessage = (e) => {
      const { type, data, error } = e.data;

      if (type === "complete") {
        worker.terminate();
        uploadJsonToStorage(data);
      } else if (type === "error") {
        console.error("Error processing file:", error);
        worker.terminate();
        setIsProcessing(false);
        setMessage(error);
        setIsUploadDisabled(false);
      }
    };

    worker.postMessage({ file });
  };

  const uploadJsonToStorage = async (jsonData) => {
    try {
      const originalName = file.name.replace(/\.csv$/, "");
      const originalFilename = file.name;

      const timeColumn = Object.keys(jsonData[0]).find((key) =>
        ["time", "week", "date"].includes(key)
      );

      const prefixMap = {
        time: "hourly",
        week: "weekly",
        date: "daily",
      };

      const prefix = prefixMap[timeColumn] || "";

      const latestFileResponse = await fetch(
        `http://localhost:8000/storage/latest-file/?data_type=json`
      );

      let nextId = 1;
      if (latestFileResponse.ok) {
        const latestFile = await latestFileResponse.json();
        if (latestFile && typeof latestFile.id === "number") {
          nextId = latestFile.id + 1;
        }
      }

      const newFilename = `${nextId}_${prefix}_${originalName}.json`;
      const blob = new Blob([JSON.stringify(jsonData, null, 2)], {
        type: "application/json",
      });

      const formData = new FormData();
      formData.append("file", blob, newFilename);
      formData.append("original_filename", originalFilename);

      const response = await fetch(
        "http://localhost:8000/storage/upload_csv/",
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        const errorText = await response.text();
        console.error("Upload failed:", errorText);
        setMessage(`Upload failed: ${errorText}`);
        setIsUploadDisabled(false);
        return;
      }

      // Fetch the token from localStorage
      const token = localStorage.getItem("token")?.trim();
      if (!token) {
        setMessage("You are not logged in. Please log in to continue.");
        setIsProcessing(false);
        setIsUploadDisabled(false);
        navigate("/login");
        return;
      }

      const forecastResponse = await fetch(
        "http://localhost:8000/api/forecasts",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`, // Add the Authorization header
          },
          body: JSON.stringify({
            filename: newFilename,
            original_filename: originalFilename,
            forecast_model: "pending",
            steps: "pending",
            granularity: prefix,
          }),
        }
      );

      if (!forecastResponse.ok) {
        if (forecastResponse.status === 401) {
          // Token is invalid or expired, redirect to login
          setMessage("Your session has expired. Please log in again.");
          localStorage.removeItem("token"); // Clear invalid token
          navigate("/login");
          return;
        }
        const errorText = await forecastResponse.text();
        throw new Error(`Failed to create forecast entry: ${errorText}`);
      }

      navigate("/select-forecast");
    } catch (error) {
      console.error("Upload error:", error);
      setMessage(`Upload error: ${error.message}`);
      setIsUploadDisabled(false);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="relative min-h-screen flex flex-col items-center justify-center">
      <div
        className="fixed inset-0"
        style={{
          backgroundImage: `url(/wind-img.png), url(/solar-img.png)`,
          backgroundSize: "cover",
          backgroundPosition: "center",
          backgroundBlendMode: "overlay",
          zIndex: -1,
        }}
      />
      <div className="fixed inset-0 bg-black/60" style={{ zIndex: -1 }} />

      <div className="w-full max-w-2xl px-4 flex flex-col items-center">
        <div className="max-w-2xl">
          <h1 className="text-[50px] font-bold mb-5 flex justify-center text-white">
            Upload CSV File
          </h1>
        </div>
        {/* Dropzone centered horizontally */}
        <div
          {...getRootProps()}
          className={`flex flex-col items-center justify-center w-full h-64 border-2 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100 dark:bg-gray-700 dark:hover:bg-gray-600 dark:border-gray-600 dark:hover:border-gray-500 p-6 text-center ${
            isDragActive ? "border-blue-500 bg-blue-50" : "border-gray-300"
          }`}>
          <div className="flex flex-col items-center justify-center pt-5 pb-6">
            <svg
              className="w-8 h-8 mb-4 text-gray-500 dark:text-gray-400"
              aria-hidden="true"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 20 16">
              <path
                stroke="currentColor"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2"
              />
            </svg>
            {file ? (
              <>
                <p className="mb-2 text-sm text-green-600">
                  Selected file: {file.name}
                </p>
                <p className="mb-5 text-sm text-gray-500 dark:text-gray-400">
                  <span className="font-semibold">Click to change file</span> or
                  drag and drop
                </p>
                <button
                  onClick={handleUpload}
                  disabled={isUploadDisabled || isProcessing}
                  className={`py-2 px-4 rounded-lg text-white whitespace-nowrap ${
                    isUploadDisabled || isProcessing
                      ? "bg-gray-400 cursor-not-allowed"
                      : "bg-blue-500 hover:bg-blue-600"
                  } transition-colors`}>
                  {isProcessing ? "Processing..." : "Proceed"}
                </button>
              </>
            ) : (
              <>
                <p className="mb-2 text-sm text-gray-500 dark:text-gray-400">
                  <span className="font-semibold">Click to upload</span> or drag
                  and drop
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  Only CSV files are accepted
                </p>
              </>
            )}
            {message && <p className="mt-4 text-sm text-red-500">{message}</p>}
          </div>
          <input {...getInputProps()} />
        </div>
      </div>
    </div>
  );
};

export default Forecast;

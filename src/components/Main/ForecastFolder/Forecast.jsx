import React, { useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { useDropzone } from "react-dropzone";

const Forecast = () => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState("");
  const [isUploadDisabled, setIsUploadDisabled] = useState(true);
  const [isNextDisabled, setIsNextDisabled] = useState(true);
  const navigate = useNavigate();

  const validateFile = useCallback((file) => {
    if (!file) {
      setMessage("Please select a valid file.");
      return false;
    }

    const reader = new FileReader();

    return new Promise((resolve) => {
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

        // Validate timestamps
        for (let i = 0; i < rows.length; i++) {
          if (!rows[i][timeIndex]) continue;
          const timestamp = rows[i][timeIndex];

          const formats = {
            date: {
              regex: /^\d{4}-\d{2}-\d{2}$/,
              example: "2024-02-05",
            },
            time: {
              regex: /^\d{4}-\d{2}-\d{2}T\d{2}:00:00$/,
              example: "2024-02-05T14:00:00",
            },
            week: {
              regex: /^\d{4}-W\d{2}$/,
              example: "2018-W01",
            },
          };

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

      const selectedFile = acceptedFiles[0];
      const isValid = await validateFile(selectedFile);

      if (isValid) {
        setFile(selectedFile);
        setMessage("");
        setIsUploadDisabled(false);
      }
    },
    [validateFile]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "text/csv": [".csv"] },
    disabled: isProcessing,
    maxFiles: 1,
  });

  const handleUpload = () => {
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
        setIsProcessing(false);
        setIsNextDisabled(false);
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
    const originalName = file.name.replace(/\.csv$/, "");

    // Get the first row of data to check which time column exists
    const timeColumn = Object.keys(jsonData[0]).find((key) =>
      ["time", "week", "date"].includes(key)
    );

    let prefix = "";
    if (timeColumn === "time") {
      prefix = "hourly";
    } else if (timeColumn === "week") {
      prefix = "weekly";
    } else if (timeColumn === "date") {
      prefix = "daily";
    }

    try {
      // Get the latest ID from json_data table
      const latestFileResponse = await fetch(
        `http://localhost:8000/storage/latest-file/?data_type=json`
      );

      let nextId = 1; // Default to 1 if no files exist

      if (latestFileResponse.ok) {
        const latestFile = await latestFileResponse.json();

        // Make sure we're getting a valid ID
        if (latestFile && typeof latestFile.id === "number") {
          nextId = latestFile.id + 1; // Increment by 1 to get next available ID
        } else {
          console.warn("No valid ID found in response:", latestFile);
        }
      } else {
        console.warn(
          "Failed to fetch latest file:",
          await latestFileResponse.text()
        );
      }

      const newFilename = `${nextId}_${prefix}_${originalName}.json`;

      const blob = new Blob([JSON.stringify(jsonData, null, 2)], {
        type: "application/json",
      });
      const formData = new FormData();
      formData.append("file", blob, newFilename);

      const response = await fetch(
        "http://localhost:8000/storage/upload_csv/",
        {
          method: "POST",
          body: formData,
        }
      );

      if (response.ok) {
        const uploadResult = await response.json();
        setMessage("File uploaded successfully!");
        setTimeout(() => {
          navigate("/SelectForecast");
        }, 2000);
      } else {
        const errorText = await response.text();
        console.error("Upload failed:", errorText);
        setMessage(`Upload failed: ${errorText}`);
        setIsUploadDisabled(false);
      }
    } catch (error) {
      console.error("Upload error:", error);
      setMessage(`Upload error: ${error.message}`);
      setIsUploadDisabled(false);
    }
  };

  return (
    <div className="p-6 bg-gray-100 min-h-screen flex flex-col items-center">
      <div
        {...getRootProps()}
        className={`w-full max-w-md p-6 mb-4 border-2 border-dashed rounded-lg ${
          isDragActive ? "border-blue-500 bg-blue-50" : "border-gray-300"
        } cursor-pointer text-center hover:bg-gray-50 transition-colors`}>
        <input {...getInputProps()} />
        {isDragActive ? (
          <p className="text-blue-500">Drop the CSV file here...</p>
        ) : (
          <div>
            <p className="mb-2">
              Drag & drop a CSV file here, or click to select a file
            </p>
            <p className="text-sm text-gray-500">Only CSV files are accepted</p>
            {file && (
              <p className="mt-2 text-green-600">Selected: {file.name}</p>
            )}
          </div>
        )}
      </div>

      <button
        onClick={handleUpload}
        disabled={isUploadDisabled || isProcessing}
        className={`w-full max-w-md py-2 px-4 rounded-lg text-white ${
          isUploadDisabled || isProcessing
            ? "bg-gray-400 cursor-not-allowed"
            : "bg-blue-500 hover:bg-blue-600"
        } transition-colors`}>
        {isProcessing ? "Processing..." : "Upload"}
      </button>

      {message && (
        <p
          className={`mt-4 ${
            message.includes("successfully") ? "text-green-500" : "text-red-500"
          }`}>
          {message}
        </p>
      )}
    </div>
  );
};

export default Forecast;

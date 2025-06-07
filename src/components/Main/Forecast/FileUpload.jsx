import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useDropzone } from "react-dropzone";
import Papa from "papaparse";

export default function FileUpload() {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState("");
  const [uploading, setUploading] = useState(false);
  const navigate = useNavigate();

  // Define required columns
  const requiredColumns = [
    "time",
    "load_power",
    "wind_power",
    "solar_power",
    "DHI",
    "DNI",
    "GHI",
    "Dew Point",
    "Solar Zenith Angle",
    "Wind Speed",
    "Relative Humidity",
    "Temperature",
  ];

  // Regular expression for validating time format (YYYY-MM-DD HH:MM:SS)
  const timeFormatRegex = /^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/;

  const onDrop = (acceptedFiles, fileRejections) => {
    if (fileRejections.length > 0) {
      setMessage("Only .csv files are allowed.");
      setFile(null);
      return;
    }

    const selectedFile = acceptedFiles[0];

    // Parse CSV to validate columns and data
    Papa.parse(selectedFile, {
      header: true,
      complete: (result) => {
        const headers = result.meta.fields;
        const missingColumns = requiredColumns.filter(
          (col) => !headers.includes(col)
        );

        if (missingColumns.length > 0) {
          setMessage(`Missing required columns: ${missingColumns.join(", ")}`);
          setFile(null);
          return;
        }

        if (headers.length > requiredColumns.length) {
          setMessage(
            `Extra columns detected. Please include only: ${requiredColumns.join(
              ", "
            )}`
          );
          setFile(null);
          return;
        }

        // Validate data rows
        let dataValid = true;
        let errorMessage = "";

        for (let i = 0; i < result.data.length; i++) {
          const row = result.data[i];

          // Skip empty or malformed rows (e.g., rows with all empty or null values)
          const isRowEmpty = Object.values(row).every(
            (value) => value === undefined || value === "" || value === null
          );
          if (isRowEmpty) continue;

          // Validate time format
          const timeValue = row["time"];
          if (!timeValue || !timeFormatRegex.test(timeValue)) {
            dataValid = false;
            errorMessage = `Invalid or missing time format in row ${
              i + 2
            }. Expected format: YYYY-MM-DD HH:MM:SS`;
            break;
          }

          // Validate numeric columns
          const numericColumns = requiredColumns.filter(
            (col) => col !== "time"
          );
          for (const col of numericColumns) {
            const value = row[col];
            if (
              value === undefined ||
              value === "" ||
              isNaN(parseFloat(value))
            ) {
              dataValid = false;
              errorMessage = `Invalid or missing numeric value for ${col} in row ${
                i + 2
              }`;
              break;
            }
          }

          if (!dataValid) break;
        }

        if (!dataValid) {
          setMessage(errorMessage);
          setFile(null);
          return;
        }

        setFile(selectedFile);
        setMessage("");
      },
      error: (error) => {
        setMessage("Error parsing CSV file: " + error.message);
        setFile(null);
      },
    });
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "text/csv": [".csv"] },
    maxFiles: 1,
  });

  const uploadTempFile = async (file) => {
    const formData = new FormData();
    formData.append("file", file);

    try {
      setUploading(true);
      const response = await fetch("http://localhost:8000/temp-upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Upload failed: ${errorText}`);
      }

      const data = await response.json();
      setUploading(false);
      return data;
    } catch (error) {
      setUploading(false);
      setMessage(error.message || "Upload failed");
      return null;
    }
  };

  const handleNext = async (e) => {
    e.stopPropagation();

    if (!file) {
      setMessage("Please select a valid CSV file.");
      return;
    }

    const uploadResult = await uploadTempFile(file);
    if (!uploadResult) return;

    navigate(
      `/select-type?tempId=${encodeURIComponent(
        uploadResult.temp_id
      )}&tempFilename=${encodeURIComponent(
        uploadResult.temp_filename
      )}&originalFileName=${encodeURIComponent(file.name)}`
    );
  };

  return (
    <div className="relative min-h-screen flex flex-col items-center">
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
        <h1 className="text-[50px] font-bold mb-5 flex justify-center text-white">
          Upload CSV File
        </h1>
        <div
          {...getRootProps()}
          className={`flex flex-col items-center justify-center w-full h-64 border-2 border-dashed rounded-lg cursor-pointer bg-gray-7 hover:bg-gray-6 dark:bg-gray-700 dark:hover:bg-gray-600 dark:border-gray-600 dark:hover:border-gray-500 p-6 text-center ${
            isDragActive ? "border-blue-500 bg-blue-50" : "border-gray-300"
          }`}>
          <div className="flex flex-col items-center justify-center pt-5 pb-6">
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
                  onClick={handleNext}
                  disabled={uploading}
                  className="py-2 px-4 rounded-lg text-white bg-blue-500 border-none hover:bg-blue-600 transition-colors disabled:opacity-50">
                  {uploading ? "Uploading..." : "Proceed"}
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
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  Supported columns: {requiredColumns.join(", ")}
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
}

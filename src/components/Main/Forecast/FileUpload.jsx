import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useDropzone } from "react-dropzone";

export default function FileUpload() {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState("");
  const [uploading, setUploading] = useState(false);
  const navigate = useNavigate();

  const onDrop = (acceptedFiles, fileRejections) => {
    if (fileRejections.length > 0) {
      setMessage("Only .csv files are allowed.");
      setFile(null);
      return;
    }
    setFile(acceptedFiles[0]);
    setMessage("");
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "text/csv": [".csv"] },
    maxFiles: 1,
  });

  // Function to upload file to backend /temp-upload
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
      return data; // { temp_filename, temp_id }
    } catch (error) {
      setUploading(false);
      setMessage(error.message || "Upload failed");
      return null;
    }
  };

  const handleNext = async (e) => {
    e.stopPropagation(); // Prevent dropzone click event

    if (!file) {
      setMessage("Please select a CSV file.");
      return;
    }

    // Upload the file first
    const uploadResult = await uploadTempFile(file);
    if (!uploadResult) return; // upload failed, error message set

    // Then navigate, passing along the temp info
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
      {/* Background overlays omitted for brevity */}
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
            {/* SVG omitted for brevity */}
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
                  className="py-2 px-4 rounded-lg text-white bg-blue-500 hover:bg-blue-600 transition-colors disabled:opacity-50">
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

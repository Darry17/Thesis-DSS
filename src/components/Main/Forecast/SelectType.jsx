import React, { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";

function useQuery() {
  return new URLSearchParams(useLocation().search);
}

export default function SelectType() {
  const query = useQuery();
  const navigate = useNavigate();
  const [error, setError] = useState("");

  const tempId = query.get("tempId");
  const tempFilename = query.get("tempFilename");
  const originalFileName = query.get("originalFileName");

  const handleSelect = (type) => {
    if (!tempId || !tempFilename) {
      setError("No file provided.");
      return;
    }

    navigate(
      `/model-selection?tempId=${encodeURIComponent(
        tempId
      )}&tempFilename=${encodeURIComponent(
        tempFilename
      )}&originalFileName=${encodeURIComponent(
        originalFileName
      )}&forecastType=${type.toLowerCase()}`
    );
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
      <div className="fixed inset-0" style={{ zIndex: -1 }} />
      <div className="p-6 flex flex-col items-center w-full">
        {error && <p className="text-red-500 mb-4">{error}</p>}

        <div className="flex justify-between w-full max-w-2xl gap-4">
          <div className="flex flex-col items-center">
            <span className="text-black text-lg font-semibold mb-2">
              Solar Energy
            </span>
            <button
              onClick={() => handleSelect("Solar")}
              className="px-6 py-3 bg-green-600 text-white rounded-md hover:bg-green-500 transition-colors">
              Solar
            </button>
          </div>
          <div className="flex flex-col items-center">
            <span className="text-white text-lg font-semibold mb-2">
              Wind Energy
            </span>
            <button
              onClick={() => handleSelect("Wind")}
              className="px-6 py-3 bg-green-600 text-white rounded-md hover:bg-green-500 transition-colors">
              Wind
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

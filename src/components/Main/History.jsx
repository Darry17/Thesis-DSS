import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

const History = () => {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchLogs = async () => {
      try {
        setLoading(true);
        const response = await fetch("http://localhost:8000/api/history-logs");

        if (!response.ok) {
          throw new Error(`Error fetching logs: ${response.statusText}`);
        }

        const data = await response.json();
        setLogs(data);
      } catch (err) {
        console.error("Error fetching history logs:", err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchLogs();
  }, []);

  const handleView = (forecastId) => {
    if (forecastId) {
      navigate("/view-logs", { state: { forecastId } });
    } else {
      alert("No forecast ID associated with this log entry");
    }
  };

  const handleDelete = async (forecastId) => {
    if (!window.confirm("Are you sure you want to delete this forecast?"))
      return;

    try {
      const response = await fetch(
        `http://localhost:8000/api/forecasts/${forecastId}`,
        {
          method: "DELETE",
        }
      );

      if (!response.ok) {
        throw new Error("Failed to delete forecast");
      }

      // Update the UI by removing the deleted log
      setLogs(logs.filter((log) => log.forecast_id !== forecastId));
      alert("Forecast deleted successfully");
    } catch (err) {
      console.error("Error deleting forecast:", err);
      alert(`Error: ${err.message}`);
    }
  };

  // Format date to YYYY/MM/DD
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return `${date.getFullYear()}/${String(date.getMonth() + 1).padStart(
      2,
      "0"
    )}/${String(date.getDate()).padStart(2, "0")}`;
  };

  if (loading) {
    return (
      <div className="p-6 min-h-screen flex justify-center items-center">
        <div className="text-xl">Loading...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 min-h-screen flex justify-center items-center">
        <div className="text-xl text-red-500">Error: {error}</div>
      </div>
    );
  }

  return (
    <div className="px-40 py-0 bg-white min-h-screen">
      <h1 className="text-4xl font-bold mb-6">Forecasted Logs</h1>

      <div className="overflow-x-auto">
        <table className="min-w-full bg-white border border-gray-200">
          <thead className="bg-gray-100">
            <tr>
              <th className="py-3 px-4 text-left border-b">File Name</th>
              <th className="py-3 px-4 text-left border-b">Model</th>
              <th className="py-3 px-4 text-left border-b">Action</th>
              <th className="py-3 px-4 text-left border-b">Date</th>
              <th className="py-3 px-4 text-left border-b">Username</th>
            </tr>
          </thead>
          <tbody>
            {logs.length === 0 ? (
              <tr>
                <td colSpan="5" className="py-4 px-4 text-center text-gray-500">
                  No logs found
                </td>
              </tr>
            ) : (
              logs.map((log) => (
                <tr key={log.id} className="hover:bg-gray-50">
                  <td className="py-3 px-4 border-b">{log.file_name}</td>
                  <td className="py-3 px-4 border-b">
                    <span className="text-blue-600">{log.model}</span>
                  </td>
                  <td className="py-3 px-4 border-b">
                    <div className="flex gap-6">
                      <button
                        onClick={() => handleView(log.forecast_id)}
                        className="px-3 py-1 bg-green-500 text-white text-xs rounded hover:bg-green-600">
                        VIEW
                      </button>
                      <button
                        onClick={() => handleDelete(log.forecast_id)}
                        className="px-3 py-1 bg-red-500 text-white text-xs rounded hover:bg-red-600">
                        DELETE
                      </button>
                    </div>
                  </td>
                  <td className="py-3 px-4 border-b">{formatDate(log.date)}</td>
                  <td className="py-3 px-4 border-b">{log.username}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default History;

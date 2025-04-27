import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

const RecoveryLogs = () => {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [userRole, setUserRole] = useState(null); // To store the user's role
  const navigate = useNavigate();

  useEffect(() => {
    const fetchUserRole = async () => {
      try {
        const token = localStorage.getItem("token")?.trim();
        if (!token) {
          throw new Error("No authentication token found");
        }

        const response = await fetch(
          "http://localhost:8000/api/validate-token",
          {
            headers: {
              Authorization: `Bearer ${token}`,
            },
          }
        );

        if (!response.ok) {
          throw new Error("Failed to validate token");
        }

        const data = await response.json();
        setUserRole(data.access_control); // Store the user's role (e.g., "ADMIN")
      } catch (err) {
        console.error("Error validating token:", err);
        setError(err.message);
      }
    };

    const fetchDeletedForecasts = async () => {
      try {
        setLoading(true);
        const token = localStorage.getItem("token")?.trim();
        if (!token) {
          throw new Error("No authentication token found");
        }

        const response = await fetch(
          "http://localhost:8000/api/deleted-forecasts",
          {
            headers: {
              Authorization: `Bearer ${token}`,
            },
          }
        );

        if (!response.ok) {
          throw new Error(
            `Error fetching deleted forecasts: ${response.statusText}`
          );
        }

        const data = await response.json();
        setLogs(data);
      } catch (err) {
        console.error("Error fetching deleted forecasts:", err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchUserRole();
    fetchDeletedForecasts();
  }, []);

  const handleRecover = async (forecastId) => {
    if (!window.confirm("Are you sure you want to recover this forecast?"))
      return;

    try {
      const token = localStorage.getItem("token")?.trim();
      if (!token) {
        throw new Error("No authentication token found");
      }

      const response = await fetch(
        `http://localhost:8000/api/recover-forecast/${forecastId}`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      if (!response.ok) {
        throw new Error("Failed to recover forecast");
      }

      // Update the UI by removing the recovered log
      setLogs(logs.filter((log) => log.forecast_id !== forecastId));
      alert("Forecast recovered successfully");
    } catch (err) {
      console.error("Error recovering forecast:", err);
      alert(`Error: ${err.message}`);
    }
  };

  const handleDelete = async (id) => {
    if (
      !window.confirm(
        "Are you sure you want to permanently delete this forecast? This action cannot be undone."
      )
    )
      return;

    try {
      const token = localStorage.getItem("token")?.trim();
      if (!token) {
        throw new Error("No authentication token found");
      }

      const response = await fetch(
        `http://localhost:8000/api/deleted-forecasts/id/${id}`,
        {
          method: "DELETE",
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      if (!response.ok) {
        throw new Error("Failed to delete forecast");
      }

      // Update the UI by removing the deleted log
      setLogs(logs.filter((log) => log.id !== id));
      alert("Forecast permanently deleted successfully");
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
    <div className="p-6 bg-gray-100 min-h-screen">
      <div className="max-w-5xl mx-auto">
        <h1 className="text-4xl font-bold mb-6">Recovery Logs</h1>

        <div className="overflow-x-auto">
          <table className="min-w-full bg-white border border-gray-200">
            <thead className="bg-gray-100">
              <tr>
                <th className="py-3 px-4 text-left border-b">File Name</th>
                <th className="py-3 px-4 text-left border-b">Model</th>
                <th className="py-3 px-4 text-left border-b">Action</th>
                <th className="py-3 px-4 text-left border-b">Date</th>
                <th className="py-3 px-4 text-left border-b">Username</th>
                <th className="py-3 px-4 text-left border-b">Deleted By</th>
              </tr>
            </thead>
            <tbody>
              {logs.length === 0 ? (
                <tr>
                  <td
                    colSpan="6"
                    className="py-4 px-4 text-center text-gray-500">
                    No deleted forecasts found
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
                        {userRole === "ADMIN" && (
                          <>
                            {log.forecast_id ? (
                              <button
                                onClick={() => handleRecover(log.forecast_id)}
                                className="px-3 py-1 bg-green-500 text-white text-xs rounded hover:bg-green-600">
                                RECOVER
                              </button>
                            ) : (
                              <span className="text-gray-500 text-xs">
                                Cannot recover (No forecast ID)
                              </span>
                            )}
                            <button
                              onClick={() => handleDelete(log.id)}
                              className="px-3 py-1 bg-red-500 text-white text-xs rounded hover:bg-red-700">
                              DELETE
                            </button>
                          </>
                        )}
                      </div>
                    </td>
                    <td className="py-3 px-4 border-b">
                      {formatDate(log.date)}
                    </td>
                    <td className="py-3 px-4 border-b">{log.username}</td>
                    <td className="py-3 px-4 border-b">{log.deleted_by}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default RecoveryLogs;

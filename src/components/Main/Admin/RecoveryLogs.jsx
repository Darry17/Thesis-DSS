import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";

const RecoveryLogs = () => {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [userRole, setUserRole] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [searchQuery, setSearchQuery] = useState("");
  const [inputValue, setInputValue] = useState("");
  const logsPerPage = 10;
  const navigate = useNavigate();

  useEffect(() => {
    const fetchUserRole = async () => {
      try {
        const token = localStorage.getItem("token")?.trim();
        if (!token) {
          throw new Error("No authentication token found");
        }

        const response = await axios.get(
          "http://localhost:8000/api/validate-token",
          {
            headers: {
              Authorization: `Bearer ${token}`,
            },
          }
        );

        setUserRole(response.data.access_control);
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

        const response = await axios.get(
          "http://localhost:8000/api/deleted-forecasts",
          {
            headers: {
              Authorization: `Bearer ${token}`,
            },
            params: {
              page: currentPage,
              limit: logsPerPage,
              search: searchQuery,
            },
          }
        );

        setLogs(response.data.logs);
        setTotalPages(response.data.total_pages);
      } catch (err) {
        console.error("Error fetching deleted forecasts:", err);
        setError(err.message || "Failed to fetch deleted forecasts");
      } finally {
        setLoading(false);
      }
    };

    fetchUserRole();
    fetchDeletedForecasts();
  }, [currentPage, searchQuery]);

  const handleRecover = async (forecastId) => {
    if (!window.confirm("Are you sure you want to recover this forecast?"))
      return;

    try {
      const token = localStorage.getItem("token")?.trim();
      if (!token) {
        throw new Error("No authentication token found");
      }

      const response = await axios.post(
        `http://localhost:8000/api/recover-forecast/${forecastId}`,
        {},
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      if (response.status !== 200) {
        throw new Error("Failed to recover forecast");
      }

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

      const response = await axios.delete(
        `http://localhost:8000/api/deleted-forecasts/id/${id}`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      if (response.status !== 200) {
        throw new Error("Failed to delete forecast");
      }

      setLogs(logs.filter((log) => log.id !== id));
      alert("Forecast permanently deleted successfully");
    } catch (err) {
      console.error("Error deleting forecast:", err);
      alert(`Error: ${err.message}`);
    }
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return `${date.getFullYear()}/${String(date.getMonth() + 1).padStart(
      2,
      "0"
    )}/${String(date.getDate()).padStart(2, "0")}`;
  };

  const handlePageChange = (newPage) => {
    if (newPage >= 1 && newPage <= totalPages) {
      setCurrentPage(newPage);
    }
  };

  const handleInputChange = (e) => {
    setInputValue(e.target.value);
  };

  const handleSearch = (e) => {
    if (e.key === "Enter") {
      setSearchQuery(inputValue);
      setCurrentPage(1);
    }
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
    <div className="p-6 min-h-screen flex-1">
      <div className="max-w-5xl mx-auto">
        <h1 className="text-4xl font-bold mb-6">Recovery Logs</h1>
        <div className="mb-4">
          <input
            type="text"
            value={inputValue}
            onChange={handleInputChange}
            onKeyDown={handleSearch}
            placeholder="Search by file name or model (press Enter to search)"
            className="w-full p-2 border border-gray-300 rounded"
          />
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full bg-white border border-gray-200">
            <thead className="bg-gray-100">
              <tr>
                <th className="py-3 px-4 text-left border-b">File Name</th>
                <th className="py-3 px-4 text-left border-b">Model</th>
                <th className="py-3 px-4 text-left border-b">Action</th>
                <th className="py-3 px-4 text-left border-b">Date</th>
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
                    <td className="py-3 px-4 border-b">{log.deleted_by}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
        <div className="mt-4 flex justify-between items-center">
          <button
            onClick={() => handlePageChange(currentPage - 1)}
            disabled={currentPage === 1}
            className="px-4 py-2 bg-gray-300 rounded hover:bg-gray-400 disabled:opacity-50 disabled:cursor-auto disabled:bg-gray-300 cursor-pointer">
            Previous
          </button>
          <span>
            Page {currentPage} of {totalPages}
          </span>
          <button
            onClick={() => handlePageChange(currentPage + 1)}
            disabled={currentPage === totalPages}
            className="px-4 py-2 bg-gray-300 rounded hover:bg-gray-400 disabled:opacity-50 disabled:cursor-auto disabled:bg-gray-300 cursor-pointer">
            Next
          </button>
        </div>
      </div>
    </div>
  );
};

export default RecoveryLogs;

// src/components/Main/History.jsx (from previous messages)
import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";

const History = () => {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [searchQuery, setSearchQuery] = useState("");
  const [inputValue, setInputValue] = useState("");
  const logsPerPage = 10;
  const navigate = useNavigate();

  useEffect(() => {
    const fetchLogs = async () => {
      try {
        setLoading(true);
        const token = localStorage.getItem("token")?.trim();
        setIsLoggedIn(!!token);

        if (token) {
          await axios.get("http://localhost:8000/api/validate-token", {
            headers: { Authorization: `Bearer ${token}` },
          });
        }

        const response = await axios.get(
          "http://localhost:8000/api/history-logs",
          {
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
        console.error("Error fetching data:", err);
        setError(err.message || "Failed to fetch logs");
        if (err.response?.status === 401) {
          localStorage.removeItem("token");
          setIsLoggedIn(false);
        }
      } finally {
        setLoading(false);
      }
    };

    fetchLogs();
  }, [currentPage, searchQuery]);

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
      const token = localStorage.getItem("token")?.trim();
      const response = await axios.delete(
        `http://localhost:8000/api/forecasts/${forecastId}`,
        {
          headers: { Authorization: `Bearer ${token}` },
        }
      );

      if (response.status !== 200) {
        throw new Error("Failed to delete forecast");
      }

      setLogs(logs.filter((log) => log.forecast_id !== forecastId));
      alert("Forecast deleted successfully");
    } catch (err) {
      console.error("Error deleting forecast:", err);
      alert(`Error: ${err.message}`);
      if (err.response?.status === 401) {
        localStorage.removeItem("token");
        setIsLoggedIn(false);
        navigate("/login");
      }
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
      <div className="p-6 flex justify-center items-center">
        <div className="text-xl">Loading...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 flex justify-center items-center">
        <div className="text-xl text-red-500">Error: {error}</div>
      </div>
    );
  }

  return (
    <div className="p-6 flex-1">
      <div className="max-w-5xl mx-auto">
        <h1 className="text-4xl font-bold mb-6">Forecasted Logs</h1>
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
                <th className="py-3 px-4 text-left border-b">Forecast Type</th>
                <th className="py-3 px-4 text-left border-b">Granularity</th>
                <th className="py-3 px-4 text-left border-b">Steps</th>
                <th className="py-3 px-4 text-left border-b">Model</th>
                <th className="py-3 px-4 text-left border-b">Action</th>
                <th className="py-3 px-4 text-left border-b">Date</th>
              </tr>
            </thead>
            <tbody>
              {logs.length === 0 ? (
                <tr>
                  <td
                    colSpan="5"
                    className="py-4 px-4 text-center text-gray-500">
                    No logs found
                  </td>
                </tr>
              ) : (
                logs.map((log) => (
                  <tr key={log.id} className="hover:bg-gray-50">
                    <td className="py-3 px-4 border-b">{log.file_name}</td>
                    <td className="py-3 px-4 border-b">
                      {log.forecast_type.charAt(0).toUpperCase() +
                        log.forecast_type.slice(1).toLowerCase()}
                    </td>

                    <td className="py-3 px-4 border-b">{log.granularity}</td>
                    <td className="py-3 px-4 border-b">{log.steps}</td>
                    <td className="py-3 px-4 border-b">
                      <span className="text-blue-600">{log.model}</span>
                    </td>
                    <td className="py-3 px-4 border-b">
                      <div className="flex gap-6">
                        <button
                          onClick={() => handleView(log.forecast_id)}
                          className="px-3 py-1 bg-green-500 border-none text-white text-xs rounded hover:bg-green-600 cursor-pointer">
                          VIEW
                        </button>
                        {isLoggedIn && (
                          <button
                            onClick={() => handleDelete(log.forecast_id)}
                            className="px-3 py-1 bg-red-500 border-none text-white text-xs rounded hover:bg-red-600 cursor-pointer">
                            DELETE
                          </button>
                        )}
                      </div>
                    </td>
                    <td className="py-3 px-4 border-b">
                      {formatDate(log.date)}
                    </td>
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

export default History;

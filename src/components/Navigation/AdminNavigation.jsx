import React, { useEffect, useState } from "react";
import { NavLink } from "react-router-dom";

const AdminNavigation = () => {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [username, setUsername] = useState("");

  // Check token and decode username on mount and route changes
  useEffect(() => {
    const token = localStorage.getItem("token")?.trim();
    if (token) {
      try {
        const payload = JSON.parse(atob(token.split(".")[1]));
        setUsername(payload.sub || "Admin");
        setIsLoggedIn(true);
      } catch (e) {
        console.error("Error decoding token:", e);
        localStorage.removeItem("token"); // Clear invalid token
        setIsLoggedIn(false);
        setUsername("");
      }
    } else {
      setIsLoggedIn(false);
      setUsername("");
    }
  }, []);

  // Handle logout
  const handleLogout = () => {
    localStorage.removeItem("token");
    setIsLoggedIn(false);
    setUsername("");
    navigate("/", { replace: true });
  };

  return (
    <div className="top-0 left-0 h-full w-64 bg-gray-400 text-white p-4 font-medium">
      <div className="flex flex-col h-full">
        {/* Navigation Links */}
        <ul className="flex flex-col space-y-5">
          <li>
            <NavLink to="/accounts">Accounts</NavLink>
          </li>
          <li>
            <NavLink to="/recovery-logs">Recovery Logs</NavLink>
          </li>
        </ul>

        {/* Spacer to push login/logout to bottom */}
        <div className="flex-grow"></div>

        {/* Login/Logout Button */}
        <div className="mb-4">
          <div className="flex flex-col space-y-2">
            <span className="px-4 py-2 text-gray-200">{username}</span>
            <button
              onClick={handleLogout}
              className="w-full px-4 py-2 bg-red-500 text-white text-sm rounded hover:bg-red-600 text-left">
              Logout
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdminNavigation;

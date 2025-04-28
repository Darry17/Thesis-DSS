import React, { useEffect, useState } from "react";
import { NavLink, useNavigate } from "react-router-dom";

const AdminNavigation = () => {
  const navigate = useNavigate();
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [username, setUsername] = useState("");

  // Check token and decode username on mount
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

  const linkClass = "block py-2 px-4 rounded w-full text-left";

  return (
    <div className="w-64 bg-[#999696] text-black p-4 font-medium min-h-screen">
      <div className="flex flex-col h-full">
        {/* Navigation Links */}
        <ul className="flex flex-col space-y-5">
          <li>
            <NavLink
              to="/admin"
              className={({ isActive }) =>
                `${linkClass} ${
                  isActive
                    ? "bg-gray-300 border-l-4 border-black"
                    : "hover:bg-gray-400"
                }`
              }>
              Preset Configuration
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/history"
              className={({ isActive }) =>
                `${linkClass} ${
                  isActive
                    ? "bg-gray-300 border-l-4 border-black"
                    : "hover:bg-gray-400"
                }`
              }>
              History
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/accounts"
              className={({ isActive }) =>
                `${linkClass} ${
                  isActive
                    ? "bg-gray-300 border-l-4 border-black"
                    : "hover:bg-gray-400"
                }`
              }>
              Accounts
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/recovery-logs"
              className={({ isActive }) =>
                `${linkClass} ${
                  isActive
                    ? "bg-gray-300 border-l-4 border-black"
                    : "hover:bg-gray-400"
                }`
              }>
              Recovery Logs
            </NavLink>
          </li>
        </ul>

        {/* Spacer to push login/logout to bottom */}
        <div className="flex-grow"></div>

        {/* Login/Logout Button */}
        <div className="mb-4">
          <div className="flex flex-col space-y-2">
            <span className="px-1 py-2 text-gray-800">{username}</span>
            <button
              onClick={handleLogout}
              className="w-full px-4 py-2 bg-red-500 text-white text-sm rounded hover:bg-red-600 text-left mb-5">
              Logout
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdminNavigation;

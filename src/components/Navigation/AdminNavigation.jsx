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

  const linkClass = "block py-2 px-4 rounded w-80% text-left";

  return (
    <div className="w-20% bg-[#999696] text-black p-4 font-medium min-h-screen fixed top-0 left-0 h-full">
      <div className="flex flex-col h-full text-lg font-semibold">
        {/* Navigation Links */}
        <ul className="flex flex-col space-y-5 list-none">
          <li>
            <NavLink
              to="/history"
              className={({ isActive }) =>
                `${linkClass} ${
                  isActive
                    ? "bg-gray-300 border-l-4 border-black no-underline text-black"
                    : "hover:bg-gray-400 no-underline text-black"
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
                    ? "bg-gray-300 border-l-4 border-black no-underline text-black"
                    : "hover:bg-gray-400 no-underline text-black"
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
                    ? "bg-gray-300 border-l-4 border-black no-underline text-black"
                    : "hover:bg-gray-400 no-underline text-black"
                }`
              }>
              Recovery Logs
            </NavLink>
          </li>
        </ul>

        <div className="flex-grow"></div>

        <div className="mb-4">
          <div className="flex flex-col space-y-2 space-x-10 mb-10">
            <span className="px-10 py-2 text-gray-800">{username}</span>
            <button
              onClick={handleLogout}
              className="w-20 px-4 py-2 bg-red-500 text-white text-sm rounded hover:bg-red-600 hover:scale-105 cursor-pointer transition transform duration-200 text-left">
              Logout
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdminNavigation;

import React, { useEffect, useState } from "react";
import { NavLink, useNavigate, useLocation } from "react-router-dom";

const Navigation = () => {
  const navigate = useNavigate();
  const location = useLocation();
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
      // Optional: Redirect to login for protected routes
      // if (location.pathname !== "/" && location.pathname !== "/history") {
      //   navigate("/", { replace: true });
      // }
    }
  }, [location.pathname, navigate]);

  // Handle logout
  const handleLogout = () => {
    localStorage.removeItem("token");
    setIsLoggedIn(false);
    setUsername("");
    navigate("/", { replace: true });
  };

  // Determine text color based on route
  const isWhiteText = ["/", "/forecast", "/generate"].includes(
    location.pathname
  );

  const forecastRelatedPages = [
    "/select-forecast",
    "/generate",
    "/forecast-result",
    "/view-graph",
    "/single-model-config",
    "/hybrid-model-config",
    "/result",
  ];

  const settingsRelatedPages = ["/accounts", "/recovery-logs"];
  const isSettingsRelated = settingsRelatedPages.includes(location.pathname);
  const isHistoryRelated = ["/view-logs"].includes(location.pathname);
  const isForecastRelated = forecastRelatedPages.includes(location.pathname);

  const linkClass = "block py-2 px-4 rounded";

  return (
    <div className="fixed top-0 left-0 right-0 text-white p-4 font-medium bg-transparent z-10">
      <div className="flex justify-between items-center max-w-7xl mx-auto">
        {/* Navigation Links */}
        <ul
          className={`flex space-x-4 ${
            isWhiteText ? "text-white" : "text-black"
          }`}>
          <li>
            <NavLink
              to="/"
              className={({ isActive }) =>
                `${linkClass} ${
                  isActive
                    ? `border-b-2 pb-0 rounded-none ${
                        isActive && !isForecastRelated
                          ? "border-white"
                          : "border-black"
                      }`
                    : ""
                }`
              }>
              Dashboard
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/forecast"
              className={({ isActive }) =>
                `${linkClass} ${
                  isActive || isForecastRelated
                    ? `border-b-2 pb-0 rounded-none ${
                        (isActive && !isForecastRelated) || isWhiteText
                          ? "border-white"
                          : "border-black"
                      }`
                    : ""
                }`
              }>
              Forecast
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/history"
              className={({ isActive }) =>
                `${linkClass} ${
                  isActive || isHistoryRelated
                    ? `border-b-2 pb-0 rounded-none ${
                        isWhiteText ? "border-white" : "border-black"
                      }`
                    : ""
                }`
              }>
              History
            </NavLink>
          </li>
        </ul>
        <NavLink
          to="/login"
          className="px-3 py-1 bg-green-500 text-white text-xs rounded hover:bg-green-600">
          Admin
        </NavLink>
      </div>
    </div>
  );
};

export default Navigation;

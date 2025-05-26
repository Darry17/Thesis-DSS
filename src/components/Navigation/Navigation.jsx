import React, { useEffect, useState } from "react";
import { NavLink, useNavigate, useLocation } from "react-router-dom";

const Navigation = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [username, setUsername] = useState("");

  useEffect(() => {
    const token = localStorage.getItem("token")?.trim();
    if (token) {
      try {
        const payload = JSON.parse(atob(token.split(".")[1]));
        setUsername(payload.sub || "Admin");
        setIsLoggedIn(true);
      } catch (e) {
        console.error("Error decoding token:", e);
        localStorage.removeItem("token");
        setIsLoggedIn(false);
        setUsername("");
      }
    } else {
      setIsLoggedIn(false);
      setUsername("");
    }
  }, [location.pathname, navigate]);

  const handleLogout = () => {
    localStorage.removeItem("token");
    setIsLoggedIn(false);
    setUsername("");
    navigate("/", { replace: true });
  };

  const isWhiteText = ["/", "/upload", "/model-selection"].includes(
    location.pathname
  );

  const forecastRelatedPages = [
    "/upload",
    "/select-type",
    "/model-selection",
    "/view-graph",
    "/configure-single",
    "/configure-hybrid",
    "/result",
  ];

  const settingsRelatedPages = ["/accounts", "/recovery-logs"];
  const isSettingsRelated = settingsRelatedPages.includes(location.pathname);
  const isHistoryRelated = ["/view-logs"].includes(location.pathname);
  const isForecastRelated = forecastRelatedPages.includes(location.pathname);

  const linkClass = "block px-4 rounded";

  return (
    <div className="top-0 left-0 right-0 p-4 font-semibold">
      <div className="flex justify-between items-center text-lg">
        <ul
          className={`flex space-x-4 list-none ${
            isWhiteText ? "text-white" : "text-black"
          }`}>
          <li>
            <NavLink
              to="/"
              className={({ isActive }) =>
                `${linkClass} ${
                  isActive
                    ? `border-b-2 rounded-none underline-offset-8 ${
                        isWhiteText
                          ? "text-white border-white"
                          : "text-black border-black"
                      }`
                    : isWhiteText
                    ? "text-white no-underline"
                    : "text-black no-underline"
                }`
              }>
              Dashboard
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/upload"
              className={({ isActive }) =>
                `${linkClass} ${
                  isActive || isForecastRelated
                    ? `border-b-2 rounded-none underline-offset-8 ${
                        isWhiteText
                          ? "text-white border-white"
                          : "text-black border-black"
                      }`
                    : isWhiteText
                    ? "text-white no-underline"
                    : "text-black no-underline"
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
                    ? `border-b-2 rounded-none underline-offset-8 ${
                        isWhiteText
                          ? "text-white border-white"
                          : "text-black border-black"
                      }`
                    : isWhiteText
                    ? "text-white no-underline"
                    : "text-black no-underline"
                }`
              }>
              History
            </NavLink>
          </li>
        </ul>
        <NavLink
          to="/login"
          className="px-3 py-1 bg-green-500 border-none text-white text-xs rounded hover:bg-green-600 no-underline">
          Admin
        </NavLink>
      </div>
    </div>
  );
};

export default Navigation;

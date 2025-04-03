import React from "react";
import { NavLink, useNavigate, useLocation } from "react-router-dom";

const Navigation = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [userRole, setUserRole] = React.useState("USER");

  // Effect to decode user role from token (example logic)
  React.useEffect(() => {
    const token = localStorage.getItem("token");
    if (token) {
      try {
        const payload = JSON.parse(atob(token.split(".")[1]));
        setUserRole(payload.access_control || "USER");
      } catch (e) {
        console.error("Error decoding token:", e);
      }
    }
  }, []);

  // Logout handler
  const handleLogout = () => {
    localStorage.removeItem("token");
    navigate("/");
  };

  // Determine if the current route requires black text
  const isWhiteText = ["/dashboard", "/forecast"].includes(location.pathname);

  const forecastRelatedPages = [
    "/select-forecast",
    "/generate",
    "/forecast-result",
    "/view-graph",
  ];

  // Check if the current page is one of the forecast-related pages
  const isForecastRelated = forecastRelatedPages.includes(location.pathname);

  // Base styling for all navigation links
  const linkClass = "block py-2 px-4 rounded";

  return (
    <div className="top-0 left-0 right-0 text-white p-4 font-bold">
      <div className="flex justify-between items-center">
        <ul
          className={`flex space-x-6 ${
            isWhiteText ? "text-white" : "text-black"
          }`}>
          <li>
            <NavLink
              to="/dashboard"
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
                        isActive && !isForecastRelated
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
                  isActive
                    ? `border-b-2 pb-0 rounded-none ${
                        isActive && !isForecastRelated
                          ? "border-black"
                          : "border-white"
                      }`
                    : ""
                }`
              }>
              History
            </NavLink>
          </li>
          {userRole !== "USER" && (
            <li>
              <NavLink
                to="/settings"
                className={({ isActive }) =>
                  `${linkClass} ${
                    isActive
                      ? `border-b-2 pb-0 rounded-none ${
                          isActive && !isForecastRelated
                            ? "border-black"
                            : "border-white"
                        }`
                      : ""
                  }`
                }>
                Settings
              </NavLink>
            </li>
          )}
        </ul>
        <NavLink
          onClick={handleLogout}
          className={`flex space-x-6 ${
            isWhiteText ? "text-white" : "text-black"
          }`}>
          Logout
        </NavLink>
      </div>
    </div>
  );
};

export default Navigation;

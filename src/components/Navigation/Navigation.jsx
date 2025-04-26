import React, { useEffect, useState } from "react";
import { NavLink, useNavigate, useLocation } from "react-router-dom";

const Navigation = () => {
  const navigate = useNavigate();
  const location = useLocation();
  // const [userRole, setUserRole] = useState("USER");
  // const [username, setUsername] = useState("");

  // // Effect to decode user role and username from token
  // useEffect(() => {
  //   const token = localStorage.getItem("token");
  //   if (token) {
  //     try {
  //       const payload = JSON.parse(atob(token.split(".")[1]));
  //       setUserRole(payload.access_control || "USER");
  //       setUsername(payload.sub || "User");
  //     } catch (e) {
  //       console.error("Error decoding token:", e);
  //       setUsername("User");
  //     }
  //   } else {
  //     // If no token, redirect to login
  //     if (location.pathname !== "/") {
  //       navigate("/", { replace: true });
  //     }
  //   }
  // }, [location.pathname, navigate]);

  // Logout handler
  // const handleLogout = () => {
  //   localStorage.removeItem("token");
  //   setUserRole("USER");
  //   setUsername("");
  //   navigate("/", { replace: true }); // Replace history entry
  // };

  // Determine if the current route requires white text
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
    <div className="top-0 left-0 right-0 text-white p-4 font-medium">
      <div className="flex justify-between items-center">
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
          {/* {userRole !== "USER" && ( */}
          <li>
            <NavLink
              to="/settings"
              className={({ isActive }) =>
                `${linkClass} ${
                  isActive || isSettingsRelated
                    ? `border-b-2 pb-0 rounded-none ${
                        isWhiteText ? "border-white" : "border-black"
                      }`
                    : ""
                }`
              }>
              Settings
            </NavLink>
          </li>
          {/* )} */}
        </ul>
        <div
          className={`flex space-x-10 items-center ${
            isWhiteText ? "text-white" : "text-black"
          }`}>
          {/* <span>{username}</span> */}
          <button
            // onClick={handleLogout}
            className={`cursor-pointer ${
              isWhiteText ? "text-white" : "text-black"
            }`}>
            Logout
          </button>
        </div>
      </div>
    </div>
  );
};

export default Navigation;

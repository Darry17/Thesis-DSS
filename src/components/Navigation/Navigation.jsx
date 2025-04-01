import React from "react";
import { NavLink, useNavigate } from "react-router-dom";

const Navigation = () => {
  const navigate = useNavigate();
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

  // Base styling for all navigation links
  const linkClass = "block py-2 px-4 rounded";

  return (
    <div className="top-0 left-0 right-0 text-white p-4">
      <div className="flex justify-between items-center">
        <ul className="flex space-x-6">
          <li>
            <NavLink
              to="/dashboard"
              className={({ isActive }) =>
                `${linkClass} ${
                  isActive ? "border-b-2 pb-0 rounded-none border-white" : ""
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
                  isActive ? "border-b-2 pb-0 rounded-none border-white" : ""
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
                  isActive ? "border-b-2 pb-0 rounded-none border-white" : ""
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
                    isActive ? "border-b-2 pb-0 rounded-none border-white" : ""
                  }`
                }>
                Settings
              </NavLink>
            </li>
          )}
        </ul>
        <button onClick={handleLogout} className="py-2 px-4">
          Logout
        </button>
      </div>
    </div>
  );
};

export default Navigation;

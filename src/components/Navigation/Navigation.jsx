import React from "react";
import { Link, useNavigate } from "react-router-dom";

const Navigation = () => {
  const navigate = useNavigate();
  const [userRole, setUserRole] = React.useState("USER"); // Default to USER

  // Get user role on component mount
  React.useEffect(() => {
    const token = localStorage.getItem("token");
    if (token) {
      // Decode the JWT token to get user info
      try {
        const payload = JSON.parse(atob(token.split(".")[1]));
        setUserRole(payload.access_control || "USER");
      } catch (e) {
        console.error("Error decoding token:", e);
      }
    }
  }, []);

  const handleLogout = () => {
    localStorage.removeItem("token");
    navigate("/");
  };

  return (
    <div className="bg-gray-800 text-white p-4">
      <div className="flex justify-between items-center">
        {/* Left-aligned navigation links */}
        <ul className="flex space-x-6">
          <li>
            <Link
              to="/dashboard"
              className="block py-2 px-4 hover:bg-gray-700 rounded">
              Dashboard
            </Link>
          </li>
          <li>
            <Link
              to="/forecast"
              className="block py-2 px-4 hover:bg-gray-700 rounded">
              Forecast
            </Link>
          </li>
          <li>
            <Link
              to="/history"
              className="block py-2 px-4 hover:bg-gray-700 rounded">
              History
            </Link>
          </li>
          {/* Only show Settings for non-USER roles */}
          {userRole !== "USER" && (
            <li>
              <Link
                to="/settings"
                className="block py-2 px-4 hover:bg-gray-700 rounded">
                Settings
              </Link>
            </li>
          )}
        </ul>

        {/* Right-aligned logout button */}
        <button onClick={handleLogout} className="py-2 px-4">
          Logout
        </button>
      </div>
    </div>
  );
};

export default Navigation;

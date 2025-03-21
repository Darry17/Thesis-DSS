// Navigation.jsx (same as Sidebar.jsx, now serving as the horizontal navigation)
import React from "react";
import { Link } from "react-router-dom"; // For navigation

const Navigation = () => {
  return (
    <div className="bg-gray-800 text-white p-4">
      <ul className="flex space-x-6">
        {" "}
        {/* Using flex for horizontal alignment */}
        <li>
          <Link to="/" className="block py-2 px-4 hover:bg-gray-700 rounded">
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
        {/* Add other links as needed */}
      </ul>
    </div>
  );
};

export default Navigation;

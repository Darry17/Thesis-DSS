import React from "react";
import { Link } from "react-router-dom";

const Navigation = () => {
  return (
    <div className="bg-gray-800 text-white p-4">
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
        <li>
          <Link
            to="/settings"
            className="block py-2 px-4 hover:bg-gray-700 rounded">
            Settings
          </Link>
        </li>
      </ul>
    </div>
  );
};

export default Navigation;

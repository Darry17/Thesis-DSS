import React from "react";
import { NavLink } from "react-router-dom";

const Settings = () => {
  return (
    <div className="min-h-screen relative">
      {/* Background Layer */}
      <div className="fixed inset-0 bg-gray-100" style={{ zIndex: -1 }} />

      {/* Content */}
      <div className="relative z-10 flex flex-col items-center font-sans flex-1 min-h-screen">
        <div className="bg-white rounded-lg shadow-lg p-20 mt-12 w-140 text-center font-bold">
          <h1 className="text-4xl font-bold mb-5">Settings</h1>
          <div className="mb-5">
            <h3 className="text-sm text-black text-left mb-2">
              Edit Preset Configuration
            </h3>
            <div className="flex gap-5">
              <NavLink className="bg-gray-200 text-gray-700 px-6 py-2 rounded-full hover:bg-gray-300 transition">
                DHR only
              </NavLink>
              <NavLink className="bg-gray-200 text-gray-700 px-6 py-2 rounded-full hover:bg-gray-300 transition">
                ESN only
              </NavLink>
              <NavLink className="bg-gray-200 text-gray-700 px-6 py-2 rounded-full hover:bg-gray-300 transition">
                DHR-ESN
              </NavLink>
            </div>
          </div>
          <div>
            <NavLink
              to="/accounts"
              className="block bg-gray-200 text-gray-700 px-4 py-2 rounded-full hover:bg-gray-300 transition">
              Accounts
            </NavLink>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Settings;

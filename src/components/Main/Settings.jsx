import React, { useState, useEffect } from "react";
import { NavLink } from "react-router-dom";

const Settings = () => {
  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center font-sans">
      {/* Settings Content */}
      <div className="bg-white rounded-lg shadow-lg p-6 mt-12 w-96 text-center font-bold">
        <h1 className="text-2xl font-bold mb-5">Settings</h1>
        <div className="mb-5">
          <h3 className="text-sm text-black text-left mb-2">
            Edit Preset Configuration
          </h3>
          <div className="flex gap-3 justify-center">
            <NavLink className="bg-gray-200 text-gray-700 px-4 py-2 rounded-full hover:bg-gray-300 transition">
              DHR only
            </NavLink>
            <NavLink className="bg-gray-200 text-gray-700 px-4 py-2 rounded-full hover:bg-gray-300 transition">
              ESN only
            </NavLink>
            <NavLink className="bg-gray-200 text-gray-700 px-4 py-2 rounded-full hover:bg-gray-300 transition">
              DHR-ESN
            </NavLink>
          </div>
        </div>
        <div>
          <NavLink
            to="/accounts"
            className="block w-full bg-gray-200 text-gray-700 px-4 py-2 rounded-full hover:bg-gray-300 transition">
            Accounts
          </NavLink>
        </div>
      </div>
    </div>
  );
};

export default Settings;

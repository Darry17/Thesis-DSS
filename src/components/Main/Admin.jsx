import React from "react";
import { NavLink } from "react-router-dom";

const Admin = () => {
  return (
    <div className="p-6 bg-gray-100 min-h-screen">
      <div className="max-w-5xl mx-auto font-bold">
        <h1 className="text-4xl font-bold mb-10">Preset Configuration</h1>
        <div className="flex gap-5">
          <NavLink className="bg-gray-200 text-gray-700 px-6 py-2 rounded-2xl hover:bg-gray-300 transition">
            DHR only
          </NavLink>
          <NavLink className="bg-gray-200 text-gray-700 px-6 py-2 rounded-2xl hover:bg-gray-300 transition">
            ESN only
          </NavLink>
          <NavLink className="bg-gray-200 text-gray-700 px-6 py-2 rounded-2xl hover:bg-gray-300 transition">
            DHR-ESN
          </NavLink>
        </div>
      </div>
    </div>
  );
};

export default Admin;

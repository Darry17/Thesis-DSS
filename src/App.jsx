import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navigation from "@/components/Navigation/Navigation";
import {
  Dashboard,
  History,
  Forecast,
  SelectForecast,
  GenerateForecast,
  SingleModelConfiguration,
  HybridModelConfiguration,
  ForecastResult,
  ViewGraph,
} from "@/components/Main";

const App = () => {
  return (
    <Router>
      <Navigation />
      <Routes>
        {/* Main routes */}
        <Route path="/" element={<Dashboard />} />
        <Route path="/forecast" element={<Forecast />} />
        <Route path="/history" element={<History />} />

        {/* Forecast and model configuration routes */}
        <Route path="/select-forecast" element={<SelectForecast />} />
        <Route path="/generate" element={<GenerateForecast />} />
        <Route
          path="/single-model-config"
          element={<SingleModelConfiguration />}
        />
        <Route
          path="/hybrid-model-config"
          element={<HybridModelConfiguration />}
        />
        <Route path="/result" element={<ForecastResult />} />
        <Route path="/view-graph" element={<ViewGraph />} />
      </Routes>
    </Router>
  );
};

export default App;

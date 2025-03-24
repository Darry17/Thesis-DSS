import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navigation from "@/components/Navigation/Navigation";
import {
  Dashboard,
  History,
  Forecast,
  ModelOption,
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
        <Route path="/ModelOption" element={<ModelOption />} />
        <Route path="/SelectForecast" element={<SelectForecast />} />
        <Route path="/GenerateForecast" element={<GenerateForecast />} />
        <Route
          path="/SingleModelConfiguration"
          element={<SingleModelConfiguration />}
        />
        <Route
          path="/HybridModelConfiguration"
          element={<HybridModelConfiguration />}
        />
        <Route path="/ForecastResult" element={<ForecastResult />} />
        <Route path="/ViewGraph" element={<ViewGraph />} />
      </Routes>
    </Router>
  );
};

export default App;

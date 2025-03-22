import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navigation from "@/components/Navigation/Navigation";
import { Dashboard, History } from "@/components/Main";
import {
  Forecast,
  ModelOption,
  SelectForecast,
  GenerateForecast,
  SingleModelConfiguration,
  HybridModelConfiguration,
  ForecastResult,
} from "./components/Main";

const App = () => {
  return (
    <Router>
      <Navigation />
      <Routes>
        {/* Main routes */}
        <Route path="/" element={<Dashboard />} />
        <Route path="/forecast" element={<Forecast />} />
        <Route path="/history" element={<History />} />
        <Route
          path="/SingleModelConfiguration"
          element={<SingleModelConfiguration />}
        />
        <Route
          path="/HybridModelConfiguration"
          element={<HybridModelConfiguration />}
        />
        <Route path="/ForecastResult" element={<ForecastResult />} />
        {/* Forecast and model configuration routes */}
        <Route path="/ModelOption" element={<ModelOption />} />
        <Route path="/SelectForecast" element={<SelectForecast />} />
        <Route path="/GenerateForecast" element={<GenerateForecast />} />
      </Routes>
    </Router>
  );
};

export default App;

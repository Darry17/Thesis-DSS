import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navigation from "@/components/Navigation/Navigation";
import {
  Dashboard,
  History,
  Forecast,
  Settings,
  SelectForecast,
  GenerateForecast,
  SingleModelConfiguration,
  HybridModelConfiguration,
  ForecastResult,
  ViewGraph,
  ViewLogs,
  Account,
  RecoveryLogs,
} from "@/components/Main";

const AppContent = () => {
  return (
    <>
      <Navigation />
      <Routes>
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
        <Route path="/view-logs" element={<ViewLogs />} />
        {/* Admin Routes */}
        <Route path="/settings" element={<Settings />} />
        <Route path="/accounts" element={<Account />} />
        <Route path="/recovery-logs" element={<RecoveryLogs />} />
      </Routes>
    </>
  );
};

const App = () => {
  return (
    <Router>
      <AppContent />
    </Router>
  );
};

export default App;

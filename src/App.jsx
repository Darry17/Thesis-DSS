import React from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  useLocation,
} from "react-router-dom";
import Navigation from "@/components/Navigation/Navigation";
import {
  Login,
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
} from "@/components/Main";

// Wrapper component that conditionally renders Navigation
const AppContent = () => {
  const location = useLocation();
  const isLoginPage = location.pathname === "/";

  return (
    <>
      {!isLoginPage && <Navigation />}
      <Routes>
        {/* Main routes */}
        <Route path="/" element={<Login />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/forecast" element={<Forecast />} />
        <Route path="/history" element={<History />} />
        <Route path="/settings" element={<Settings />} />

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

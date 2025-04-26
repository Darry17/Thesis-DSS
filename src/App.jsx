import React from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  useLocation,
} from "react-router-dom";
import Navigation from "@/components/Navigation/Navigation";
import ProtectedRoute from "@/components/Main/Auth/ProtectedRoute";
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
  Account,
  RecoveryLogs,
} from "@/components/Main";

// Wrapper component that conditionally renders Navigation
const AppContent = () => {
  const location = useLocation();
  const isLoginPage = location.pathname === "/";

  return (
    <>
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

      {!isLoginPage && <Navigation />}
      <Routes>
        {/* Protected routes */}
        <Route element={<ProtectedRoute />}>
          <Route path="/settings" element={<Settings />} />
          <Route path="/accounts" element={<Account />} />
          <Route path="/recovery-logs" element={<RecoveryLogs />} />
        </Route>
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

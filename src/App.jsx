import React from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  useLocation,
} from "react-router-dom";
import Navigation from "@/components/Navigation/Navigation";
import AdminNavigation from "@/components/Navigation/AdminNavigation";
import {
  Dashboard,
  History,
  Forecast,
  Admin,
  SelectForecast,
  GenerateForecast,
  SingleModelConfiguration,
  HybridModelConfiguration,
  ForecastResult,
  ViewGraph,
  ViewLogs,
  Account,
  RecoveryLogs,
  Login,
} from "@/components/Main";

// Component to handle conditional navigation rendering
const AppContent = () => {
  const location = useLocation();

  // Define admin-related routes
  const adminRoutes = ["/admin", "/accounts", "/recovery-logs"];
  const isAdminRoute = adminRoutes.some((route) =>
    location.pathname.startsWith(route)
  );

  return (
    <>
      {isAdminRoute ? <AdminNavigation /> : <Navigation />}
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/forecast" element={<Forecast />} />
        <Route path="/history" element={<History />} />
        <Route path="/login" element={<Login />} />
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
        <Route path="/admin" element={<Admin />} />
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

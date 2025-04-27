import React from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  useLocation,
} from "react-router-dom";
import ProtectedRoute from "@/components/Main/Auth/ProtectedRoute";
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

const AppContent = () => {
  const location = useLocation();
  const adminRoutes = ["/admin", "/accounts", "/recovery-logs"];
  const isAdminRoute = adminRoutes.some((route) =>
    location.pathname.startsWith(route)
  );

  return (
    <div className="flex min-h-screen">
      {/* Sidebar */}
      {isAdminRoute ? <AdminNavigation /> : <Navigation />}
      {/* Main Content */}
      <div className="flex-1 p-4">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/forecast" element={<Forecast />} />
          <Route path="/history" element={<History />} />
          <Route path="/login" element={<Login />} />
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

          {/* Protected Routes for Admin */}
          <Route element={<ProtectedRoute />}>
            <Route path="/admin" element={<Admin />} />
            <Route path="/accounts" element={<Account />} />
            <Route path="/recovery-logs" element={<RecoveryLogs />} />
          </Route>
        </Routes>
      </div>
    </div>
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

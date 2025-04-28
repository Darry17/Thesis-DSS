import React, { useEffect } from "react";
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
  const isLoggedIn = !!localStorage.getItem("token");
  const adminRoutes = ["/admin", "/accounts", "/recovery-logs"];

  // Check if path is admin OR if it's /history AND logged in
  const isAdminRoute =
    adminRoutes.some((route) => location.pathname.startsWith(route)) ||
    (location.pathname === "/history" && isLoggedIn);

  useEffect(() => {
    const noOverflowRoutes = [
      "/",
      "/forecast",
      "/select-forecast",
      "/generate",
      "/single-model-config",
      "/hybrid-model-config",
      "/history",
      "/login",
    ];
    const isNoOverflow = noOverflowRoutes.includes(location.pathname);

    document.body.classList.toggle("no-overflow", isNoOverflow);

    return () => {
      document.body.classList.remove("no-overflow");
    };
  }, [location.pathname]);

  return (
    <div className="min-h-screen flex flex-col">
      {/* Navigation for non-admin routes */}
      {!isAdminRoute && <Navigation />}
      <div className="flex flex-1">
        {/* Sidebar for admin routes */}
        {isAdminRoute && <AdminNavigation />}
        {/* Main Content */}
        <div className={isAdminRoute ? "w-full" : "w-full"}>
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
            <Route element={<ProtectedRoute />}>
              <Route path="/admin" element={<Admin />} />
              <Route path="/accounts" element={<Account />} />
              <Route path="/recovery-logs" element={<RecoveryLogs />} />
            </Route>
          </Routes>
        </div>
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

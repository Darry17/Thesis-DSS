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

  const isAdminRoute =
    adminRoutes.some((route) => location.pathname.startsWith(route)) ||
    (location.pathname === "/history" && isLoggedIn);

  useEffect(() => {
    const noOverflowRoutes = [
      "/",
      "/forecast",
      "/select-forecast",
      "/login",
      "/admin",
      "/accounts",
      "/recovery-logs",
    ];

    const responsiveOverflowRoutes = [
      "/history",
      "/generate",
      "/single-model-config",
      "/hybrid-model-config",
    ];

    const handleOverflow = () => {
      const isResponsiveRoute = responsiveOverflowRoutes.includes(
        location.pathname
      );
      let isNoOverflow = false;

      // Target the main content area (excluding Navigation and AdminNavigation)
      const mainContent = document.querySelector(".main-content");
      const viewportHeight = window.innerHeight; // Use window.innerHeight for reliability

      if (isResponsiveRoute) {
        if (mainContent) {
          // Measure the main content height
          const contentHeight = mainContent.scrollHeight;
          isNoOverflow = contentHeight <= viewportHeight;
        } else {
          // Fallback to body height
          const bodyHeight = document.body.scrollHeight;
          isNoOverflow = bodyHeight <= viewportHeight;
        }
      } else {
        // Apply overflow: hidden for noOverflowRoutes
        isNoOverflow = noOverflowRoutes.includes(location.pathname);
      }

      // Toggle the no-overflow class on body
      document.body.classList.toggle("no-overflow", isNoOverflow);
    };

    // Run initially with a shorter delay and on next tick
    setTimeout(handleOverflow, 100); // Reduced to 100ms
    window.addEventListener("resize", handleOverflow);

    // Cleanup on route change or unmount
    return () => {
      document.body.classList.remove("no-overflow");
      window.removeEventListener("resize", handleOverflow);
    };
  }, [location.pathname, isLoggedIn]);

  return (
    <div className="min-h-screen flex flex-col">
      {!isAdminRoute && <Navigation />}
      <div className="flex flex-1 pt-16">
        {isAdminRoute && <AdminNavigation />}
        <div
          className={
            isAdminRoute
              ? "w-full flex-1 overflow-y-auto main-content pl-64"
              : "w-full flex-1 overflow-y-auto main-content"
          }>
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

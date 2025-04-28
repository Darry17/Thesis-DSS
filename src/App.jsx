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
      "/generate",
      "/select-forecast",
      "/single-model-config",
      "/hybrid-model-config",
      "/login",
      "/admin",
      "/accounts",
      "/recovery-logs",
    ];

    const responsiveOverflowRoutes = [
      "/history",
      "/generate",
      "single-model-config",
      "/hybrid-model-config",
    ];

    const handleOverflow = () => {
      const isResponsiveRoute = responsiveOverflowRoutes.includes(
        location.pathname
      );
      let isNoOverflow = false;

      // Target the root container and main content area
      const rootContainer = document.querySelector(
        ".min-h-screen.flex.flex-col"
      );
      const mainContent = document.querySelector(".main-content");
      const viewportHeight = document.documentElement.clientHeight;

      if (isResponsiveRoute) {
        if (mainContent && isAdminRoute) {
          // In admin view, measure the main content height (excluding AdminNavigation)
          const contentHeight = mainContent.scrollHeight;
          isNoOverflow = contentHeight <= viewportHeight;
          console.log(
            `Route: ${location.pathname}, Admin View, Viewport Height: ${viewportHeight}, Main Content Height: ${contentHeight}, Is No Overflow: ${isNoOverflow}`
          );
        } else if (rootContainer) {
          // In non-admin view or as fallback, measure the root container
          const contentHeight = rootContainer.scrollHeight;
          isNoOverflow = contentHeight <= viewportHeight;
          console.log(
            `Route: ${location.pathname}, Non-Admin View, Viewport Height: ${viewportHeight}, Root Content Height: ${contentHeight}, Is No Overflow: ${isNoOverflow}`
          );
        } else {
          // Fallback to body height
          const bodyHeight = document.body.scrollHeight;
          isNoOverflow = bodyHeight <= viewportHeight;
          console.log(
            `Route: ${location.pathname}, Fallback, Viewport Height: ${viewportHeight}, Body Height: ${bodyHeight}, Is No Overflow: ${isNoOverflow}`
          );
        }
      } else {
        // Apply overflow: hidden for other noOverflowRoutes
        isNoOverflow = noOverflowRoutes.includes(location.pathname);
      }

      // Toggle the no-overflow class on body
      document.body.classList.toggle("no-overflow", isNoOverflow);
    };

    // Run initially with a delay to ensure DOM is rendered
    setTimeout(handleOverflow, 300); // Increased to 300ms for safety
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
      <div className="flex flex-1">
        {isAdminRoute && <AdminNavigation />}
        <div
          className={
            isAdminRoute
              ? "w-full flex-1 overflow-y-auto main-content"
              : "w-full main-content"
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

import React, { useEffect } from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  useLocation,
} from "react-router-dom";
import ProtectedRoute from "./components/Main/Auth/ProtectedRoute";
import Navigation from "./components/Navigation/Navigation";
import AdminNavigation from "./components/Navigation/AdminNavigation";

import {
  ForecastResult,
  SelectTypePage,
  ConfigureSingle,
  Dashboard,
  FileUpload,
  Account,
  RecoveryLogs,
  Login,
  ModelSelection,
  History,
  ViewGraph,
  ViewLogs,
} from "@/components/Main";
import { ConfigureHybrid } from "./components/Main";

const AppContent = () => {
  const location = useLocation();
  const isLoggedIn = !!localStorage.getItem("token");
  const adminRoutes = ["/admin", "/accounts", "/recovery-logs"];

  const isAdminRoute =
    adminRoutes.some((route) => location.pathname.startsWith(route)) ||
    ((location.pathname === "/history" || location.pathname === "/view-logs") &&
      isLoggedIn);

  useEffect(() => {
    const noOverflowRoutes = [
      "/",
      "/upload",
      "/generate",
      "/select-forecast",
      "/configure-single",
      "/configure-hybrid",
      "/login",
      "/admin",
      "/accounts",
      "/recovery-logs",
    ];

    const responsiveOverflowRoutes = [
      "/history",
      "/view-logs",
      "/generate",
      "/configure-single",
      "/configure-hybrid",
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
        } else if (rootContainer) {
          // In non-admin view or as fallback, measure the root container
          const contentHeight = rootContainer.scrollHeight;
          isNoOverflow = contentHeight <= viewportHeight;
        } else {
          // Fallback to body height
          const bodyHeight = document.body.scrollHeight;
          isNoOverflow = bodyHeight <= viewportHeight;
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
              ? "w-full flex-1 overflow-y-auto main-content pl-64"
              : "w-full flex-1 overflow-y-hidden main-content"
          }>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/upload" element={<FileUpload />} />
            <Route path="/select-type" element={<SelectTypePage />} />
            <Route path="/model-selection" element={<ModelSelection />} />
            <Route path="/configure-single" element={<ConfigureSingle />} />
            <Route path="/configure-hybrid" element={<ConfigureHybrid />} />
            <Route path="/result" element={<ForecastResult />} />
            <Route path="/view-graph" element={<ViewGraph />} />
            <Route path="/view-logs" element={<ViewLogs />} />
            <Route path="/login" element={<Login />} />
            <Route path="/history" element={<History />} />

            <Route element={<ProtectedRoute />}>
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

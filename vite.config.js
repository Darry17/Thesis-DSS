import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": "/src", // Map '@' to '/src' directory
    },
  },
  worker: {
    format: "es",
  },
  optimizeDeps: {
    include: ["csv-parse/browser/esm/sync"],
  },
});

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import UnoCSS from "unocss/vite";

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    UnoCSS({
      safelist: ["text-white", "text-black", "border-white", "border-black"],
    }),
  ],
  resolve: {
    alias: {
      "@": "/src",
    },
  },
});

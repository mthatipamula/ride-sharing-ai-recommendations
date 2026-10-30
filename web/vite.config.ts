import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5180,
    proxy: { "/api": { target: "http://localhost:8090", rewrite: (p) => p.replace(/^\/api/, "") } },
  },
});

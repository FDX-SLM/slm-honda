import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base "./" để asset nạp được khi FastAPI phục vụ dist trên cùng cổng.
// Dev: proxy /api → backend FastAPI (mặc định 127.0.0.1:8600).
export default defineConfig({
  base: "./",
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:8600", changeOrigin: true },
    },
  },
});

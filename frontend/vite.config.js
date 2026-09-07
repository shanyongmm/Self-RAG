import { fileURLToPath, URL } from "node:url";

import vue from "@vitejs/plugin-vue";
import { defineConfig } from "vite";

// 后端默认跑在 127.0.0.1:8001。
// 开发模式:前端(5173)通过下面的代理直接访问 FastAPI,无需处理 CORS。
const BACKEND_TARGET = process.env.VITE_BACKEND_TARGET || "http://127.0.0.1:8001";

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/health": BACKEND_TARGET,
      "/ask": BACKEND_TARGET,
      "/ask-naive": BACKEND_TARGET,
      "/auth": BACKEND_TARGET,
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    chunkSizeWarningLimit: 700,
  },
});

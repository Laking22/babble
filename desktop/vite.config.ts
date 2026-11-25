import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base "./": the same build works at http://127.0.0.1:7895/app/ and inside the desktop shell (Tauri)
export default defineConfig({
  plugins: [react()],
  base: "./",
  server: { port: 1420, strictPort: true },
  build: { target: "es2022", chunkSizeWarningLimit: 900 },
});

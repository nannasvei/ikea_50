import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  base: "/ikea_50/",
  plugins: [react()],
  build: {
    outDir: "pages-dist",
    emptyOutDir: true
  }
});

import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The build output is a single HTML file. `scripts/emit-single.mjs` inlines the
// hashed JS and CSS into `src/rlvr/web/index.html`, so the Python server can
// serve the page with no asset routes and no network access.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    outDir: 'dist',
    cssCodeSplit: false,
    assetsInlineLimit: 1024 * 1024,
    modulePreload: false,
    reportCompressedSize: false,
  },
})

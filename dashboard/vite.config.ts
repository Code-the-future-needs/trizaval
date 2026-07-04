import { defineConfig, type PluginOption } from 'vite'
import react from '@vitejs/plugin-react'

import wasmPlugin from 'vite-plugin-wasm'
const wasm = wasmPlugin as unknown as () => PluginOption

// https://vite.dev/config/
export default defineConfig({
  plugins: [wasm(), react()],
  build: {
    target: 'esnext',
  },
})
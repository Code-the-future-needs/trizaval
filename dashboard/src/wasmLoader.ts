
type WasmExports = Record<string, unknown>

let wasmPromise: Promise<WasmExports> | null = null

export function loadWasm(): Promise<WasmExports> {
  if (!wasmPromise) {
    wasmPromise = (async () => {
      const wasmModule = (await import(/* @vite-ignore */ '../wasm/trizaval_wasm.js')) as {
        default: (path?: unknown) => Promise<unknown>
      } & WasmExports
      await wasmModule.default()
      return wasmModule
    })()
  }
  return wasmPromise
}
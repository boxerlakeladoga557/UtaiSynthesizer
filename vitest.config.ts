import { defineConfig } from "vitest/config";

// Frontend unit tests (S48 Phase 3+). Node env — the store/history/bundle modules are driven headless;
// Tauri `invoke` and i18n are mocked per-test. jsdom is not needed (no rendering).
export default defineConfig({
  test: {
    environment: "node",
    globals: true,
    include: ["src/**/*.test.ts"],
  },
});

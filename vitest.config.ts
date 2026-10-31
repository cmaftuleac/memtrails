import { defineConfig } from 'vitest/config';

export default defineConfig({ test: { include: ['src/**/*.test.ts'], setupFiles: ['./vitest.setup.ts'], exclude: ['**/node_modules/**', '**/*.integration.test.ts', '**/*.live.test.ts'] } });

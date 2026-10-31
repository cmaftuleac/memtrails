import { defineConfig } from 'vitest/config';

// Against the real embedding model and a real Qdrant (fetched into the cache, or MEMTRAILS_MODEL_DIR / QDRANT_BINARY).
export default defineConfig({ test: { include: ['src/**/*.integration.test.ts'] } });

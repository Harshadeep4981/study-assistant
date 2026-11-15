import react from '@vitejs/plugin-react'

export default {
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: './src/test/setup.js',
    clearMocks: true,
    testTimeout: 15000,
  },
}

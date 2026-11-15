/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    colors: {
      transparent: 'transparent',
      current: 'currentColor',
      deep: 'var(--color-deep)',
      forest: 'var(--color-forest)',
      gold: 'var(--color-gold)',
      'gold-ink': 'var(--color-gold-ink)',
      sage: 'var(--color-sage)',
      ivory: 'var(--color-ivory)',
      paper: 'var(--color-paper)',
      ink: 'var(--color-ink)',
      muted: 'var(--color-muted)',
      line: 'var(--color-line)',
      moss: 'var(--color-moss)',
      success: 'var(--color-success)',
      error: 'var(--color-error)',
    },
    extend: {
      boxShadow: {
        none: 'none',
      },
    },
  },
  plugins: [],
}

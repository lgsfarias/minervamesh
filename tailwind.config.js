/** @type {import('tailwindcss').Config} */
module.exports = {
  // Os .py entram porque alguns módulos montam HTML com classes Tailwind em strings Python.
  content: ["./index.html", "./simulations/**/*.{html,py}", "./assets/app.js"],
  // Tema escuro por atributo (assets/theme.js escreve <html data-theme>), não pelo sistema.
  darkMode: ["selector", '[data-theme="dark"]'],
  theme: {
    extend: {
      // Cores da v1.2.0: variáveis CSS definidas em assets/tailwind.src.css (claro e escuro).
      // Uso: bg-mm-surface, text-mm-muted, border-mm-border, text-mm-accent...
      colors: {
        mm: {
          bg: "var(--mm-bg)",
          surface: "var(--mm-surface)",
          surface2: "var(--mm-surface2)",
          text: "var(--mm-text)",
          muted: "var(--mm-muted)",
          border: "var(--mm-border)",
          accent: "var(--mm-accent)",
          "accent-strong": "var(--mm-accent-strong)",
          "accent-soft": "var(--mm-accent-soft)",
          "on-accent": "var(--mm-on-accent)",
          ok: "var(--mm-ok)",
          "ok-strong": "var(--mm-ok-strong)",
          "ok-soft": "var(--mm-ok-soft)",
          danger: "var(--mm-danger)",
          "danger-strong": "var(--mm-danger-strong)",
          "fem-bg": "var(--mm-fem-bg)",
          "fem-fg": "var(--mm-fem-fg)",
          "fdm-bg": "var(--mm-fdm-bg)",
          "fdm-fg": "var(--mm-fdm-fg)",
          plate: "var(--mm-plate)",
        },
      },
      fontFamily: {
        sans: ['"IBM Plex Sans"', "system-ui", "-apple-system", '"Segoe UI"', "Helvetica", "Arial", "sans-serif"],
        mono: ['"IBM Plex Mono"', "ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"],
      },
    },
  },
  plugins: [],
};

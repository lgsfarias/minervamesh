/** @type {import('tailwindcss').Config} */
module.exports = {
  // Os .py entram porque alguns módulos montam HTML com classes Tailwind em strings Python.
  content: ["./index.html", "./simulations/**/*.{html,py}"],
  theme: { extend: {} },
  plugins: [],
};

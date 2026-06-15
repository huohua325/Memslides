import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        studio: {
          ink: "#020617",
          teal: "#0d9488",
          cyan: "#06b6d4",
        },
      },
      boxShadow: {
        studio: "0 18px 45px rgba(15, 23, 42, 0.10)",
      },
      borderRadius: {
        studio: "14px",
      },
    },
  },
  plugins: [],
} satisfies Config;

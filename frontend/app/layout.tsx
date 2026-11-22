import type { Metadata, Viewport } from "next";
import { Inter, Libre_Caslon_Text } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-ui", display: "swap" });
const caslon = Libre_Caslon_Text({ subsets: ["latin"], weight: ["400", "700"], variable: "--font-heading", display: "swap" });

export const metadata: Metadata = {
  title: "AgentiCart",
  description:
    "AgentiCart answers questions about orders, refunds, returns and products, in English, Hindi and Hinglish.",
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#a7d8ff" },
    { media: "(prefers-color-scheme: dark)", color: "#0f1d2e" },
  ],
};

const initializeTheme = `
  (() => {
    try {
      const saved = localStorage.getItem("ecom-support-theme");
      document.documentElement.dataset.theme = saved || "light";
    } catch {
      document.documentElement.dataset.theme = "light";
    }
  })();
`;

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${inter.variable} ${caslon.variable}`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: initializeTheme }} />
      </head>
      <body>{children}</body>
    </html>
  );
}

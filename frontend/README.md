# E-commerce Support Assistant frontend

This directory contains the independent Next.js App Router frontend. It has an interactive chat UI connected to the existing Flask API; messages and session IDs are not persisted by the frontend.

## Requirements

- Node.js 20.9 or newer
- npm

On Windows, install the current Node.js LTS release using the official [Windows installer](https://nodejs.org/en/download). Open a new PowerShell window and verify `node --version` and `npm.cmd --version`. The frontend does not require adding a machine-specific Node installation directory to `PATH` manually.

## Configure and run

Create `.env.local` from the example and point it to the Flask base URL. The URL is public client configuration, so do not add credentials or API keys:

```powershell
if (-not (Test-Path .env.local)) { Copy-Item .env.example .env.local }
# Edit .env.local only if Flask is not at http://127.0.0.1:5000.
npm.cmd install
npm.cmd run dev
```

Before starting Flask, configure its exact CORS origin, for example `CORS_ALLOWED_ORIGINS=http://localhost:3000`. Wildcard origins are rejected. Start Flask from the project root with `python run.py`, then open `http://localhost:3000`.

The client sends `{session_id, message}` to `${NEXT_PUBLIC_CHAT_API_BASE_URL}/api/chat` and displays the returned `answer`. A UUID is created on first send and reused until **New chat** is selected. It is a conversation key, not authentication. The UI displays API and network errors without substituting a mock answer. The browser does not save messages or the session ID across refreshes.

For a production build and local production server:

```powershell
npm.cmd run lint
npm.cmd run build
npm.cmd run start
```

The UI includes a responsive sidebar, an initial assistant greeting, right-aligned customer bubbles, left-aligned assistant bubbles, a typing indicator, and a composer. Enter sends, Shift+Enter inserts a newline, and the view scrolls to the newest message. Light/dark appearance follows the saved browser choice or system preference and persists in `localStorage`.

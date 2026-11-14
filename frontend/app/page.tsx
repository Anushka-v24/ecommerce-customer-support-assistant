"use client";

import { useEffect, useRef, useState } from "react";

type ChatMessage = {
  id: number;
  role: "assistant" | "user" | "error";
  content: string;
};

const INITIAL_GREETING = "Hi! How can we help you today?";
const CHAT_API_BASE_URL = process.env.NEXT_PUBLIC_CHAT_API_BASE_URL?.trim().replace(/\/+$/, "");

function isJsonObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

type IconName =
  | "bag"
  | "chat"
  | "menu"
  | "plus"
  | "sun"
  | "moon"
  | "sparkles"
  | "arrow"
  | "close";

function Icon({ name, className = "" }: { name: IconName; className?: string }) {
  const shared = {
    className,
    width: 20,
    height: 20,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.8,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    "aria-hidden": true as const,
  };

  switch (name) {
    case "bag":
      return <svg {...shared}><path d="M5 8h14l1 12H4L5 8Z" /><path d="M9 9V6a3 3 0 0 1 6 0v3" /><path d="M9 13h.01M15 13h.01" /></svg>;
    case "chat":
      return <svg {...shared}><path d="M20 11.5a7.5 7.5 0 0 1-7.5 7.5H6l-3 2v-6.5a7.5 7.5 0 1 1 17-3Z" /><path d="M8 11h8M8 14h5" /></svg>;
    case "menu":
      return <svg {...shared}><path d="M4 6h16M4 12h16M4 18h16" /></svg>;
    case "plus":
      return <svg {...shared}><path d="M12 5v14M5 12h14" /></svg>;
    case "sun":
      return <svg {...shared}><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.93 4.93l1.42 1.42m11.3 11.3 1.42 1.42M2 12h2m16 0h2M4.93 19.07l1.42-1.42m11.3-11.3 1.42-1.42" /></svg>;
    case "moon":
      return <svg {...shared}><path d="M20.5 14.3A8.5 8.5 0 0 1 9.7 3.5 8.7 8.7 0 1 0 20.5 14.3Z" /></svg>;
    case "sparkles":
      return <svg {...shared}><path d="m12 3 1.5 5.5L19 10l-5.5 1.5L12 17l-1.5-5.5L5 10l5.5-1.5L12 3Z" /><path d="m19 15 .8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8L19 15Z" /></svg>;
    case "arrow":
      return <svg {...shared}><path d="M12 19V5M6 11l6-6 6 6" /></svg>;
    case "close":
      return <svg {...shared}><path d="m6 6 12 12M18 6 6 18" /></svg>;
  }
}

function Sidebar({
  mobileOpen,
  onClose,
  onNewChat,
  onToggleTheme,
}: {
  mobileOpen: boolean;
  onClose: () => void;
  onNewChat: () => void;
  onToggleTheme: () => void;
}) {
  return (
    <>
      {mobileOpen && (
        <button
          className="sidebar-backdrop fixed inset-0 z-30 md:hidden"
          aria-label="Close navigation"
          onClick={onClose}
        />
      )}
      <aside
        className={`sidebar fixed inset-y-0 left-0 z-40 flex w-[280px] flex-col border-r p-4 transition-transform duration-200 md:static md:z-auto md:w-[264px] md:translate-x-0 ${mobileOpen ? "translate-x-0" : "-translate-x-full"}`}
        aria-label="Main navigation"
      >
        <div className="mb-8 flex items-center justify-between px-1">
          <a className="brand flex min-w-0 items-center gap-3" href="#home" aria-label="E-commerce Support Assistant home">
            <span className="brand-mark flex size-10 shrink-0 items-center justify-center rounded-xl">
              <Icon name="bag" className="size-[21px]" />
            </span>
            <span className="text-sm font-semibold leading-tight tracking-[-0.01em]">
              E-commerce<br />Support Assistant
            </span>
          </a>
          <button className="icon-button inline-flex md:hidden" onClick={onClose} aria-label="Close navigation">
            <Icon name="close" />
          </button>
        </div>

        <button className="new-chat-button flex h-11 items-center gap-3 rounded-xl px-3 text-sm font-medium transition-colors" type="button" onClick={onNewChat}>
          <Icon name="plus" className="size-[18px]" />
          <span>New chat</span>
        </button>

        <div className="mt-9 flex-1">
          <p className="section-label px-2 text-[11px] font-semibold uppercase tracking-[0.12em]">Recent conversations</p>
          <div className="history-placeholder mt-3 rounded-xl px-3 py-4 text-xs leading-relaxed">
            Your conversations will appear here.
          </div>
        </div>

        <div className="sidebar-footer border-t pt-3">
          <button
            className="theme-button flex h-11 w-full items-center gap-3 rounded-xl px-3 text-sm transition-colors"
            onClick={onToggleTheme}
            type="button"
            aria-label="Toggle light and dark appearance"
          >
            <span className="theme-icon-light"><Icon name="sun" className="size-[18px]" /></span>
            <span className="theme-icon-dark"><Icon name="moon" className="size-[18px]" /></span>
            <span>Appearance</span>
          </button>
          <div className="profile-row mt-2 flex items-center gap-3 rounded-xl px-2 py-3">
            <span className="avatar flex size-8 items-center justify-center rounded-full text-xs font-semibold">EA</span>
            <span className="text-xs font-medium">Customer support</span>
          </div>
        </div>
      </aside>
    </>
  );
}

export default function Home() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [draft, setDraft] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([
    { id: 0, role: "assistant", content: INITIAL_GREETING },
  ]);
  const [isRequesting, setIsRequesting] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const nextMessageId = useRef(1);
  const sessionIdRef = useRef<string | null>(null);
  const requestInProgressRef = useRef(false);
  const requestGenerationRef = useRef(0);
  const abortControllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, isRequesting]);

  function startNewChat() {
    requestGenerationRef.current += 1;
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    requestInProgressRef.current = false;
    sessionIdRef.current = null;
    nextMessageId.current = 1;
    setIsRequesting(false);
    setMessages([{ id: 0, role: "assistant", content: INITIAL_GREETING }]);
    setDraft("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    setMobileOpen(false);
  }

  async function sendMessage() {
    const content = draft.trim();
    if (!content || requestInProgressRef.current) return;

    requestInProgressRef.current = true;
    setIsRequesting(true);
    const generation = requestGenerationRef.current;
    const sessionId = sessionIdRef.current ?? globalThis.crypto.randomUUID();
    sessionIdRef.current = sessionId;
    const controller = new AbortController();
    abortControllerRef.current = controller;

    setMessages((current) => [
      ...current,
      { id: nextMessageId.current++, role: "user", content },
    ]);
    setDraft("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";

    try {
      if (!CHAT_API_BASE_URL) {
        throw new Error("Chat API is not configured. Set NEXT_PUBLIC_CHAT_API_BASE_URL in frontend/.env.local.");
      }

      const response = await fetch(`${CHAT_API_BASE_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message: content }),
        signal: controller.signal,
      });
      const payload: unknown = await response.json().catch(() => null);

      if (generation !== requestGenerationRef.current) return;

      if (!response.ok) {
        const serverMessage = isJsonObject(payload) && typeof payload.message === "string"
          ? payload.message
          : "The chat service could not process this message. Please try again.";
        throw new Error(serverMessage);
      }

      if (
        !isJsonObject(payload) ||
        typeof payload.answer !== "string" ||
        !payload.answer.trim() ||
        payload.session_id !== sessionId
      ) {
        throw new Error("The chat service returned an invalid response. Please try again.");
      }

      setMessages((current) => [
        ...current,
        { id: nextMessageId.current++, role: "assistant", content: payload.answer as string },
      ]);
    } catch (error) {
      if (generation !== requestGenerationRef.current) return;

      const message = error instanceof Error && error.name === "AbortError"
        ? null
        : error instanceof TypeError
          ? "Could not connect to the chat service. Check that Flask is running and its CORS allowlist includes this frontend origin."
          : error instanceof Error
            ? error.message
            : "The chat service could not process this message. Please try again.";

      if (message) {
        setMessages((current) => [
          ...current,
          { id: nextMessageId.current++, role: "error", content: message },
        ]);
      }
    } finally {
      if (generation === requestGenerationRef.current) {
        requestInProgressRef.current = false;
        abortControllerRef.current = null;
        setIsRequesting(false);
      }
    }
  }

  function toggleTheme() {
    const currentTheme = document.documentElement.dataset.theme === "dark" ? "dark" : "light";
    const nextTheme = currentTheme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = nextTheme;
    try {
      localStorage.setItem("ecom-support-theme", nextTheme);
    } catch {
      // Keep the current page usable when browser storage is unavailable.
    }
  }

  return (
    <main id="home" className="app-shell flex min-h-svh overflow-hidden">
      <Sidebar
        mobileOpen={mobileOpen}
        onClose={() => setMobileOpen(false)}
        onNewChat={startNewChat}
        onToggleTheme={toggleTheme}
      />

      <section className="chat-panel relative flex min-w-0 flex-1 flex-col">
        <header className="mobile-header flex h-14 shrink-0 items-center justify-between px-4 md:hidden">
          <button className="icon-button inline-flex" onClick={() => setMobileOpen(true)} aria-label="Open navigation">
            <Icon name="menu" />
          </button>
          <span className="text-sm font-semibold">E-commerce Customer Support</span>
          <button className="icon-button inline-flex" onClick={toggleTheme} aria-label="Toggle light and dark appearance">
            <span className="theme-icon-light"><Icon name="sun" className="size-[18px]" /></span>
            <span className="theme-icon-dark"><Icon name="moon" className="size-[18px]" /></span>
          </button>
        </header>

        <div className="chat-scroll flex min-h-0 min-w-0 w-full flex-1 flex-col items-center overflow-y-auto px-4 sm:px-6">
          <div className="chat-start my-auto w-full max-w-[720px] py-10 sm:py-14">
            <h1 className="welcome-title mb-8 text-center text-[28px] font-semibold leading-tight tracking-[-0.04em] sm:text-[40px]">
              E-commerce Customer Support
            </h1>
            <div className="message-list flex flex-col gap-5" aria-live="polite">
              {messages.map((message) =>
                message.role === "assistant" ? (
                  <div className="assistant-greeting flex items-start gap-3" key={message.id}>
                    <span className="welcome-icon flex size-9 shrink-0 items-center justify-center rounded-full">
                      <Icon name="sparkles" className="size-[18px]" />
                    </span>
                    <p className="assistant-message max-w-[85%] whitespace-pre-wrap break-words rounded-2xl border px-4 py-3 text-[15px] leading-6">
                      {message.content}
                    </p>
                  </div>
                ) : message.role === "user" ? (
                  <div className="flex justify-end" key={message.id}>
                    <p className="user-message max-w-[85%] whitespace-pre-wrap break-words rounded-2xl px-4 py-3 text-[15px] leading-6">
                      {message.content}
                    </p>
                  </div>
                ) : (
                  <div className="assistant-greeting flex items-start gap-3" key={message.id} role="alert">
                    <span className="error-mark flex size-9 shrink-0 items-center justify-center rounded-full text-sm font-semibold" aria-hidden="true">!</span>
                    <p className="error-message max-w-[85%] whitespace-pre-wrap break-words rounded-2xl border px-4 py-3 text-[15px] leading-6">
                      {message.content}
                    </p>
                  </div>
                ),
              )}
              {isRequesting && (
                <div className="assistant-greeting flex items-start gap-3" role="status" aria-label="Assistant is typing">
                  <span className="welcome-icon flex size-9 shrink-0 items-center justify-center rounded-full">
                    <Icon name="sparkles" className="size-[18px]" />
                  </span>
                  <div className="assistant-message typing-indicator flex h-12 items-center gap-1 rounded-2xl border px-4">
                    <span />
                    <span />
                    <span />
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            <footer className="composer-wrap sticky bottom-0 z-10 mt-4 w-full max-w-[780px] pb-4 pt-2 sm:pb-6">
              <form
                className="composer flex items-end gap-2 rounded-[22px] border p-2 pl-4 shadow-sm"
                onSubmit={(event) => {
                  event.preventDefault();
                  sendMessage();
                }}
              >
                <textarea
                  ref={textareaRef}
                  aria-label="Message the E-commerce Support Assistant"
                  className="composer-input min-h-[42px] max-h-32 min-w-0 flex-1 resize-none overflow-y-auto bg-transparent py-3 text-sm leading-5 outline-none placeholder:opacity-60"
                  placeholder="Write a message..."
                  rows={1}
                  value={draft}
                  onChange={(event) => {
                    setDraft(event.target.value);
                    event.currentTarget.style.height = "auto";
                    event.currentTarget.style.height = `${Math.min(event.currentTarget.scrollHeight, 128)}px`;
                  }}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                      event.preventDefault();
                      sendMessage();
                    }
                  }}
                />
                <button className="send-button mb-0.5 flex size-9 shrink-0 items-center justify-center rounded-xl transition-colors" type="submit" disabled={!draft.trim() || isRequesting} aria-label="Send message">
                  <Icon name="arrow" className="size-[18px]" />
                </button>
              </form>
              <p className="composer-note mt-2 text-center text-[11px]">
                Chat history is not stored by this frontend.
              </p>
            </footer>
          </div>
        </div>
      </section>
    </main>
  );
}

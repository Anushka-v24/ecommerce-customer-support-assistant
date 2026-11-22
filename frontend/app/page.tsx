"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type Rating = "up" | "down";

type QuickReply = { label: string; value: string };

type Ticket = {
  ticket_id: string;
  category: string;
  subject: string;
  description: string;
  order_id: string | null;
  status: string;
  priority: string;
  created_at: string;
};

type Customer = { customer_id: string; name: string; order_ids: string[] };

type TimelineStep = { label: string; state: "done" | "current" | "upcoming" };

type Block =
  | { type: "order_card"; order_id: string; status: string; eta: string | null; product_name: string | null; note: string | null; timeline: TimelineStep[] }
  | { type: "product_card"; product_id: string; name: string; stock: number | null; in_stock: boolean; restock_date: string | null }
  | { type: "sources"; items: (string | { title?: string; document_id?: string })[] };

type View = "chat" | "tickets";

type ChatMessage = {
  id: number;
  role: "assistant" | "user" | "error";
  content: string;
  // Set on server answers so the customer can rate them.
  turnId?: string;
  question?: string;
  rating?: Rating;
  quickReplies?: QuickReply[];
  ticket?: Ticket;
  blocks?: Block[];
};

const INITIAL_GREETING = "Hi! I'm AgentiCart. How can I help you today?";
const CHAT_API_BASE_URL = process.env.NEXT_PUBLIC_CHAT_API_BASE_URL?.trim().replace(/\/+$/, "");

const CUSTOMER_STORAGE_KEY = "ecom-support-customer";

function isJsonObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseQuickReplies(value: unknown): QuickReply[] {
  if (!Array.isArray(value)) return [];
  return value.filter(
    (item): item is QuickReply => isJsonObject(item) && typeof item.label === "string" && typeof item.value === "string",
  );
}

function parseTicket(value: unknown): Ticket | undefined {
  return isJsonObject(value) && typeof value.ticket_id === "string" ? (value as unknown as Ticket) : undefined;
}

const BLOCK_TYPES = new Set(["order_card", "product_card", "sources"]);

function parseBlocks(value: unknown): Block[] {
  if (!Array.isArray(value)) return [];
  // Ticket blocks are shown through the existing ticket field.
  return value.filter((item): item is Block => isJsonObject(item) && BLOCK_TYPES.has(item.type as string));
}

function OrderCard({ block }: { block: Extract<Block, { type: "order_card" }> }) {
  return (
    <div className="ticket-card w-full rounded-2xl border px-4 py-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="ticket-id font-mono text-xs font-semibold">{block.order_id}</span>
        <span className="pill pill-open rounded-full px-2 py-0.5 text-[11px] font-medium">{block.status}</span>
      </div>
      {block.product_name && <p className="ticket-subject mt-2 text-sm font-semibold">{block.product_name}</p>}
      {block.timeline.length > 0 && (
        <ol className="timeline mt-3 flex items-start" aria-label="Delivery progress">
          {block.timeline.map((step) => (
            <li key={step.label} className={`timeline-step timeline-${step.state} flex flex-1 flex-col items-center gap-1.5 text-center`}>
              <span className="timeline-dot size-3 rounded-full" />
              <span className="text-[11px] leading-4">{step.label}</span>
            </li>
          ))}
        </ol>
      )}
      <p className="ticket-meta mt-2 text-[11px]">{block.eta ? `Expected ${block.eta}` : "No delivery date yet"}</p>
    </div>
  );
}

function ProductCard({ block }: { block: Extract<Block, { type: "product_card" }> }) {
  return (
    <div className="ticket-card flex w-full items-center justify-between gap-3 rounded-2xl border px-4 py-3">
      <span className="flex min-w-0 flex-col">
        <span className="ticket-subject truncate text-sm font-semibold">{block.name}</span>
        <span className="ticket-meta font-mono text-[11px]">{block.product_id}</span>
      </span>
      <span className={`pill rounded-full px-2 py-0.5 text-[11px] font-medium ${block.in_stock ? "pill-open" : "pill-priority-high"}`}>
        {block.in_stock ? `${block.stock} in stock` : block.restock_date ? `Back ${block.restock_date}` : "Out of stock"}
      </span>
    </div>
  );
}

function SourcesChip({ block }: { block: Extract<Block, { type: "sources" }> }) {
  const names = block.items.map((item) => (typeof item === "string" ? item : item.title ?? item.document_id ?? "Document"));
  return (
    <div className="sources flex flex-wrap items-center gap-1.5 text-[11px]">
      <span className="ticket-meta font-medium">Sources:</span>
      {names.map((name) => <span key={name} className="pill rounded-full px-2 py-0.5">{name}</span>)}
    </div>
  );
}

function BlockView({ block }: { block: Block }) {
  if (block.type === "order_card") return <OrderCard block={block} />;
  if (block.type === "product_card") return <ProductCard block={block} />;
  return <SourcesChip block={block} />;
}

/** Read Server-Sent Events from a fetch response; calls onEvent for each complete event. */
async function readEvents(response: Response, onEvent: (event: string, data: unknown) => void) {
  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const chunk = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const fields = Object.fromEntries(
        chunk.split("\n").map((line) => [line.slice(0, line.indexOf(": ")), line.slice(line.indexOf(": ") + 2)]),
      );
      if (fields.event && fields.data) onEvent(fields.event, JSON.parse(fields.data));
      boundary = buffer.indexOf("\n\n");
    }
    if (done) return;
  }
}

function formatDate(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function TicketCard({ ticket, onView }: { ticket: Ticket; onView?: () => void }) {
  return (
    <div className="ticket-card w-full rounded-2xl border px-4 py-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="ticket-id font-mono text-xs font-semibold">{ticket.ticket_id}</span>
        <span className="flex gap-1.5">
          <span className={`pill pill-${ticket.status} rounded-full px-2 py-0.5 text-[11px] font-medium capitalize`}>
            {ticket.status.replace("_", " ")}
          </span>
          <span className={`pill pill-priority-${ticket.priority} rounded-full px-2 py-0.5 text-[11px] font-medium capitalize`}>
            {ticket.priority} priority
          </span>
        </span>
      </div>
      <p className="ticket-subject mt-2 text-sm font-semibold">{ticket.subject}</p>
      <p className="ticket-description mt-1 text-[13px] leading-5">{ticket.description}</p>
      <div className="mt-2 flex flex-wrap items-center justify-between gap-2">
        <span className="ticket-meta text-[11px]">Raised {formatDate(ticket.created_at)}</span>
        {onView && (
          <button type="button" className="link-button text-xs font-medium" onClick={onView}>
            View in My tickets
          </button>
        )}
      </div>
    </div>
  );
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
  | "close"
  | "thumbUp"
  | "thumbDown"
  | "ticket"
  | "camera";

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
    case "thumbUp":
      return <svg {...shared}><path d="M7 10v11H4V10h3Z" /><path d="M7 10l4-7a2 2 0 0 1 3 2l-1 5h6a2 2 0 0 1 2 2.3l-1.4 7A2 2 0 0 1 17.6 21H7" /></svg>;
    case "thumbDown":
      return <svg {...shared}><path d="M17 14V3h3v11h-3Z" /><path d="M17 14l-4 7a2 2 0 0 1-3-2l1-5H5a2 2 0 0 1-2-2.3l1.4-7A2 2 0 0 1 6.4 3H17" /></svg>;
    case "camera":
      return <svg {...shared}><path d="M4 8h3l2-3h6l2 3h3v11H4V8Z" /><circle cx="12" cy="13" r="3.5" /></svg>;
    case "ticket":
      return <svg {...shared}><path d="M3 8a2 2 0 0 0 2-2h14a2 2 0 0 0 2 2v2a2 2 0 0 0 0 4v2a2 2 0 0 0-2 2H5a2 2 0 0 0-2-2v-2a2 2 0 0 0 0-4V8Z" /><path d="M14 6v12" strokeDasharray="2 2" /></svg>;
  }
}

function BotFace({ className = "" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 120 130" fill="none" aria-hidden="true">
      <ellipse cx="60" cy="122" rx="30" ry="5" fill="#0b3558" opacity=".18" />
      <line x1="60" y1="14" x2="60" y2="26" stroke="#ffffff" strokeWidth="4" strokeLinecap="round" />
      <circle className="bot-antenna" cx="60" cy="11" r="6" fill="#2f78b7" />
      <rect x="14" y="26" width="92" height="74" rx="32" fill="#ffffff" />
      <rect x="14" y="26" width="92" height="74" rx="32" stroke="#cfe0ee" strokeWidth="2" />
      <rect x="4" y="52" width="10" height="22" rx="5" fill="#ffffff" stroke="#cfe0ee" strokeWidth="2" />
      <rect x="106" y="52" width="10" height="22" rx="5" fill="#ffffff" stroke="#cfe0ee" strokeWidth="2" />
      <rect x="26" y="38" width="68" height="50" rx="24" fill="#eaf4fb" />
      <g className="bot-eyes">
        <ellipse cx="46" cy="59" rx="6.5" ry="8" fill="#1d4468" />
        <ellipse cx="74" cy="59" rx="6.5" ry="8" fill="#1d4468" />
        <circle cx="48.2" cy="56" r="2.2" fill="#ffffff" />
        <circle cx="76.2" cy="56" r="2.2" fill="#ffffff" />
      </g>
      <path d="M46 74c5 7.5 23 7.5 28 0" stroke="#1d4468" strokeWidth="4.5" strokeLinecap="round" />
      <circle cx="36" cy="72" r="4.5" fill="#f6b9a6" opacity=".65" />
      <circle cx="84" cy="72" r="4.5" fill="#f6b9a6" opacity=".65" />
      <rect x="40" y="100" width="40" height="14" rx="7" fill="#ffffff" stroke="#cfe0ee" strokeWidth="2" />
    </svg>
  );
}

function BotHero({ docked, onOpen }: { docked: boolean; onOpen: () => void }) {
  const [showBubble, setShowBubble] = useState(false);
  useEffect(() => {
    const timer = window.setTimeout(() => setShowBubble(true), 900);
    return () => window.clearTimeout(timer);
  }, []);
  return (
    <div className={`bot-hero flex flex-col items-center${docked ? " docked" : ""}`}>
      {showBubble && !docked && (
        <button type="button" className="bot-bubble rounded-3xl px-6 py-4 text-center" onClick={onOpen}>
          <span className="bot-bubble-title block">Welcome to AgentiCart! 😊</span>
          <span className="block text-[15px] sm:text-[17px]">What can I help you with?</span>
        </button>
      )}
      <button
        type="button"
        className="bot-launcher"
        onClick={onOpen}
        aria-label={docked ? "Close the AgentiCart chat" : "Open the AgentiCart assistant"}
      >
        <BotFace className="bot-figure" />
      </button>
      {!docked && <span className="bot-hint">Click me to chat</span>}
    </div>
  );
}

function TopBar({
  onClose,
  onNewChat,
  onToggleTheme,
  view,
  onSelectView,
  customers,
  customerId,
  onSelectCustomer,
  ticketCount,
}: {
  onClose: () => void;
  onNewChat: () => void;
  onToggleTheme: () => void;
  view: View;
  onSelectView: (view: View) => void;
  customers: Customer[];
  customerId: string;
  onSelectCustomer: (customerId: string) => void;
  ticketCount: number;
}) {
  return (
    <header className="chat-header flex shrink-0 flex-wrap items-center justify-between gap-x-2 gap-y-2 px-3 py-3">
      <nav className="view-tabs glass flex gap-1 rounded-full p-1" role="tablist" aria-label="Views">
        {([["chat", "Chat"], ["tickets", "My tickets"]] as const).map(([key, label]) => (
          <button
            key={key}
            type="button"
            role="tab"
            aria-selected={view === key}
            className="view-tab rounded-full px-3.5 py-1.5 text-sm font-medium transition-colors"
            onClick={() => onSelectView(key)}
          >
            {label}
            {key === "tickets" && ticketCount > 0 && (
              <span className="tab-count ml-1.5 rounded-full px-1.5 text-[11px]">{ticketCount}</span>
            )}
          </button>
        ))}
      </nav>

      <div className="flex min-w-0 items-center gap-2">
        <label className="profile-row glass flex min-w-0 items-center gap-2 rounded-full py-1 pl-1 pr-3">
          <span className="avatar flex size-7 shrink-0 items-center justify-center rounded-full text-[11px] font-semibold">
            {customers.find((item) => item.customer_id === customerId)?.name.split(" ").map((part) => part[0]).join("").slice(0, 2) || "?"}
          </span>
          <select
            className="customer-select min-w-0 max-w-[64px] rounded-md bg-transparent text-xs font-medium outline-none"
            value={customerId}
            onChange={(event) => onSelectCustomer(event.target.value)}
            aria-label="Demo customer"
          >
            <option value="">Guest</option>
            {customers.map((customer) => (
              <option key={customer.customer_id} value={customer.customer_id}>
                {customer.name} · {customer.order_ids.join(", ")}
              </option>
            ))}
          </select>
        </label>
        <button className="icon-button glass inline-flex" type="button" onClick={onNewChat} aria-label="New chat" title="New chat">
          <Icon name="plus" className="size-[18px]" />
        </button>
        <button className="icon-button glass inline-flex" type="button" onClick={onClose} aria-label="Close chat" title="Close chat">
          <Icon name="close" className="size-[18px]" />
        </button>
        <button className="icon-button glass inline-flex" type="button" onClick={onToggleTheme} aria-label="Toggle light and dark appearance">
          <span className="theme-icon-light"><Icon name="sun" className="size-[18px]" /></span>
          <span className="theme-icon-dark"><Icon name="moon" className="size-[18px]" /></span>
        </button>
      </div>
    </header>
  );
}

export default function Home() {
  const [greeting, setGreeting] = useState("Hello");
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([
    { id: 0, role: "assistant", content: INITIAL_GREETING },
  ]);
  const [isRequesting, setIsRequesting] = useState(false);
  const [statusLabel, setStatusLabel] = useState("");
  const [view, setView] = useState<View>("chat");
  const [customers, setCustomers] = useState<Customer[]>([]);
  const [customerId, setCustomerId] = useState("");
  const [tickets, setTickets] = useState<Ticket[]>([]);
  const [ticketsStatus, setTicketsStatus] = useState<"idle" | "loading" | "error">("idle");
  const [toast, setToast] = useState<Ticket | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const photoInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const nextMessageId = useRef(1);
  const sessionIdRef = useRef<string | null>(null);
  const requestInProgressRef = useRef(false);
  const requestGenerationRef = useRef(0);
  const abortControllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    const hour = new Date().getHours();
    setGreeting(hour < 12 ? "Good morning" : hour < 18 ? "Good afternoon" : "Good evening");
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, isRequesting]);

  useEffect(() => {
    if (!CHAT_API_BASE_URL) return;
    fetch(`${CHAT_API_BASE_URL}/api/customers`)
      .then((response) => (response.ok ? response.json() : null))
      .then((payload: unknown) => {
        if (!isJsonObject(payload) || !Array.isArray(payload.customers)) return;
        const loaded = payload.customers as Customer[];
        setCustomers(loaded);
        // Restore the last demo customer only if it still exists.
        let saved: string | null = null;
        try {
          saved = localStorage.getItem(CUSTOMER_STORAGE_KEY);
        } catch {
          // Storage can be unavailable; the customer can still pick from the menu.
        }
        if (saved && loaded.some((item) => item.customer_id === saved)) setCustomerId(saved);
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(null), 8000);
    return () => window.clearTimeout(timer);
  }, [toast]);

  const loadTickets = useCallback(async () => {
    const sessionId = sessionIdRef.current;
    if (!CHAT_API_BASE_URL || (!customerId && !sessionId)) {
      setTickets([]);
      setTicketsStatus("idle");
      return;
    }
    setTicketsStatus("loading");
    const query = customerId ? `customer_id=${encodeURIComponent(customerId)}` : `session_id=${encodeURIComponent(sessionId!)}`;
    try {
      const response = await fetch(`${CHAT_API_BASE_URL}/api/tickets?${query}`);
      const payload: unknown = await response.json().catch(() => null);
      if (!response.ok || !isJsonObject(payload) || !Array.isArray(payload.tickets)) throw new Error("tickets");
      setTickets(payload.tickets.map(parseTicket).filter((item): item is Ticket => Boolean(item)));
      setTicketsStatus("idle");
    } catch {
      setTicketsStatus("error");
    }
  }, [customerId]);

  useEffect(() => {
    if (view === "tickets") loadTickets();
  }, [view, loadTickets]);

  function selectView(next: View) {
    setView(next);
  }

  function selectCustomer(next: string) {
    setCustomerId(next);
    try {
      if (next) localStorage.setItem(CUSTOMER_STORAGE_KEY, next);
      else localStorage.removeItem(CUSTOMER_STORAGE_KEY);
    } catch {
      // Keep the selection for this page view only.
    }
    startNewChat();
  }

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
  }

  async function sendMessage(reply?: QuickReply) {
    const content = reply ? reply.value : draft.trim();
    const shown = reply ? reply.label : content;
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
      { id: nextMessageId.current++, role: "user", content: shown },
    ]);
    if (!reply) {
      setDraft("");
      if (textareaRef.current) textareaRef.current.style.height = "auto";
    }

    try {
      if (!CHAT_API_BASE_URL) {
        throw new Error("Chat API is not configured. Set NEXT_PUBLIC_CHAT_API_BASE_URL in frontend/.env.local.");
      }

      setStatusLabel("");
      const response = await fetch(`${CHAT_API_BASE_URL}/api/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message: content, ...(customerId ? { customer_id: customerId } : {}) }),
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        const errorPayload: unknown = await response.json().catch(() => null);
        const serverMessage = isJsonObject(errorPayload) && typeof errorPayload.message === "string"
          ? errorPayload.message
          : "The chat service could not process this message. Please try again.";
        throw new Error(serverMessage);
      }

      // Filled by the event callback; read once the stream has ended.
      const received: { payload: unknown; error: string | null } = { payload: null, error: null };
      await readEvents(response, (event, data) => {
        if (generation !== requestGenerationRef.current) return;
        if (event === "status" && isJsonObject(data) && typeof data.label === "string") setStatusLabel(data.label);
        else if (event === "result") received.payload = data;
        else if (event === "error") {
          received.error = isJsonObject(data) && typeof data.message === "string" ? data.message : "The chat service could not process this message.";
        }
      });

      if (generation !== requestGenerationRef.current) return;
      if (received.error) throw new Error(received.error);
      const payload = received.payload;

      if (
        !isJsonObject(payload) ||
        typeof payload.answer !== "string" ||
        !payload.answer.trim() ||
        payload.session_id !== sessionId
      ) {
        throw new Error("The chat service returned an invalid response. Please try again.");
      }

      const turnId = typeof payload.turn_id === "string" ? payload.turn_id : undefined;
      const quickReplies = parseQuickReplies(payload.quick_replies);
      const ticket = parseTicket(payload.ticket);
      const blocks = parseBlocks(payload.blocks);
      setMessages((current) => [
        ...current,
        {
          id: nextMessageId.current++, role: "assistant", content: payload.answer as string,
          turnId, question: content, quickReplies, ticket, blocks,
        },
      ]);
      if (ticket) {
        setToast(ticket);
        setTickets((current) => [ticket, ...current.filter((item) => item.ticket_id !== ticket.ticket_id)]);
      }
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

  // Upload a damage photo, then send its description as the customer's next message.
  async function sendPhoto(file: File) {
    if (!CHAT_API_BASE_URL || requestInProgressRef.current) return;
    const sessionId = sessionIdRef.current ?? globalThis.crypto.randomUUID();
    sessionIdRef.current = sessionId;
    setIsRequesting(true);
    setStatusLabel("Looking at your photo");
    const form = new FormData();
    form.append("session_id", sessionId);
    form.append("photo", file);
    try {
      const response = await fetch(`${CHAT_API_BASE_URL}/api/evidence`, { method: "POST", body: form });
      const payload: unknown = await response.json().catch(() => null);
      if (!response.ok || !isJsonObject(payload) || typeof payload.summary !== "string") {
        throw new Error(isJsonObject(payload) && typeof payload.message === "string" ? payload.message : "The photo could not be uploaded.");
      }
      setIsRequesting(false);
      await sendMessage({ label: `📷 Sent a photo (${file.name})`, value: payload.summary });
    } catch (error) {
      setIsRequesting(false);
      setMessages((current) => [
        ...current,
        { id: nextMessageId.current++, role: "error", content: error instanceof Error ? error.message : "The photo could not be uploaded." },
      ]);
    }
  }

  async function rateAnswer(message: ChatMessage, rating: Rating) {
    if (!CHAT_API_BASE_URL || !message.turnId || !sessionIdRef.current) return;
    const previous = message.rating;
    const setRating = (value: Rating | undefined) =>
      setMessages((current) => current.map((item) => (item.id === message.id ? { ...item, rating: value } : item)));
    setRating(rating);
    try {
      const response = await fetch(`${CHAT_API_BASE_URL}/api/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionIdRef.current,
          turn_id: message.turnId,
          rating,
          message: message.question,
          answer: message.content,
        }),
      });
      if (!response.ok) setRating(previous);
    } catch {
      setRating(previous);
    }
  }

  const lastMessageId = messages[messages.length - 1]?.id;

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

  const customer = customers.find((item) => item.customer_id === customerId);
  const firstName = customer?.name.split(" ")[0];
  const isHero = messages.length === 1 && !isRequesting;

  const composer = (
    <form
      className="composer flex flex-col gap-1 rounded-[22px] px-3.5 pb-2.5 pt-2.5"
      onSubmit={(event) => {
        event.preventDefault();
        sendMessage();
      }}
    >
      <input
        ref={photoInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        className="hidden"
        onChange={(event) => {
          const file = event.target.files?.[0];
          event.target.value = "";
          if (file) sendPhoto(file);
        }}
      />
      <textarea
        ref={textareaRef}
        aria-label="Message AgentiCart"
        className="composer-input min-h-[34px] max-h-32 min-w-0 resize-none overflow-y-auto bg-transparent px-1 py-1.5 text-[15px] leading-6 outline-none"
        placeholder="Message AgentiCart"
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
      <div className="flex items-center justify-between">
        <button
          type="button"
          className="photo-button inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-[13px] font-medium"
          aria-label="Attach a photo of the damage"
          title="Attach a photo of the damage"
          disabled={isRequesting}
          onClick={() => photoInputRef.current?.click()}
        >
          <Icon name="camera" className="size-[17px]" />
          <span>Damage photo</span>
        </button>
        <button className="send-button flex size-10 shrink-0 items-center justify-center rounded-full transition-colors" type="submit" disabled={!draft.trim() || isRequesting} aria-label="Send message">
          <Icon name="arrow" className="size-[18px]" />
        </button>
      </div>
    </form>
  );

  const heroChips = ["Track my order", "Return an item", "Raise a complaint", "How do I pay?", "Talk to a human"];
  const commonQuestions = ["How do I place an order?", "How do I reset my password?", "What is the return window?"];

  const orderChips = (customer?.order_ids ?? []).slice(0, 3).map((orderId) => ({ label: `Track ${orderId}`, value: `Where is my order ${orderId}?` }));

  return (
    <main id="home" className="landing">
      <div className="stage-inner">
        <header className="landing-brand flex items-center gap-3.5">
          <span className="brand-mark landing-mark flex shrink-0 items-center justify-center rounded-2xl">
            <Icon name="bag" className="size-[30px]" />
          </span>
          <span className="flex flex-col gap-1">
            <span className="brand-name landing-title leading-none">AgentiCart</span>
            <span className="landing-tagline">Your shopping support, instantly</span>
          </span>
        </header>

        <BotHero docked={open} onOpen={() => setOpen((value) => !value)} />
        {open && (
      <section className="chat-window flex flex-col" aria-label="AgentiCart chat">
      <TopBar
        onClose={() => setOpen(false)}
        onNewChat={() => {
          startNewChat();
          setView("chat");
        }}
        onToggleTheme={toggleTheme}
        view={view}
        onSelectView={selectView}
        customers={customers}
        customerId={customerId}
        onSelectCustomer={selectCustomer}
        ticketCount={tickets.length}
      />

      <div className="chat-panel relative flex min-h-0 min-w-0 flex-1 flex-col">
        {toast && (
          <div className="toast glass fixed bottom-24 left-1/2 z-50 flex w-[calc(100%-32px)] max-w-[420px] -translate-x-1/2 items-center gap-3 rounded-2xl px-4 py-3 text-sm" role="status">
            <Icon name="ticket" className="size-[18px] shrink-0" />
            <span className="min-w-0 flex-1">
              Ticket <span className="font-mono font-semibold">{toast.ticket_id}</span> raised
            </span>
            <button type="button" className="link-button shrink-0 text-xs font-semibold" onClick={() => { selectView("tickets"); setToast(null); }}>
              View
            </button>
            <button type="button" className="icon-button inline-flex size-7 shrink-0" aria-label="Dismiss" onClick={() => setToast(null)}>
              <Icon name="close" className="size-[14px]" />
            </button>
          </div>
        )}

        {view === "tickets" ? (
          <div className="chat-scroll flex min-h-0 w-full flex-1 flex-col items-center overflow-y-auto px-4 sm:px-6">
            <div className="w-full max-w-[720px] py-5">
              <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
                <div>
                  <h1 className="welcome-title text-[26px] leading-tight">My tickets</h1>
                  <p className="ticket-meta mt-1 text-sm">
                    {customerId
                      ? `Tickets for ${customer?.name ?? customerId}`
                      : "Tickets from this conversation. Pick a demo customer at the top to see all of theirs."}
                  </p>
                </div>
                <button type="button" className="new-chat-button glass h-9 rounded-full px-4 text-sm font-medium" onClick={() => loadTickets()}>
                  Refresh
                </button>
              </div>
              {ticketsStatus === "error" ? (
                <p className="error-message rounded-2xl border px-4 py-3 text-sm" role="alert">Tickets could not be loaded. Check that Flask is running.</p>
              ) : ticketsStatus === "loading" && tickets.length === 0 ? (
                <p className="ticket-meta text-sm">Loading tickets…</p>
              ) : tickets.length === 0 ? (
                <div className="history-placeholder glass rounded-2xl px-4 py-6 text-center text-sm">
                  No tickets yet. When the assistant raises one for you, it will appear here.
                </div>
              ) : (
                <div className="flex flex-col gap-3">
                  {tickets.map((ticket) => <TicketCard key={ticket.ticket_id} ticket={ticket} />)}
                </div>
              )}
            </div>
          </div>
        ) : isHero ? (
          <div className="chat-scroll flex min-h-0 min-w-0 w-full flex-1 flex-col items-center overflow-y-auto px-4 sm:px-6">
            <div className="hero my-auto flex w-full flex-col items-center py-6">
              <h1 className="welcome-title mb-5 text-center text-[26px] leading-[1.15] sm:text-[32px]">
                {greeting}{firstName ? `, ${firstName}` : ""}
                <br />
                What can we help with?
              </h1>
              <div className="flex flex-wrap justify-center gap-2" role="group" aria-label="Suggestions">
                {[...heroChips.map((chip) => ({ label: chip, value: chip })), ...orderChips, ...commonQuestions.map((question) => ({ label: question, value: question }))].map((chip) => (
                  <button key={chip.label} type="button" className="hero-chip rounded-full px-3.5 py-2 text-[13px] font-medium" onClick={() => sendMessage(chip)}>
                    {chip.label}
                  </button>
                ))}
              </div>
              {!customer && <p className="info-empty mt-4 text-center text-xs">Pick a demo customer above to track their orders.</p>}
            </div>
          </div>
        ) : (
        <>
        <div className="chat-scroll flex min-h-0 min-w-0 w-full flex-1 flex-col items-center overflow-y-auto px-4 sm:px-6">
          <div className="chat-start w-full max-w-[720px] py-6">
            <div className="message-list flex flex-col gap-5" aria-live="polite">
              {messages.map((message) =>
                message.role === "assistant" ? (
                  <div className="assistant-greeting flex items-start gap-3" key={message.id}>
                    <span className="welcome-icon flex size-9 shrink-0 items-center justify-center rounded-full">
                      <Icon name="sparkles" className="size-[18px]" />
                    </span>
                    <div className="flex max-w-[85%] flex-col items-start gap-1">
                      <p className="assistant-message whitespace-pre-wrap break-words rounded-3xl border px-4 py-3 text-[15px] leading-6">
                        {message.content}
                      </p>
                      {message.blocks?.map((block, index) => <BlockView key={`${block.type}-${index}`} block={block} />)}
                      {message.ticket && <TicketCard ticket={message.ticket} onView={() => selectView("tickets")} />}
                      {message.quickReplies && message.quickReplies.length > 0 && message.id === lastMessageId && !isRequesting && (
                        <div className="flex flex-wrap gap-2 pt-1" role="group" aria-label="Suggested replies">
                          {message.quickReplies.map((reply) => (
                            <button
                              key={reply.value}
                              type="button"
                              className="quick-reply rounded-full border px-3.5 py-1.5 text-[13px] font-medium transition-colors"
                              onClick={() => sendMessage(reply)}
                            >
                              {reply.label}
                            </button>
                          ))}
                        </div>
                      )}
                      {message.turnId && (
                        <div className="feedback-row flex gap-1 pl-1" role="group" aria-label="Rate this answer">
                          {(["up", "down"] as const).map((rating) => (
                            <button
                              key={rating}
                              type="button"
                              className="feedback-button flex size-7 items-center justify-center rounded-lg transition-colors"
                              aria-label={rating === "up" ? "Helpful answer" : "Unhelpful answer"}
                              aria-pressed={message.rating === rating}
                              onClick={() => rateAnswer(message, rating)}
                            >
                              <Icon name={rating === "up" ? "thumbUp" : "thumbDown"} className="size-[15px]" />
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                ) : message.role === "user" ? (
                  <div className="flex justify-end" key={message.id}>
                    <p className="user-message max-w-[85%] whitespace-pre-wrap break-words rounded-3xl px-4 py-3 text-[15px] leading-6">
                      {message.content}
                    </p>
                  </div>
                ) : (
                  <div className="assistant-greeting flex items-start gap-3" key={message.id} role="alert">
                    <span className="error-mark flex size-9 shrink-0 items-center justify-center rounded-full text-sm font-semibold" aria-hidden="true">!</span>
                    <p className="error-message max-w-[85%] whitespace-pre-wrap break-words rounded-3xl border px-4 py-3 text-[15px] leading-6">
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
                  <div className="assistant-message flex h-12 items-center gap-3 rounded-3xl border px-4">
                    <span className="typing-indicator flex items-center gap-1">
                      <span />
                      <span />
                      <span />
                    </span>
                    {statusLabel && <span className="status-label text-[13px]">{statusLabel}…</span>}
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          </div>
        </div>
        </>
        )}
        {view === "chat" && (
          <footer className="composer-wrap z-10 w-full shrink-0 px-3 pb-3 pt-1">{composer}</footer>
        )}
      </div>
      </section>
        )}
      </div>
    </main>
  );
}

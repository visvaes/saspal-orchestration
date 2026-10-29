import { useState } from "react";

const starterMessage = {
  role: "assistant",
  content: "Hi! I can help summarize, research, or plan a task. Ask me anything.",
};

const agentLabels = {
  general_chat: "General Chat",
  research: "Research",
  data_analysis: "Data Analysis",
  task_planning: "Task Planning",
};

export function getChatApiUrl(env = import.meta.env) {
  const configuredBaseUrl = env.VITE_API_BASE_URL?.trim();
  if (!configuredBaseUrl) {
    if (env.PROD) {
      throw new Error(
        "VITE_API_BASE_URL must be set to your backend URL in production.",
      );
    }
    return "/api/chat";
  }

  const apiBaseUrl = configuredBaseUrl.replace(/\/+$/, "");
  let parsedBaseUrl;
  try {
    parsedBaseUrl = new URL(apiBaseUrl);
  } catch {
    throw new Error(
      "VITE_API_BASE_URL must contain only an absolute backend URL, not a variable assignment.",
    );
  }
  if (
    parsedBaseUrl.protocol !== "http:" &&
    parsedBaseUrl.protocol !== "https:"
  ) {
    throw new Error("VITE_API_BASE_URL must use HTTP or HTTPS.");
  }

  return `${apiBaseUrl}/api/chat`;
}

export default function App() {
  const [messages, setMessages] = useState([starterMessage]);
  const [draft, setDraft] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(event) {
    event.preventDefault();
    const trimmed = draft.trim();
    if (!trimmed || loading) {
      return;
    }
    const userMessage = { role: "user", content: trimmed };
    const nextMessages = [...messages, userMessage];
    setMessages(nextMessages);
    setDraft("");
    setLoading(true);

    try {
      const response = await fetch(getChatApiUrl(), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: trimmed,
          history: nextMessages.slice(0, -1).map(({ role, content }) => ({
            role,
            content,
          })),
        }),
      });

      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.detail || "The request failed.");
      }

      const assistantMessage = {
        role: "assistant",
        content: payload.response || "I didn't get a useful answer back.",
        agentUsed:
          payload.agent_used || payload.agents_used?.[0] || null,
      };
      setMessages([...nextMessages, assistantMessage]);
    } catch (error) {
      setMessages([
        ...nextMessages,
        {
          role: "assistant",
          content: error.message || "Something went wrong. Please try again.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="chat-panel">
        <header className="chat-header">
          <h1>Agent Orchestration</h1>
        </header>

        <div className="message-list" aria-live="polite">
          {messages.map((message, index) => (
            <div
              key={`${message.role}-${index}`}
              className={`message ${message.role}`}
            >
              <strong>{message.role === "user" ? "You" : "Assistant"}</strong>
              <p>{message.content}</p>
              {message.role === "assistant" &&
                agentLabels[message.agentUsed] && (
                  <span className="agent-used">
                    Handled by: {agentLabels[message.agentUsed]}
                  </span>
                )}
            </div>
          ))}
          {loading && (
            <div className="message assistant loading">
              <strong>Assistant</strong>
              <p>Thinking…</p>
            </div>
          )}
        </div>

        <form className="composer" onSubmit={handleSubmit}>
          <textarea
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            rows="3"
            placeholder="Type your message..."
            aria-label="Message"
          />
          <button type="submit" disabled={loading || !draft.trim()}>
            {loading ? "Sending..." : "Send"}
          </button>
        </form>
      </section>
    </main>
  );
}

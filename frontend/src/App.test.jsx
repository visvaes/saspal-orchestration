import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import "@testing-library/jest-dom/vitest";

import App from "./App";

describe("App", () => {
  it("renders the opening message and submits a prompt", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        response: "I can help with that.",
        agent_used: "general_chat",
        agents_used: ["general_chat"],
      }),
    });
    global.fetch = fetchMock;

    render(<App />);

    expect(
      screen.getByText(/Hi! I can help summarize, research, or plan a task./i),
    ).toBeInTheDocument();

    await user.type(screen.getByLabelText(/message/i), "Test prompt");
    await user.click(screen.getByRole("button", { name: /send/i }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    expect(await screen.findByText("I can help with that.")).toBeInTheDocument();
    expect(
      screen.getByText("Handled by: General Chat"),
    ).toBeInTheDocument();
  });
});

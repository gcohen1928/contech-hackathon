export type MessageRole = "user" | "assistant";

interface Message {
  role: MessageRole;
  content: string;
}

interface Citation {
  source_filename: string;
  page_number: number;
}

interface ChatResponse {
  message: string;
  citations: Citation[];
}

export const sendChatMessage = async (
  messages: Message[],
  context: string[]
): Promise<ChatResponse> => {
  const response = await fetch("http://localhost:8000/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ messages, allowed_context: context }),
  });

  if (!response.ok) {
    throw new Error("Failed to send chat message");
  }

  return response.json();
}

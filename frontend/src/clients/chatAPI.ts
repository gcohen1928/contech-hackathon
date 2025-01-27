export type MessageRole = "user" | "assistant";

interface Message {
  role: MessageRole;
  content: string;
}

interface ChatResponse {
  message: string;
}

export const sendChatMessage = async (
  messages: Message[]
): Promise<ChatResponse> => {
  const response = await fetch("http://localhost:8000/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ messages }),
  });

  if (!response.ok) {
    throw new Error("Failed to send chat message");
  }

  return response.json();
}

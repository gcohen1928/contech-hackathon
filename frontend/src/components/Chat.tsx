import { CornerDownLeft } from "lucide-react";
import { Button } from "./ui/button";
import { ChatBubble, ChatBubbleMessage } from "./ui/chat/chat-bubble";
import { ChatInput } from "./ui/chat/chat-input";
import { ChatMessageList } from "./ui/chat/chat-message-list";
import { useEffect, useRef, useState } from "react";
import ContextSelector from "./ContextSelector";
import { MessageRole, sendChatMessage } from "@/clients/chatAPI";
import { useMessagesStore } from "@/state/messages";
import { useContextStore } from "@/state/context";

interface Message {
  id: string;
  content: string;
  role: MessageRole;
}

const ChatTextbox = () => {
  const [content, setContent] = useState("");

  const { messages, addMessage, setLoading } = useMessagesStore();
  const { selectedContexts } = useContextStore();

  const handleAddMessage = (content: string) => {
    setContent("");
    // Add user message to state
    const userMessage = {
      id: crypto.randomUUID(),
      content,
      role: "user" as const,
    };

    addMessage(userMessage);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (content && e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleAddMessage(content);
    }
  };

  const handleSendMessage = async () => {
    const lastMessage = messages[messages.length - 1];

    if (lastMessage?.role !== "user") {
      return;
    }

    setLoading(true);

    try {
      // Send message to API
      const response = await sendChatMessage(
        [...messages.filter((m) => m.id !== "loading" && m.id !== "error")],
        selectedContexts.map((c) => c.value)
      );

      // Add assistant response to state
      // Remove loading message
      addMessage({
        id: crypto.randomUUID(),
        content: response.message,
        role: "assistant" as const,
      });
    } catch (error) {
      console.error("Failed to send message:", error);
      // Add error message to state
      addMessage({
        id: "error",
        content: "Sorry, I'm having trouble responding. Please try again.",
        role: "assistant" as const,
      });
    }
  };

  // After a message is sent, wait 2 seconds before asking assistant to respond
  useEffect(() => {
    const timeoutID = setTimeout(() => {
      console.log("Calling assistant");
      handleSendMessage();
    }, 2_000);

    return () => clearTimeout(timeoutID);
  }, [messages, content]);

  return (
    <>
      <ChatInput
        placeholder="Type your message here..."
        className="min-h-12 resize-none rounded-lg bg-background border-0 p-3 shadow-none focus-visible:ring-0"
        value={content}
        onChange={(e) => setContent(e.target.value)}
        onKeyDown={handleKeyDown}
      />
      <div className="flex items-center p-3 pt-0">
        <Button
          size="sm"
          className="ml-auto gap-1.5"
          onClick={() => handleAddMessage(content)}
        >
          Send
          <CornerDownLeft className="size-3.5" />
        </Button>
      </div>
    </>
  );
};

const Chat = () => {
  const { messages } = useMessagesStore();

  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const formRef = useRef<HTMLFormElement>(null);

  useEffect(() => {
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTop =
        messagesContainerRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <div className="flex h-full flex-col p-4">
      {messages.length > 0 ? (
        <>
          <ContextSelector />

          <div className="flex-1 overflow-y-auto space-y-4">
            <ChatMessageList ref={messagesContainerRef}>
              {messages.map((message) => (
                <ChatBubble
                  key={message.id}
                  variant={message.role === "user" ? "sent" : "received"}
                >
                  <ChatBubbleMessage
                    variant={message.role === "user" ? "sent" : "received"}
                    isLoading={message.id === "loading"}
                  >
                    {message.content}
                  </ChatBubbleMessage>
                </ChatBubble>
              ))}
            </ChatMessageList>
          </div>

          <form
            className="relative rounded-lg border bg-background focus-within:ring-1 focus-within:ring-ring p-1"
            ref={formRef}
          >
            <ChatTextbox />
          </form>
        </>
      ) : (
        <div className="flex flex-1 flex-col items-center justify-center space-y-8">
          <div className="text-3xl font-bold text-center">
            What can Omniscope help you with?
          </div>

          <form
            className="relative w-full max-w-2xl rounded-lg border bg-background focus-within:ring-1 focus-within:ring-ring p-1"
            ref={formRef}
          >
            <ChatTextbox />
          </form>
        </div>
      )}
    </div>
  );
};

export default Chat;

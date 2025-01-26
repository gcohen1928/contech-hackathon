import { CornerDownLeft } from "lucide-react";
import { Button } from "./ui/button";
import {
  ChatBubble,
  ChatBubbleAvatar,
  ChatBubbleMessage,
} from "./ui/chat/chat-bubble";
import { ChatInput } from "./ui/chat/chat-input";
import { ChatMessageList } from "./ui/chat/chat-message-list";
import { useEffect, useState } from "react";
import { useSelectedFile } from "@/state/selectedFile";
import ContextSelector from "./ContextSelector";

interface Message {
  id: string;
  content: string;
  sender: "user" | "assistant";
}

const defaultMessages: Message[] = [
  { id: "1", content: "Hello, how are you?", sender: "user" },
  { id: "2", content: "I'm fine, thank you!", sender: "assistant" },
];

const ChatTextbox = () => {
  return (
    <>
      <ChatInput
        placeholder="Type your message here..."
        className="min-h-12 resize-none rounded-lg bg-background border-0 p-3 shadow-none focus-visible:ring-0"
      />
      <div className="flex items-center p-3 pt-0">
        <Button size="sm" className="ml-auto gap-1.5">
          Send
          <CornerDownLeft className="size-3.5" />
        </Button>
      </div>
    </>
  );
};

const Chat = () => {
  const [messages, setMessages] = useState<Message[]>([]);

  const { filePath } = useSelectedFile();

  useEffect(() => {
    if (filePath && messages.length === 0) {
      setMessages([
        {
          id: "1",
          content: "How can I help you with this file?",
          sender: "assistant",
        },
      ]);
    }
  }, [filePath]);

  // TODO: handle sending/receiving messages w/context in payload

  return (
    <div className="flex h-full flex-col">
      <div className="flex flex-1 flex-col p-4">
        {messages.length > 0 ? (
          <>
            <div className="flex-1 overflow-y-auto space-y-4">
              <ContextSelector />

              <ChatMessageList>
                {messages.map((message) => (
                  <ChatBubble
                    key={message.id}
                    variant={message.sender === "user" ? "sent" : "received"}
                  >
                    <ChatBubbleAvatar
                      fallback={message.sender === "user" ? "US" : "AI"}
                    />
                    <ChatBubbleMessage
                      variant={message.sender === "user" ? "sent" : "received"}
                    >
                      {message.content}
                    </ChatBubbleMessage>
                  </ChatBubble>
                ))}
              </ChatMessageList>
            </div>

            <form className="relative rounded-lg border bg-background focus-within:ring-1 focus-within:ring-ring p-1">
              <ChatTextbox />
            </form>
          </>
        ) : (
          <div className="flex flex-1 flex-col items-center justify-center space-y-8">
            <div className="text-3xl font-bold text-center">
              What can Omniscope help you with?
            </div>

            <form className="relative w-full max-w-2xl rounded-lg border bg-background focus-within:ring-1 focus-within:ring-ring p-1">
              <ChatTextbox />
            </form>
          </div>
        )}
      </div>
    </div>
  );
};

export default Chat;

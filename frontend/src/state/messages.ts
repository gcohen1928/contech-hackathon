import { create } from 'zustand'
import { MessageRole } from '@/clients/chatAPI'

interface Message {
  id: string
  content: string
  role: MessageRole
}

interface MessagesState {
  messages: Message[]
  setLoading: (loading: boolean) => void;
  addMessage: (message: Message) => void
  clearMessages: () => void
}

export const useMessagesStore = create<MessagesState>((set) => ({
  messages: [],

  setLoading: (loading) =>
    set((state) => {
      const lastMessage: Message | undefined =
        state.messages[state.messages.length - 1];

      if (loading) {
        // Add loading message
        if (lastMessage?.id === "loading") {
          return state;
        } else {
          return {
            messages: [...state.messages, {
              id: "loading",
              content: "Thinking...",
              role: "assistant"
            }]
          };
        }
      } else {
        if (lastMessage?.id === "loading") {
          return state;
        } else {
          // Remove loading message
          return {
            messages: state.messages.filter((msg) => msg.id !== "loading"),
          };
        }
      }
    }),

  addMessage: (message) => 
    set((state) => ({
      messages: [...state.messages.filter((msg) => msg.id !== "loading"), message]
    })),

  clearMessages: () =>
    set(() => ({
      messages: []
    }))
}))

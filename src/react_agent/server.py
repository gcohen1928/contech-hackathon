from typing import List

from langchain_core.messages import HumanMessage, AIMessage

from pydantic import BaseModel

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

from react_agent.graph import graph

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/status")
async def health_check():
    return {"message": "up"}


class Message(BaseModel):
    role: str
    content: str


class ChatPayload(BaseModel):
    messages: List[Message]


@app.post("/chat")
async def chat(payload: ChatPayload):
    print("payload:", payload)

    # Create a chat message
    messages = [
        (
            HumanMessage(content=message.content)
            if message.role == "user"
            else AIMessage(content=message.content)
        )
        for message in payload.messages
    ]

    try:
        final_state = await graph.ainvoke({"messages": messages})
    except Exception as e:
        print("Error during graph invocation:", e)
        return {"message": "Sorry, I'm having trouble answering that question."}

    return {"message": final_state["messages"][-1].content}

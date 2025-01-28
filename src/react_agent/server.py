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
    allowed_context: List[str]


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
        final_state = await graph.ainvoke(
            {"messages": messages, "allowed_context": payload.allowed_context}
        )
    except Exception as e:
        print("Error during graph invocation:", e)
        return {"message": "Sorry, I'm having trouble answering that question."}

    # print("FINAL STATE:", final_state)
    final_msg = []

    if "reply" in final_state and final_state["reply"] is not None:
        final_msg.append(final_state["reply"])

    if "user_question" in final_state and final_state["user_question"] is not None:
        final_msg.append(final_state["user_question"])

    citations = []
    seen_citations = set()

    if "semantic_citations" in final_state:
        for citation in final_state["semantic_citations"]:
            citation_key = frozenset(citation.items())
            if citation_key not in seen_citations:
                citations.append(citation)
                seen_citations.add(citation_key)

    return {
        "message": "\n\n".join(final_msg),
        "citations": citations,
    }

from langchain_core.messages import HumanMessage
import chainlit as cl

from react_agent.graph import graph

@cl.on_message
async def main(message: cl.Message):
    # Create a chat message
    response_msg = cl.Message(content="")
    messages = [
        HumanMessage(content=message.content)
    ]
    
    # Get response from the model
    await response_msg.send()
    final_state = await graph.ainvoke({
        "messages": messages
    })

    print(final_state)

    for msg in final_state.messages:
        await response_msg.stream_token(msg.content)

    # Send a response back to the user
    # await cl.Message(
    #     content=f"Received: {message.content}",
    # ).send()
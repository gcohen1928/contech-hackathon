from langchain_core.messages import HumanMessage
import chainlit as cl

from react_agent.graph import graph


@cl.on_message
async def main(message: cl.Message):
    # TODO: track chat history
    # Create a chat message
    response_msg = cl.Message(content="")
    messages = [HumanMessage(content=message.content)]

    # Get response from the model
    await response_msg.send()

    try:
        final_state = await graph.ainvoke({"messages": messages})
    except Exception as e:
        print("Error during graph invocation:", e)
        await response_msg.stream_token(
            "Sorry, I'm having trouble answering that question."
        )
        return

    print("final_state:", final_state)
    await response_msg.stream_token(final_state["messages"][-1].content)

    # Send a response back to the user
    # await cl.Message(
    #     content=f"Received: {message.content}",
    # ).send()

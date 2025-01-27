"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from datetime import datetime, timezone
import os
from typing import Dict, List, Literal, cast, Union, Set, Annotated
import base64

import pandas as pd

# from vanna.remote import VannaDefault
# from vanna.flask import VannaFlaskApp
from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore
from react_agent.gryps_utils import get_ims_handler_from_env, IMSQueryHandler

from litellm import embedding, completion
import typesense
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.state import InputState, State
from react_agent.tools import TOOLS
from react_agent.utils import load_chat_model

# Global VannaDefault instance
_vanna_instance = None
_typesense_client = None
_failed_queries: Set[str] = set()


def get_typesense_client():
    global _typesense_client

    if _typesense_client is None:
        _typesense_client = typesense.Client(
            {
                "nodes": [
                    {
                        "host": "localhost",
                        "port": "8108",
                        "protocol": "http",
                    }
                ],
                "api_key": "admin",
                "connection_timeout_seconds": 2,
            }
        )

    return _typesense_client


class VannaInstance(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)


def get_vanna_instance() -> VannaInstance:
    """Get or create the VannaDefault singleton instance."""
    global _vanna_instance
    if _vanna_instance is None:
        # vanna_model_name = "contech1"
        # vn = VannaDefault(model=vanna_model_name, api_key='1873d412c54d469c9fa46d05aec90ed4')
        vn = VannaInstance(
            config={"api_key": os.getenv("OPENAI_API_KEY"), "model": "gpt-4o"}
        )
        sql_client = get_ims_handler_from_env()

        vn.run_sql = run_sql
        vn.run_sql_is_set = True

        df_information_schema = vn.run_sql(
            "SELECT * FROM INFORMATION_SCHEMA.COLUMNS",
            sql_client=sql_client,
            db="dob_bis",
        )
        plan = vn.get_training_plan_generic(df_information_schema)
        vn.train(plan=plan)
        _vanna_instance = vn

    return _vanna_instance


def run_sql(sql: str, sql_client: IMSQueryHandler, db: str) -> pd.DataFrame:
    """Execute SQL query using the IMS Query Handler."""
    sql_client = get_ims_handler_from_env()
    return sql_client.query(query=sql, database=db)


def image_to_base64(input_path: str):
    with open(input_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_image_embedding(input_path: str):
    with open(input_path, "rb") as image_file:
        response = embedding(
            model="cohere/embed-english-v3.0", input=[image_to_base64(input_path)]
        )

    return response.data[0]["embedding"]


def train_natural_language_to_sql() -> VannaInstance:
    vn = get_vanna_instance()

    sql_client = get_ims_handler_from_env()

    vn.run_sql = run_sql
    vn.run_sql_is_set = True

    df_information_schema = vn.run_sql(
        "SELECT * FROM INFORMATION_SCHEMA.COLUMNS", sql_client=sql_client, db="dob_bis"
    )
    plan = vn.get_training_plan_generic(df_information_schema)
    vn.train(plan=plan)
    return vn


def execute_natural_language_to_sql(prompt: str) -> pd.DataFrame:
    vn = get_vanna_instance()

    if vn.get_training_data().empty:
        vn = train_natural_language_to_sql()

    sql_client = get_ims_handler_from_env()
    sql = vn.generate_sql(prompt)

    vn.run_sql = run_sql
    vn.run_sql_is_set = True

    result = vn.run_sql(sql, sql_client=sql_client, db="dob_bis")
    return result


# Mock functions for demonstration
async def decompose_question(state: State, config: RunnableConfig) -> Dict:
    """Function to decompose the main question into sequential, query-able sub-questions."""
    # Reset state if this is a new question (detected by checking if the last message is from human)
    if state.messages and isinstance(state.messages[-1], HumanMessage):
        latest_message = state.messages[-1]  # Save the latest human message
        state.reset()
        state.messages = [latest_message]  # Restore the latest human message

    if not state.is_decomposition_done:
        configuration = Configuration.from_runnable_config(config)
        model = load_chat_model(configuration.model)

        main_question = state.messages[0].content

        # Create a system prompt for question decomposition
        system_message = SystemMessage(
            content="""You are an expert at breaking down complex construction-related questions into sequential, query-able sub-questions.

You have access to two specific NYC building data sources:

1. Vector Store containing Certificate of Occupancy (CO) PDFs for 4 specific NYC buildings
   - These COs contain official occupancy information and building details
   - Limited to only 4 buildings' CO documentation

2. Department of Buildings (DoB) BIS Database containing:
   - Building complaints
   - Violations
   - Job filing documents
   - Zoning documents
   - Virtual jobs
   - Other structured building-related data

Each sub-question should:
1. Be self-contained and answerable with a single SQL query or vector store search
2. Build on previous questions' answers when needed
3. Be ordered logically where later questions may depend on earlier answers
4. Use precise language
5. Use as few questions as possible - if the original question can be answered with a single query, do not break it down further
6. Never break down into more than 3 sub-questions
7. Only reference information available in the two data sources
8. Only output the numbered questions, nothing else
9. Make sure that you don't give ANY directions or ask for any addresses
                                       10. JUST output short questions

For example, if asked "What are the trends in permit applications across different boroughs and building types?", you would output exactly:
1. What is the distribution of permit applications by borough and building type over time?
2. What are the month-over-month changes in application volume?

However, a simple question like "How many active construction permits are there in Manhattan?" should output exactly:
1. How many active construction permits are there in Manhattan?

For questions about Certificates of Occupancy, remember you can only reference the 4 buildings in the vector store."""
        )

        # Ask the model to decompose the question
        response = await model.ainvoke(
            [
                system_message,
                HumanMessage(
                    content=f"Break down this question into sequential, query-able sub-questions: {main_question}"
                ),
            ],
            config,
        )

        # Extract sub-questions from the response
        # Assuming the model returns a numbered list, split on newlines and clean up
        sub_questions = [
            q.strip().split(". ", 1)[1] if ". " in q.strip() else q.strip()
            for q in response.content.split("\n")
            if q.strip() and any(c.isdigit() for c in q)
        ]

        return {
            "sub_questions": sub_questions,
            "is_decomposition_done": True,
            "messages": [
                AIMessage(
                    content=f"I've broken this down into {len(sub_questions)} questions:\n"
                    + "\n".join(f"{i+1}. {q}" for i, q in enumerate(sub_questions))
                )
            ],
        }
    return {}


async def query_execution(state: State, config: RunnableConfig) -> Dict:
    """Execute natural language queries using Vanna."""
    global _failed_queries

    if not state.sub_questions:
        return {"all_questions_answered": True}

    # Find next unanswered question, excluding globally failed queries
    current_question = next(
        (
            q
            for q in state.sub_questions
            if q not in state.answers and q not in _failed_queries
        ),
        None,
    )

    # If no more questions to answer
    if current_question is None:
        return {"all_questions_answered": True}

    try:
        # Execute the query using Vanna singleton
        result = execute_natural_language_to_sql(current_question)

        # Convert DataFrame to string representation for the answer
        answer = result.to_string() if not result.empty else "No results found"

        # Update state with new answer
        new_answers = dict(state.answers)
        new_answers[current_question] = answer

        # Check if this was the last question (including globally failed queries)
        all_answered = len(new_answers) + len(_failed_queries) == len(
            state.sub_questions
        )

        return {
            "answers": new_answers,
            "current_context": {"last_answered": current_question},
            "all_questions_answered": all_answered,
            "messages": [
                AIMessage(content=f"Found answer for: {current_question}\n{answer}")
            ],
        }
    except Exception as e:
        # Track failed query globally
        _failed_queries.add(current_question)

        return {
            "messages": [
                AIMessage(
                    content=f"Failed to execute query for: {current_question}\nError: {str(e)}\nSkipping this question permanently."
                )
            ]
        }


def gather_query_context(query: str):
    embedding_str = ",".join(
        str(v)
        for v in embedding(model="cohere/embed-english-v3.0", input=[query]).data[0][
            "embedding"
        ]
    )

    searches = {
        "searches": [{"query_by": "content", "q": query, "exclude_fields": "embedding"}]
    }

    search_parameters = {
        "collection": "pdfs",
        "vector_query": f"embedding:([{embedding_str}], alpha: 0.4, k: 4)",
        "per_page": 25,
    }

    results = get_typesense_client().multi_search.perform(searches, search_parameters)
    # print("RESULTS", results)
    return results


def synthesize_query(query: str):
    # Get context
    context = gather_query_context(query)
    base64_images = []

    # Get images from context
    for result in context["results"][0]["hits"]:
        image_base64 = image_to_base64(
            f"data/images/{result['document']['source_filename']}/page_{result['document']['page_number']}.png"
        )

        base64_images.append(image_base64)

    # Respond to query using images
    image_query_response = completion(
        model="openai/gpt-4o",
        # response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "Attempt to answer the following query using the following images",
            },
            {"role": "user", "content": f"Query: {query}"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    }
                    for base64_image in base64_images
                ],
            },
        ],
    )
    return image_query_response.choices[0].message.content


async def semantic_search_execution(state: State, config: RunnableConfig) -> Dict:
    """Execute semantic search queries in parallel with SQL queries."""
    if not state.sub_questions:
        return {"all_semantic_questions_answered": True}

    current_question = next(
        (q for q in state.sub_questions if q not in state.semantic_answers), None
    )

    if current_question is None:
        return {"all_semantic_questions_answered": True}

    try:
        # Use actual semantic search implementation
        semantic_answer = synthesize_query(current_question)
        print("SEMANTIC ANSWER", semantic_answer)

        # Update semantic answers
        new_semantic_answers = dict(state.semantic_answers)
        new_semantic_answers[current_question] = semantic_answer

        # Check if this was the last question
        all_answered = len(new_semantic_answers) == len(state.sub_questions)

        # Update semantic context
        new_semantic_context = dict(state.semantic_context)
        new_semantic_context["last_search"] = current_question

        return {
            "semantic_answers": new_semantic_answers,
            "semantic_context": new_semantic_context,
            "all_semantic_questions_answered": all_answered,
            "messages": [
                AIMessage(
                    content=f"Found semantic search result for: {current_question}\n{semantic_answer}"
                )
            ],
        }
    except Exception as e:
        return {
            "messages": [
                AIMessage(
                    content=f"Failed semantic search for: {current_question}\nError: {str(e)}"
                )
            ]
        }


async def synthesize_answers(state: State, config: RunnableConfig) -> Dict:
    """Synthesize all answers into a final response that directly addresses the original question."""
    try:
        print("ANSWERS", state)
        if state.all_questions_answered and state.all_semantic_questions_answered:
            print("state.semantic_answers:", state.semantic_answers)
            configuration = Configuration.from_runnable_config(config)
            model = load_chat_model(configuration.model)

            original_question = state.messages[0].content

            context = "Here are the results from our analysis:\n\n"

            # Include both SQL query results and semantic search results
            for question in state.sub_questions:
                context += f"Question: {question}\n"
                if question in state.answers:
                    context += f"SQL Query Result: {state.answers[question]}\n"
                if question in state.semantic_answers:
                    context += (
                        f"Semantic Search Result: {state.semantic_answers[question]}\n"
                    )
                context += "\n"

            system_message = SystemMessage(
                content="""You are an expert at synthesizing information to answer questions about construction and building data.
Your task is to:
1. Review the original question and available answers from both SQL queries and semantic search
2. Determine if the information is sufficient and relevant to answer the original question
3. If the information is insufficient or irrelevant, clearly state that an answer cannot be generated
4. If the information is useful, provide a clear, concise synthesis that directly answers the original question
5. Only include relevant information in your response"""
            )

            try:
                response = await model.ainvoke(
                    [
                        system_message,
                        HumanMessage(
                            content=f"""Original Question: {original_question}

Available Information:
{context}

Please synthesize this information to answer the original question, or indicate if an answer cannot be generated."""
                        ),
                    ],
                    config,
                )
            except Exception as e:
                return {
                    "messages": [AIMessage(content=f"Error during synthesis: {str(e)}")]
                }

            # Reset flags and state for next question
            return {
                "messages": [AIMessage(content=response.content)],
                "all_questions_answered": False,
                "all_semantic_questions_answered": False,
                "is_decomposition_done": False,
                "sub_questions": [],
                "answers": {},
                "semantic_answers": {},
                "sql_context": {},
                "semantic_context": {},
            }
    except Exception as e:
        return {
            "messages": [
                AIMessage(content=f"Unexpected error during synthesis: {str(e)}")
            ]
        }
    return {
        "messages": [
            AIMessage(
                content="Still gathering information from queries and semantic search..."
            )
        ]
    }


# Define the function that calls the model


async def call_model(
    state: State, config: RunnableConfig
) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    configuration = Configuration.from_runnable_config(config)

    # Initialize the model with tool binding. Change the model or add more tools here.
    model = load_chat_model(configuration.model).bind_tools(TOOLS)

    # Format the system prompt. Customize this to change the agent's behavior.
    system_message = configuration.system_prompt.format(
        system_time=datetime.now(tz=timezone.utc).isoformat()
    )

    # Get the model's response
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages], config
        ),
    )

    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    # Return the model's response as a list to be added to existing messages
    return {"messages": [response]}


# Define a new graph

builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Add nodes
builder.add_node("decompose", decompose_question)
builder.add_node("execute_query", query_execution)
builder.add_node("semantic_search", semantic_search_execution)
builder.add_node("synthesize", synthesize_answers)
builder.add_node("call_model", call_model)
builder.add_node("tools", ToolNode(TOOLS))

# Set the entrypoint
builder.add_edge("__start__", "decompose")


def route_decomposition(state: State) -> List[str]:
    """Route after question decomposition."""
    if state.is_decomposition_done:
        # Fan out to both query execution and semantic search
        return ["execute_query", "semantic_search"]
    return ["call_model"]


def route_execution(state: State) -> Literal["synthesize", "execute_query"]:
    """Route after query execution."""
    if state.all_questions_answered:
        return "synthesize"
    return "execute_query"


def route_semantic_search(state: State) -> Literal["synthesize", "semantic_search"]:
    """Route after semantic search."""
    if state.all_semantic_questions_answered:
        return "synthesize"
    return "semantic_search"


def route_synthesis(state: State) -> Literal["__end__"]:
    """Route after synthesis."""
    return "__end__"


def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """Route after model output."""
    if state.is_last_step:
        return "__end__"
    return "tools"


# Add conditional edges for branching logic
builder.add_conditional_edges(
    "decompose", route_decomposition, ["execute_query", "semantic_search", "call_model"]
)

# Add recursive edges for execution nodes
builder.add_conditional_edges(
    "execute_query",
    route_execution,
    {"synthesize": "synthesize", "execute_query": "execute_query"},
)

builder.add_conditional_edges(
    "semantic_search",
    route_semantic_search,
    {"synthesize": "synthesize", "semantic_search": "semantic_search"},
)

# Add edge to end
builder.add_edge("synthesize", "__end__")

# Add regular edge for tools back to model
builder.add_edge("tools", "call_model")

# Compile the builder into an executable graph
graph = builder.compile()
graph.name = "Construction Info ReAct Agent"

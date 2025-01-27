"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from datetime import datetime, timezone
from typing import Dict, List, Literal, cast, Union, Set, Annotated
import os
import base64
import json

import pandas as pd

# from vanna.remote import VannaDefault
# from vanna.flask import VannaFlaskApp
from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore
from react_agent.gryps_utils import get_ims_handler_from_env, IMSQueryHandler

from litellm import acompletion, embedding, completion
import typesense
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Send
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


async def extract_question(state: State, config: RunnableConfig) -> State:
    """Extract the question from the state."""
    extract_question_response = await acompletion(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "Distill the user's input to one, consice question",
            },
            # {
            #     "role": "system",
            #     "content": "If there is no question or the question is not related to NYC building and construction data, reply to the user's input",
            # },
            {
                "role": "system",
                "content": "Return the question in a JSON object with a 'question' key and a 'reply' key if there is no question",
            },
            {
                "role": "user",
                "content": "Here are the last messages in the conversation:",
            },
        ]
        + [
            {"role": "user", "content": f"{msg.type}: {msg.content}"}
            for msg in state.messages[-8:]
        ],
    )

    json_response = json.loads(extract_question_response.choices[0].message.content)
    print("Extract Question Response:", json_response)

    return {
        "question": json_response.get("question"),
        "reply": json_response.get("reply"),
    }


# Mock functions for demonstration
async def plan_question(state: State, config: RunnableConfig) -> State:
    """Function to decompose the main question into sequential, query-able sub-questions."""
    # Reset state if this is a new question (detected by checking if the last message is from human)
    # TODO: retrieve question from state
    # if state.messages and isinstance(state.messages[-1], HumanMessage):
    #     latest_message = state.messages[-1]  # Save the latest human message
    #     state.reset()
    #     state.messages = [latest_message]  # Restore the latest human message

    print("ALLOWED CONTEXT", state.allowed_context)
    # configuration = Configuration.from_runnable_config(config)
    # model = load_chat_model(configuration.model)

    main_question = state.question

    messages = [
        {
            "role": "system",
            "content": "You are an expert at breaking down complex construction-related questions into sequential, query-able sub-questions.",
        },
        {
            "role": "system",
            "content": "You have access to the following NYC building data sources:",
        },
    ]

    all_context_str = " ".join(state.allowed_context)
    possible_context = []

    if "data/Certificates of Occupancy" in all_context_str:
        messages.append(
            {
                "role": "system",
                "content": "1. Vector Store containing Certificate of Occupancy (CO) PDFs for 4 specific NYC buildings (VECTOR_STORE) \n"
                "- These COs contain official occupancy information and building details\n"
                "- Limited to only 4 buildings' CO documentation",
            }
        )
        possible_context.append("VECTOR_STORE")

    if "databases/" in all_context_str:
        messages.append(
            {
                "role": "system",
                "content": "2. Department of Buildings (DoB) BIS Database (SQL_DATABASE) containing: \n"
                "- Job filing documents\n"
                "- Zoning documents\n"
                "- Virtual jobs\n"
                "- Other structured building-related data",
            }
        )
        possible_context.append("SQL_DATABASE")

    messages.extend(
        [
            {
                "role": "system",
                "content": f"Determine which data sources you will use to answer the question: {main_question}",
            },
            {
                "role": "system",
                "content": "Return the data sources you will use to answer the question in a JSON object"
                f"with a 'data_sources' key containing a list of possible values: {", ".join(possible_context)}",
            },
            {
                "role": "system",
                "content": "Example: {'data_sources': ['VECTOR_STORE', 'SQL_DATABASE']}",
            },
        ]
    )

    # TODO: restrict prompt depending on allowed context
    query_routing_response = await acompletion(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=messages,
    )

    json_response = json.loads(query_routing_response.choices[0].message.content)
    print("Query Routing Response:", json_response)

    return {
        "selected_tools": json_response.get("data_sources", []),
    }


async def determine_sql_subquestions(state: State, config: RunnableConfig) -> State:
    """Determine the SQL subquestions to answer."""
    available_databases = [
        ctx.replace("databases/", "")
        for ctx in state.allowed_context
        if "databases/" in ctx
    ]

    sql_subquestions_response = await acompletion(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are creating subqueries to answer a question about construction data using a SQL database",
            },
            {
                "role": "system",
                "content": "Each query should build on previous questions' answers when needed",
            },
            {"role": "system", "content": "Do not break down into more than 3 queries"},
            {
                "role": "system",
                "content": f"Your database has access to the following tables: {", ".join(available_databases)}",
            },
            {"role": "user", "content": "Question: " + state.question},
            {
                "role": "user",
                "content": "Return the sub-questions in a JSON object with a 'sub_questions' key containing a list of sub-questions",
            },
            {
                "role": "user",
                "content": "Example: {'sub_questions': ['What is the total number of permits issued in Manhattan?', 'What is the average permit value in Manhattan?']}",
            },
        ],
    )

    json_response = json.loads(sql_subquestions_response.choices[0].message.content)
    print("SQL Subquestions Response:", json_response)

    # replace sub_questions
    # state.sql_sub_questions = json_response.get("sub_questions", [])
    state.sql_sub_questions.clear()
    return {
        "sql_sub_questions": json_response.get("sub_questions", []),
    }


async def determine_semantic_subquestions(
    state: State, config: RunnableConfig
) -> State:
    """Determine the semantic subquestions to answer."""
    available_documents = [
        ctx.replace("data/Certificates of Occupancy", "")
        for ctx in state.allowed_context
        if "data/Certificates of Occupancy" in ctx
    ]

    sql_subquestions_response = await acompletion(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are creating subqueries to answer a question about construction data using a vector store",
            },
            {
                "role": "system",
                "content": "Each query should build on previous questions' answers when needed",
            },
            {"role": "system", "content": "Do not break down into more than 3 queries"},
            {
                "role": "system",
                "content": f"Your vector store has access to the following documents: {", ".join(available_documents)}",
            },
            {"role": "user", "content": "Question: " + state.question},
            {
                "role": "user",
                "content": "Return the sub-questions in a JSON object with a 'sub_questions' key containing a list of sub-questions",
            },
            {
                "role": "user",
                "content": "Example: {'sub_questions': ['What is the total number of permits issued in Manhattan?', 'What is the average permit value in Manhattan?']}",
            },
        ],
    )

    json_response = json.loads(sql_subquestions_response.choices[0].message.content)
    print("Semantic Subquestions Response:", json_response)

    # Replace sub_questions
    # state.semantic_sub_questions = json_response.get("sub_questions", [])
    state.semantic_sub_questions.clear()
    return {
        "semantic_sub_questions": json_response.get("sub_questions", []),
    }


async def query_execution(state: State, config: RunnableConfig) -> State:
    """Execute natural language queries using Vanna."""
    global _failed_queries

    # if not state.sql_sub_questions:
    #     return {"all_questions_answered": True}

    # Find next unanswered question, excluding globally failed queries
    # current_question = next(
    #     (
    #         q
    #         for q in state.sub_questions
    #         if q not in state.answers and q not in _failed_queries
    #     ),
    #     None,
    # )
    current_question = state["current_sql_sub_question"]

    # If no more questions to answer
    # if current_question is None:
    #     return {"all_questions_answered": True}
    print("CURRENT QUESTION", current_question)

    try:
        # Execute the query using Vanna singleton
        result = execute_natural_language_to_sql(current_question)

        # Convert DataFrame to string representation for the answer
        answer = result.to_string() if not result.empty else "No results found"
        print("SQL ANSWER", f"{current_question}: {answer}")

        # state.sql_answers.append(f"{current_question}: {answer}")
        return {
            "sql_answers": [f"{current_question}: {answer}"],
        }
        # return {
        #     "sql_answers": [f"{current_question}: {answer}"],
        # }

        # Update state with new answer
        # new_answers = dict(state.answers)
        # new_answers[current_question] = answer

        # # Check if this was the last question (including globally failed queries)
        # all_answered = len(new_answers) + len(_failed_queries) == len(
        #     state.sub_questions
        # )

        # return {
        #     "answers": new_answers,
        #     "current_context": {"last_answered": current_question},
        #     "all_questions_answered": all_answered,
        #     "messages": [
        #         AIMessage(content=f"Found answer for: {current_question}\n{answer}")
        #     ],
        # }
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


def gather_query_context(query: str, allowed_filenames: List[str]):
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
        "filter_by": f"source_filename:{[f'`{filename}`' for filename in allowed_filenames]}",
        "per_page": 25,
    }

    results = get_typesense_client().multi_search.perform(searches, search_parameters)
    # print("RESULTS", results)
    return results


async def synthesize_query(query: str, allowed_filenames: List[str]):
    # Get context
    context = gather_query_context(query, allowed_filenames)
    base64_images = []

    # Get images from context
    for result in context["results"][0]["hits"]:
        image_base64 = image_to_base64(
            f"data/images/{result['document']['source_filename']}/page_{result['document']['page_number']}.png"
        )

        base64_images.append(image_base64)

    # Respond to query using images
    image_query_response = await acompletion(
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


async def semantic_search_execution(state: State, config: RunnableConfig) -> State:
    """Execute semantic search queries in parallel with SQL queries."""
    # if not state.sub_questions:
    #     return {"all_semantic_questions_answered": True}

    # current_question = next(
    #     (q for q in state.sub_questions if q not in state.semantic_answers), None
    # )

    # if current_question is None:
    #     return {"all_semantic_questions_answered": True}
    print("SEMANTIC STATE EXECUTION:", state)

    current_question = state["current_semantic_sub_question"]
    allowed_filenames = [
        os.path.basename(ctx)
        for ctx in state["allowed_context"]
        if "data/Certificates of Occupancy" in ctx
    ]

    try:
        # Use actual semantic search implementation
        semantic_answer = await synthesize_query(current_question, allowed_filenames)
        print("SEMANTIC ANSWER", f"{current_question}: {semantic_answer}")

        return {
            "semantic_answers": [f"{current_question}: {semantic_answer}"],
        }

        # Update semantic answers
        # new_semantic_answers = dict(state.semantic_answers)
        # new_semantic_answers[current_question] = semantic_answer

        # Check if this was the last question
        # all_answered = len(new_semantic_answers) == len(state.sub_questions)

        # # Update semantic context
        # new_semantic_context = dict(state.semantic_context)
        # new_semantic_context["last_search"] = current_question

        # # TODO: include cited sources in state

        # return {
        #     "semantic_answers": new_semantic_answers,
        #     "semantic_context": new_semantic_context,
        #     "all_semantic_questions_answered": all_answered,
        #     "messages": [
        #         AIMessage(
        #             content=f"Found semantic search result for: {current_question}\n{semantic_answer}"
        #         )
        #     ],
        # }
    except Exception as e:
        return {
            "messages": [
                AIMessage(
                    content=f"Failed semantic search for: {current_question}\nError: {str(e)}"
                )
            ]
        }


async def synthesize_answers(state: State, config: RunnableConfig) -> State:
    """Synthesize all answers into a final response that directly addresses the original question."""
    print("SYNTHESIZE ANSWERS STATE", state)

    synthesize_answer_response = await acompletion(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are trying to answer the following question: "
                + state.question,
            },
            {
                "role": "system",
                "content": "You have answers to the following questions:\n\n-"
                + "\n- ".join(state.semantic_answers + state.sql_answers),
            },
            {
                "role": "system",
                "content": "Determine if the information is sufficient and relevant to answer the original question",
            },
            {
                "role": "user",
                "content": "If it is not sufficient or relevant, create a new question to ask the user or the model",
            },
            {
                "role": "user",
                "content": "Return the final answer in a JSON object with a 'reply' key containing either the final answer if there is sufficient information,"
                "or a 'user_question' key containing follow-up questions to ask the user, or a 'model_question' to ask the model",
            },
        ],
    )

    json_response = json.loads(synthesize_answer_response.choices[0].message.content)

    return {
        "reply": json_response.get("reply"),
        "user_question": json_response.get("user_question"),
        "question": json_response.get("model_question"),
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
builder.add_node("extract_question", extract_question)
builder.add_node("plan", plan_question)
builder.add_node("determine_sql_subquestions", determine_sql_subquestions)
builder.add_node("determine_semantic_subquestions", determine_semantic_subquestions)
builder.add_node("execute_query", query_execution)
builder.add_node("semantic_search", semantic_search_execution)
builder.add_node("synthesize", synthesize_answers)
builder.add_node("call_model", call_model)
builder.add_node("tools", ToolNode(TOOLS))

# Set the entrypoint
builder.add_edge("__start__", "extract_question")


def route_question(state: State) -> Literal["__end__", "plan"]:
    """If no question extracted and reply already exists, end the conversation."""
    if (not state.question or state.question == "") and (
        state.reply and state.reply != ""
    ):
        return "__end__"

    return "plan"


def route_decomposition(state: State) -> List[str]:
    """Route after question decomposition."""
    # Fan out to both query execution and semantic search
    routes = []

    if "VECTOR_STORE" in state.selected_tools:
        routes.append("determine_semantic_subquestions")

    if "SQL_DATABASE" in state.selected_tools:
        routes.append("determine_sql_subquestions")

    if routes == []:
        return ["synthesize"]

    return routes


def route_sql_subqueries(state: State) -> List[str]:
    return [
        Send(
            "execute_query",
            {"current_sql_sub_question": sq, "allowed_context": state.allowed_context},
        )
        for sq in state.sql_sub_questions
    ]


def route_semantic_subqueries(state: State) -> List[str]:
    return [
        Send(
            "semantic_search",
            {
                "current_semantic_sub_question": sq,
                "allowed_context": state.allowed_context,
            },
        )
        for sq in state.semantic_sub_questions
    ]


# def route_execution(state: State) -> Literal["synthesize", "execute_query"]:
#     """Route after query execution."""
#     if state.all_questions_answered:
#         return "synthesize"
#     return "execute_query"


# def route_semantic_search(state: State) -> Literal["synthesize", "semantic_search"]:
#     """Route after semantic search."""
#     if state.all_semantic_questions_answered:
#         return "synthesize"
#     return "semantic_search"


def route_synthesis(state: State) -> Literal["plan", "__end__"]:
    """Route after synthesis."""
    print("ROUTE SYNTHESIS:", state)

    # If still have a question, go back to plan
    if state.question:
        return "plan"

    return "__end__"


def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """Route after model output."""
    if state.is_last_step:
        return "__end__"
    return "tools"


"""
1. extract_question
2. determine plan
3a. If using a vector store, determine semantic subquestions
3b. If using a SQL database, determine SQL subquestions
4a. Execute semantic search for each subquestion
4b. Execute SQL query for each subquestion
5. Synthesize answers
5. Check if all original question have been answered or if follow-up questions need to be asked to the user or to the model
6. If follow-up questions can be asked to model, return to step 2.
7. If original question has been answered or follow up questions need to be asked to user, return response.
"""

# Add conditional edges for branching logic
builder.add_conditional_edges("extract_question", route_question, ["__end__", "plan"])
builder.add_conditional_edges(
    "plan",
    route_decomposition,
    ["determine_semantic_subquestions", "determine_sql_subquestions", "call_model"],
)

builder.add_conditional_edges(
    "determine_sql_subquestions",
    route_sql_subqueries,
    ["execute_query"],
)

builder.add_conditional_edges(
    "determine_semantic_subquestions",
    route_semantic_subqueries,
    ["semantic_search"],
)

builder.add_edge("execute_query", "synthesize")
builder.add_edge("semantic_search", "synthesize")

# Add recursive edges for execution nodes
# builder.add_conditional_edges(
#     "execute_query",
#     route_execution,
#     {"synthesize": "synthesize", "execute_query": "execute_query"},
# )

# builder.add_conditional_edges(
#     "semantic_search",
#     route_semantic_search,
#     {"synthesize": "synthesize", "semantic_search": "semantic_search"},
# )

# Add edge to end
# builder.add_edge("synthesize", "__end__")
builder.add_conditional_edges("synthesize", route_synthesis, ["plan", "__end__"])

# Add regular edge for tools back to model
builder.add_edge("tools", "call_model")

# Compile the builder into an executable graph
graph = builder.compile()
graph.name = "Construction Info ReAct Agent"

"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from datetime import datetime, timezone
from typing import Dict, List, Literal, cast, Union, Set
import pandas as pd
from vanna.remote import VannaDefault
from vanna.flask import VannaFlaskApp
from react_agent.gryps_utils import get_ims_handler_from_env, IMSQueryHandler
 
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.state import InputState, State
from react_agent.tools import TOOLS
from react_agent.utils import load_chat_model

# Global VannaDefault instance
_vanna_instance = None
_failed_queries: Set[str] = set()

def get_vanna_instance() -> VannaDefault:
    """Get or create the VannaDefault singleton instance."""
    global _vanna_instance
    if _vanna_instance is None:
        vanna_model_name = 'lmao-model-watup'    
        vn = VannaDefault(model=vanna_model_name, api_key='1873d412c54d469c9fa46d05aec90ed4')
        sql_client = get_ims_handler_from_env()

        vn.run_sql = run_sql
        vn.run_sql_is_set = True
        
        df_information_schema = vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS", sql_client=sql_client, db='dob_bis')
        plan = vn.get_training_plan_generic(df_information_schema)
        vn.train(plan=plan)
        _vanna_instance = vn
    return _vanna_instance

def run_sql(sql: str, sql_client: IMSQueryHandler, db: str) -> pd.DataFrame:
    """Execute SQL query using the IMS Query Handler."""
    sql_client = get_ims_handler_from_env()
    return sql_client.query(query=sql, database=db)

def train_natural_language_to_sql() -> VannaDefault:
    vanna_model_name = 'contech'    
    vn = VannaDefault(model=vanna_model_name, api_key='27db8165836f4265b58df4ef4279b3e2')
    sql_client = get_ims_handler_from_env()

    vn.run_sql = run_sql
    vn.run_sql_is_set = True
    
    df_information_schema = vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS",sql_client=sql_client, db='dob_bis')
    plan = vn.get_training_plan_generic(df_information_schema)
    vn.train(plan=plan)
    return vn

def execute_natural_language_to_sql(prompt: str) -> pd.DataFrame:
    vanna_model_name = 'contech'    
    vn = VannaDefault(model=vanna_model_name, api_key='27db8165836f4265b58df4ef4279b3e2')
    if vn.get_training_data().empty:
        vn = train_natural_language_to_sql()
    sql_client = get_ims_handler_from_env()
    sql = vn.generate_sql(prompt)
    
    vn.run_sql = run_sql
    vn.run_sql_is_set = True
    
    result = vn.run_sql(sql,sql_client=sql_client, db='dob_bis')
    return result

# Mock functions for demonstration
async def decompose_question(state: State, config: RunnableConfig) -> Dict:
    """Function to decompose the main question into sequential, query-able sub-questions."""
    if not state.is_decomposition_done:
        configuration = Configuration.from_runnable_config(config)
        model = load_chat_model(configuration.model)
        
        main_question = state.messages[0].content
        
        # Create a system prompt for question decomposition
        system_message = SystemMessage(content="""You are an expert at breaking down complex construction-related questions into sequential, query-able sub-questions.

Each sub-question should:
1. Be self-contained and answerable with a single SQL query
2. Build on previous questions' answers when needed
3. Be ordered logically where later questions may depend on earlier answers
4. Use precise language
5. Use as few questions as possible - if the original question can be answered with a single query, do not break it down further
6. Never break down into more than 3 sub-questions
7. Only output the numbered questions, nothing else

For example, if asked "What are the trends in permit applications across different boroughs and building types?", you would output exactly:
1. What is the distribution of permit applications by borough and building type over time?
2. What are the month-over-month changes in application volume?

However, a simple question like "How many active construction permits are there in Manhattan?" should output exactly:
1. How many active construction permits are there in Manhattan?""")

        
        # Ask the model to decompose the question
        response = await model.ainvoke(
            [
                system_message,
                HumanMessage(content=f"Break down this question into sequential, query-able sub-questions: {main_question}")
            ],
            config
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
            "messages": [AIMessage(content=f"I've broken this down into {len(sub_questions)} questions:\n" + "\n".join(f"{i+1}. {q}" for i, q in enumerate(sub_questions)))]
        }
    return {}

async def query_execution(state: State, config: RunnableConfig) -> Dict:
    """Execute natural language queries using Vanna."""
    global _failed_queries
    
    if not state.sub_questions:
        return {"all_questions_answered": True}
        
    # Find next unanswered question, excluding globally failed queries
    current_question = next((q for q in state.sub_questions if q not in state.answers and q not in _failed_queries), None)
    
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
        all_answered = len(new_answers) + len(_failed_queries) == len(state.sub_questions)
        
        return {
            "answers": new_answers,
            "current_context": {"last_answered": current_question},
            "all_questions_answered": all_answered,
            "messages": [AIMessage(content=f"Found answer for: {current_question}\n{answer}")]
        }
    except Exception as e:
        # Track failed query globally
        _failed_queries.add(current_question)
        
        return {
            "messages": [AIMessage(content=f"Failed to execute query for: {current_question}\nError: {str(e)}\nSkipping this question permanently.")]
        }

async def synthesize_answers(state: State, config: RunnableConfig) -> Dict:
    """Synthesize all answers into a final response that directly addresses the original question."""
    if state.all_questions_answered:
        configuration = Configuration.from_runnable_config(config)
        model = load_chat_model(configuration.model)
        
        # Get the original question from the first message
        original_question = state.messages[0].content
        
        # Prepare the context for synthesis
        context = "Here are the results from our analysis:\n\n"
        for question, answer in state.answers.items():
            context += f"Question: {question}\nAnswer: {answer}\n\n"
            
        # Create a system prompt for synthesis
        system_message = SystemMessage(content="""You are an expert at synthesizing information to answer questions about construction and building data.
Your task is to:
1. Review the original question and available answers
2. Determine if the information is sufficient and relevant to answer the original question
3. If the information is insufficient or irrelevant, clearly state that an answer cannot be generated
4. If the information is useful, provide a clear, concise synthesis that directly answers the original question
5. Only include relevant information in your response""")
        
        # Ask the model to synthesize
        response = await model.ainvoke(
            [
                system_message,
                HumanMessage(content=f"""Original Question: {original_question}

Available Information:
{context}

Please synthesize this information to answer the original question, or indicate if an answer cannot be generated.""")
            ],
            config
        )
        
        return {
            "messages": [AIMessage(content=response.content)]
        }
    return {}

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
builder.add_node("synthesize", synthesize_answers)
builder.add_node("call_model", call_model)
builder.add_node("tools", ToolNode(TOOLS))

# Set the entrypoint
builder.add_edge("__start__", "decompose")

def route_decomposition(state: State) -> Literal["execute_query", "call_model"]:
    """Route after decomposition."""
    if state.is_decomposition_done:
        return "execute_query"
    return "call_model"

def route_execution(state: State) -> Literal["synthesize", "execute_query"]:
    """Route after query execution."""
    if state.all_questions_answered:
        return "synthesize"
    return "execute_query"

def route_synthesis(state: State) -> Literal["__end__"]:
    """Route after synthesis."""
    return "__end__"

def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """Determine the next node based on the model's output."""
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    if not last_message.tool_calls:
        return "__end__"
    return "tools"

# Add conditional edges for routing logic
builder.add_conditional_edges(
    "decompose",
    route_decomposition,
    {
        "execute_query": "execute_query",
        "call_model": "call_model"
    }
)

builder.add_conditional_edges(
    "execute_query",
    route_execution,
    {
        "synthesize": "synthesize",
        "execute_query": "execute_query"
    }
)

builder.add_conditional_edges(
    "synthesize",
    route_synthesis,
    {
        "__end__": "__end__"
    }
)

builder.add_conditional_edges(
    "call_model",
    route_model_output,
    {
        "__end__": "__end__",
        "tools": "tools"
    }
)

# Add regular edge for tools back to model
builder.add_edge("tools", "call_model")

# Compile the builder into an executable graph
graph = builder.compile()
graph.name = "Construction Info ReAct Agent"

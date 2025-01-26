"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Self

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated


@dataclass
class InputState:
    """Defines the input state for the agent, representing a narrower interface to the outside world.

    This class is used to define the initial state and structure of incoming data.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )
    """
    Messages tracking the primary execution state of the agent.

    Typically accumulates a pattern of:
    1. HumanMessage - user input
    2. AIMessage with .tool_calls - agent picking tool(s) to use to collect information
    3. ToolMessage(s) - the responses (or errors) from the executed tools
    4. AIMessage without .tool_calls - agent responding in unstructured format to the user
    5. HumanMessage - user responds with the next conversational turn

    Steps 2-5 may repeat as needed.

    The `add_messages` annotation ensures that new messages are merged with existing ones,
    updating by ID to maintain an "append-only" state unless a message with the same ID is provided.
    """


@dataclass
class State(InputState):
    """Represents the complete state of the agent, extending InputState with additional attributes.

    This class can be used to store any information needed throughout the agent's lifecycle.
    """

    is_last_step: IsLastStep = field(default=False)
    """
    Indicates whether the current step is the last one before the graph raises an error.

    This is a 'managed' variable, controlled by the state machine rather than user code.
    It is set to 'True' when the step count reaches recursion_limit - 1.
    """

    sub_questions: list[str] = field(default_factory=list)
    """List of decomposed questions that need to be answered"""
    
    answers: dict[str, str] = field(default_factory=dict)
    """Dictionary mapping questions to their SQL query answers"""
    
    semantic_answers: dict[str, str] = field(default_factory=dict)
    """Dictionary mapping questions to their semantic search answers"""
    
    sql_context: dict = field(default_factory=dict)
    """Context for SQL query execution"""
    
    semantic_context: dict = field(default_factory=dict)
    """Context for semantic search execution"""
    
    is_decomposition_done: bool = field(default=False)
    """Flag indicating if question decomposition is complete"""
    
    all_questions_answered: bool = field(default=False)
    """Flag indicating if all SQL sub-questions have been answered"""
    
    all_semantic_questions_answered: bool = field(default=False)
    """Flag indicating if all semantic search sub-questions have been answered"""

    def reset(self) -> Self:
        """Reset all mutable state fields to their default values.
        
        Returns:
            Self: The state instance with reset values
        """
        self.messages = []
        self.sub_questions = []
        self.answers = {}
        self.semantic_answers = {}
        self.sql_context = {}
        self.semantic_context = {}
        self.is_decomposition_done = False
        self.all_questions_answered = False
        self.all_semantic_questions_answered = False
        return self

    # Additional attributes can be added here as needed.
    # Common examples include:
    # retrieved_documents: List[Document] = field(default_factory=list)
    # extracted_entities: Dict[str, Any] = field(default_factory=dict)
    # api_connections: Dict[str, Any] = field(default_factory=dict)

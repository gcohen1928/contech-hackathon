"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
import operator
from typing import List, Optional, Sequence, Self

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated


CONTEXT_IDS = [
    "data/Certificates of Occupancy/161 W. 56th Street/M000093215.PDF",
    "data/Certificates of Occupancy/161 W. 56th Street/M00093381B.PDF",
    "data/Certificates of Occupancy/161 W. 56th Street/M000093520.PDF",
    "data/Certificates of Occupancy/152 W. 57th Street/M00103924B.PDF",
    "databases/job_filings",
    "databases/complaints",
    "databases/violations",
]


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

    allowed_context: List[str] = field(default_factory=list)
    """List of allowed context for the agent to use"""


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

    # TODO: track count of recursive rounds

    question: Optional[str] = field(default=None)
    """The current question being asked of the agent"""

    reply: Optional[str] = field(default=None)
    """The current reply from the agent"""

    selected_tools: List[str] = field(default_factory=list)
    """List of tools selected by the agent"""

    sql_sub_questions: List[str] = field(default_factory=list)
    """List of SQL sub-questions to answer"""

    current_sql_sub_question: Optional[str] = field(default=None)
    """The current SQL sub-question being answered"""

    sql_answers: Annotated[List[str], operator.add] = field(default_factory=list)
    """List of SQL answers"""

    semantic_sub_questions: List[str] = field(default_factory=list)
    """List of semantic sub-questions to answer"""

    current_semantic_sub_question: Optional[str] = field(default=None)
    """The current semantic sub-question being answered"""

    semantic_answers: Annotated[List[str], operator.add] = field(default_factory=list)
    """List of semantic answers"""

    user_question: Optional[str] = field(default=None)
    """The current user question being asked of the agent"""

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

        self.is_decomposition_done = False
        self.all_questions_answered = False
        self.all_semantic_questions_answered = False
        return self

    # Additional attributes can be added here as needed.
    # Common examples include:
    # retrieved_documents: List[Document] = field(default_factory=list)
    # extracted_entities: Dict[str, Any] = field(default_factory=dict)
    # api_connections: Dict[str, Any] = field(default_factory=dict)

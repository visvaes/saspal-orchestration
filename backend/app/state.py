from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

AgentName = Literal["general_chat", "research", "data_analysis", "task_planning"]


class WorkflowInput(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class WorkflowState(WorkflowInput, total=False):
    agent_plan: list[AgentName]
    agent_used: AgentName
    pending_agents: list[AgentName]
    completed_agents: list[AgentName]
    agent_results: dict[str, str]
    steps: int
    final_response: str
    agents_used: list[AgentName]


class WorkflowOutput(TypedDict):
    final_response: str
    agent_used: AgentName
    agents_used: list[AgentName]

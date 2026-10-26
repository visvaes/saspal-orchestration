from collections.abc import Callable, Hashable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, StateGraph

from app.agents import data_analysis, general_chat, research, task_planning
from app.agents.common import ModelProvider, message_text
from app.agents.supervisor import MAX_AGENT_CALLS, supervise
from app.llm import get_chat_model
from app.state import AgentName, WorkflowInput, WorkflowOutput, WorkflowState

AGENT_NODES: dict[AgentName, Callable[..., dict[str, object]]] = {
    "general_chat": general_chat.run,
    "research": research.run,
    "data_analysis": data_analysis.run,
    "task_planning": task_planning.run,
}
FINALIZE = "finalize"

FINAL_RESPONSE_INSTRUCTIONS = """Combine the specialist results into one useful,
coherent response to the user's latest request. Resolve overlap, preserve useful
details, and be clear about uncertainty. Do not add unsupported facts or claim
that external tools or services were used."""


def _next_node(state: WorkflowState) -> Hashable:
    pending = state.get("pending_agents", [])
    if not pending:
        return FINALIZE
    if state.get("steps", 0) >= MAX_AGENT_CALLS:
        return FINALIZE
    return pending[0]


def _finalize(
    state: WorkflowState,
    get_model: ModelProvider,
) -> dict[str, object]:
    results = state.get("agent_results", {})
    plan = state.get("agent_plan", [])
    if not results:
        raise RuntimeError("The supervisor did not produce any agent results.")

    agent_used = state.get("agent_used")
    if agent_used is None:
        completed_agents = state.get("completed_agents", [])
        if not completed_agents:
            raise RuntimeError("The workflow completed without selecting an agent.")
        agent_used = completed_agents[0]

    if len(results) == 1:
        response = next(iter(results.values()))
    else:
        ordered_results = [
            f"{agent}:\n{results[agent]}"
            for agent in plan
            if agent in results
        ]
        answer = get_model().invoke(
            [
                SystemMessage(
                    content=FINAL_RESPONSE_INSTRUCTIONS
                    + "\n\nSpecialist results:\n"
                    + "\n\n".join(ordered_results)
                ),
                *state["messages"],
            ]
        )
        response = message_text(answer)
        if not response.strip():
            raise RuntimeError("The response coordinator returned an empty response.")

    return {
        "final_response": response,
        "agent_used": agent_used,
        "agents_used": state.get("completed_agents", []),
    }


def build_graph(model: BaseChatModel | None = None):
    """Compile the workflow; an injected model keeps tests independent of OpenAI."""
    get_model: ModelProvider
    if model is None:
        get_model = get_chat_model
    else:
        get_model = lambda: model

    builder = StateGraph(
        WorkflowState,
        input_schema=WorkflowInput,
        output_schema=WorkflowOutput,
    )
    builder.add_node("supervisor", lambda state: supervise(state, get_model()))
    for agent_name, agent_node in AGENT_NODES.items():
        builder.add_node(
            agent_name,
            lambda state, node=agent_node: node(state, get_model),
        )
    builder.add_node("finalize", lambda state: _finalize(state, get_model))
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        _next_node,
        {**{name: name for name in AGENT_NODES}, FINALIZE: FINALIZE},
    )
    for agent_name in AGENT_NODES:
        builder.add_conditional_edges(
            agent_name,
            _next_node,
            {**{name: name for name in AGENT_NODES}, FINALIZE: FINALIZE},
        )
    builder.add_edge("finalize", END)
    return builder.compile()

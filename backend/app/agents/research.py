from app.agents.common import ModelProvider, run_specialist
from app.state import WorkflowState

INSTRUCTIONS = """You are the research specialist. Produce a structured, concise
analysis of the user's question. Separate well-established information from
assumptions or uncertainty. You have no live internet or external research tools;
do not invent citations, current facts, or claim that you verified information
online. State when a question needs up-to-date sources."""


def run(state: WorkflowState, get_model: ModelProvider) -> dict[str, object]:
    return run_specialist(state, "research", INSTRUCTIONS, get_model)

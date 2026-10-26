from app.agents.common import ModelProvider, run_specialist
from app.state import WorkflowState

INSTRUCTIONS = """You are the task planning specialist. Turn the user's objective
into a practical, ordered plan with clear, actionable steps. Call out important
dependencies, decisions, and risks. Do not claim that you performed any of the
planned actions."""


def run(state: WorkflowState, get_model: ModelProvider) -> dict[str, object]:
    return run_specialist(state, "task_planning", INSTRUCTIONS, get_model)

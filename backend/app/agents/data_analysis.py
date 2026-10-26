from app.agents.common import ModelProvider, run_specialist
from app.state import WorkflowState

INSTRUCTIONS = """You are the data analysis specialist. Analyze only structured
data and facts supplied in the conversation. Show relevant calculations and
summarize the findings clearly. Identify missing, ambiguous, or inconsistent
inputs. Never invent records or imply access to an unconnected database."""


def run(state: WorkflowState, get_model: ModelProvider) -> dict[str, object]:
    return run_specialist(state, "data_analysis", INSTRUCTIONS, get_model)

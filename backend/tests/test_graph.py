import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from app.graph import build_graph
from app.agents.supervisor import RoutingDecision


class FakeStructuredModel:
    def __init__(self, agents):
        self.agents = agents
        self.invoked = []

    def invoke(self, messages):
        self.invoked.append(messages)
        return {} if self.agents is None else {"agents": self.agents}


class FakeModel:
    def __init__(self, agents=("general_chat",), response="Mocked specialist response"):
        self.invoked = []
        self.structured_model = FakeStructuredModel(agents)
        self.response = response

    def with_structured_output(self, schema):
        return self.structured_model

    def invoke(self, messages):
        self.invoked.append(messages)
        return AIMessage(content=self.response)


def test_graph_uses_injected_model_for_supervisor_and_agent():
    model = FakeModel()

    result = build_graph(model=model).invoke(
        {"messages": [HumanMessage(content="Hello")]}
    )

    assert result["final_response"] == "Mocked specialist response"
    assert result["agent_used"] == "general_chat"
    assert result["agents_used"] == ["general_chat"]
    assert len(model.invoked) == 1
    assert len(model.structured_model.invoked) == 1


def test_graph_uses_configured_model_for_supervisor_and_agent(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr("app.graph.get_chat_model", lambda: model)

    result = build_graph().invoke({"messages": [HumanMessage(content="Hello")]})

    assert result["final_response"] == "Mocked specialist response"
    assert result["agent_used"] == "general_chat"
    assert result["agents_used"] == ["general_chat"]
    assert len(model.invoked) == 1


@pytest.mark.parametrize(
    ("message", "agent", "specialist_response"),
    [
        ("Hi, how are you?", "general_chat", "I'm doing well. How can I help?"),
        (
            "Explain RAG and compare it with fine-tuning",
            "research",
            "RAG retrieves context; fine-tuning changes model behavior.",
        ),
        (
            "Calculate the average of 100, 150, and 200",
            "data_analysis",
            "The average is 150.",
        ),
        (
            "Create a five-step plan for an AI chatbot project",
            "task_planning",
            "1. Define goals. 2. Design. 3. Build. 4. Test. 5. Launch.",
        ),
        ("Help me", None, "What would you like help with?"),
    ],
)
def test_supervisor_routes_requests_and_returns_selected_agent(
    message, agent, specialist_response
):
    selected_agent = agent or "general_chat"
    model = FakeModel(
        agents=(agent,) if agent is not None else None,
        response=specialist_response,
    )

    result = build_graph(model=model).invoke(
        {"messages": [HumanMessage(content=message)]}
    )

    assert result["agent_used"] == selected_agent
    assert result["agents_used"] == [selected_agent]
    assert result["final_response"] == specialist_response
    assert len(model.structured_model.invoked) == 1
    assert len(model.invoked) == 1
    assert model.structured_model.invoked[0][-1].content == message


def test_sequential_routing_records_primary_agent_and_execution_order():
    model = FakeModel(
        agents=("research", "task_planning"),
        response="Combined specialist response",
    )

    result = build_graph(model=model).invoke(
        {"messages": [HumanMessage(content="Research and plan an AI rollout")]}
    )

    assert result["agent_used"] == "research"
    assert result["agents_used"] == ["research", "task_planning"]
    assert result["final_response"] == "Combined specialist response"
    assert len(model.structured_model.invoked) == 1
    assert len(model.invoked) == 3


def test_invalid_route_is_rejected():
    with pytest.raises(ValidationError):
        RoutingDecision.model_validate({"agents": ["unknown_agent"]})

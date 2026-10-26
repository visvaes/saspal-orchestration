from fastapi.testclient import TestClient
from httpx import Request, Response
from openai import AuthenticationError, BadRequestError, NotFoundError

from app.main import app


def test_config_health_reports_presence_without_exposing_secrets(monkeypatch):
    monkeypatch.setattr(
        "app.main.get_llm_config_status",
        lambda: {
            "env_file_found": True,
            "api_key_configured": True,
            "model_configured": True,
        },
    )

    response = TestClient(app).get("/api/health/config")

    assert response.status_code == 200
    assert response.json() == {
        "env_file_found": True,
        "api_key_configured": True,
        "model_configured": True,
    }


class FakeGraph:
    def invoke(self, payload):
        assert payload["messages"][-1].content == "Hello from the client"
        return {
            "final_response": "Hi there!",
            "agent_used": "general_chat",
            "agents_used": ["general_chat"],
        }


def test_chat_endpoint(monkeypatch):
    monkeypatch.setattr("app.main.build_graph", lambda: FakeGraph())

    client = TestClient(app)
    response = client.post(
        "/api/chat",
        json={
            "message": "Hello from the client",
            "history": [{"role": "assistant", "content": "Hi again"}],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "response": "Hi there!",
        "agent_used": "general_chat",
        "agents_used": ["general_chat"],
    }


def test_prefers_non_blank_message_over_prompt(monkeypatch):
    monkeypatch.setattr("app.main.build_graph", lambda: FakeGraph())

    response = TestClient(app).post(
        "/api/chat",
        json={
            "message": "Hello from the client",
            "prompt": "Should not be used",
            "history": [],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "response": "Hi there!",
        "agent_used": "general_chat",
        "agents_used": ["general_chat"],
    }


def test_prompt_is_used_when_message_is_blank(monkeypatch):
    monkeypatch.setattr("app.main.build_graph", lambda: FakeGraph())

    response = TestClient(app).post(
        "/api/chat",
        json={"message": "   ", "prompt": "Hello from the client", "history": []},
    )

    assert response.status_code == 200
    assert response.json() == {
        "response": "Hi there!",
        "agent_used": "general_chat",
        "agents_used": ["general_chat"],
    }


def test_system_history_entries_are_preserved(monkeypatch):
    monkeypatch.setattr("app.main.build_graph", lambda: FakeGraph())

    response = TestClient(app).post(
        "/api/chat",
        json={
            "message": "Hello from the client",
            "history": [{"role": "system", "content": "You are helpful."}],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "response": "Hi there!",
        "agent_used": "general_chat",
        "agents_used": ["general_chat"],
    }


def test_invalid_model_error_is_safe_and_actionable(monkeypatch):
    class FailingGraph:
        def invoke(self, payload):
            raise NotFoundError(
                "secret-free SDK error",
                response=Response(
                    404,
                    request=Request("POST", "https://api.openai.com/v1/chat/completions"),
                ),
                body={"error": "model_not_found"},
            )

    monkeypatch.setattr("app.main.build_graph", lambda: FailingGraph())

    response = TestClient(app).post("/api/chat", json={"message": "Hello"})

    assert response.status_code == 503
    assert response.json() == {
        "detail": "The configured OpenAI model was not found. Verify OPENAI_MODEL."
    }


def test_missing_api_key_error_is_safe_and_actionable(monkeypatch):
    class FailingGraph:
        def invoke(self, payload):
            raise RuntimeError(
                "OPENAI_API_KEY is not configured. Set it in the project .env file."
            )

    monkeypatch.setattr("app.main.build_graph", lambda: FailingGraph())

    response = TestClient(app).post("/api/chat", json={"message": "Hello"})

    assert response.status_code == 503
    assert response.json() == {
        "detail": "OPENAI_API_KEY is not configured. Set it in the project .env file."
    }


def test_authentication_error_does_not_expose_provider_details(monkeypatch):
    class FailingGraph:
        def invoke(self, payload):
            raise AuthenticationError(
                "provider detail containing test-secret",
                response=Response(
                    401,
                    request=Request("POST", "https://api.openai.com/v1/chat/completions"),
                ),
                body={"error": "invalid_api_key"},
            )

    monkeypatch.setattr("app.main.build_graph", lambda: FailingGraph())

    response = TestClient(app).post("/api/chat", json={"message": "Hello"})

    assert response.status_code == 503
    assert "test-secret" not in response.text
    assert response.json() == {
        "detail": "OpenAI rejected the configured API key. Verify OPENAI_API_KEY."
    }


def test_bad_model_request_error_has_safe_guidance(monkeypatch):
    class FailingGraph:
        def invoke(self, payload):
            raise BadRequestError(
                "provider detail",
                response=Response(
                    400,
                    request=Request("POST", "https://api.openai.com/v1/chat/completions"),
                ),
                body={"error": "unsupported_model"},
            )

    monkeypatch.setattr("app.main.build_graph", lambda: FailingGraph())

    response = TestClient(app).post("/api/chat", json={"message": "Hello"})

    assert response.status_code == 503
    assert response.json() == {
        "detail": (
            "OpenAI rejected the request. Verify OPENAI_MODEL is supported "
            "by your account and the request is valid."
        )
    }

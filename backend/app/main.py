
from __future__ import annotations

from os import environ
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from openai import AuthenticationError, BadRequestError, NotFoundError
from pydantic import BaseModel, Field

from app.graph import build_graph
from app.llm import get_llm_config_status
from app.state import AgentName


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"] = "user"
    content: str = ""


class ChatRequest(BaseModel):
    message: str | None = None
    prompt: str | None = None
    history: list[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    response: str
    agent_used: AgentName | None = None
    agents_used: list[str] = Field(default_factory=list)


def _allowed_origins() -> list[str]:
    origins = [
        "https://saspal-orchestrationfrontend-git-main-visvaes1.vercel.app",
        "https://saspal-orchestrationfrontend.vercel.app",
    ]

    configured = environ.get("FRONTEND_ORIGINS", "").strip()

    if configured:
        extra_origins = [
            origin.strip().rstrip("/")
            for origin in configured.split(",")
            if origin.strip()
        ]

        if "*" in extra_origins:
            raise ValueError(
                "FRONTEND_ORIGINS must contain explicit origins, not '*'."
            )

        origins.extend(origin for origin in extra_origins if origin not in origins)
    else:
        origins.extend(
            [
                "http://127.0.0.1:5173",
                "http://localhost:5173",
            ]
        )

    return origins


app = FastAPI(title="Agent Orchestration Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/health/config")
def config_health() -> dict[str, bool]:
    return get_llm_config_status()


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if request.message is not None and request.message.strip():
        user_message = request.message
    elif request.prompt is not None and request.prompt.strip():
        user_message = request.prompt
    else:
        raise HTTPException(
            status_code=400,
            detail="A message is required.",
        )

    messages: list[HumanMessage | AIMessage | SystemMessage] = []

    for item in request.history:
        if not item.content:
            continue

        if item.role == "user":
            messages.append(HumanMessage(content=item.content))
        elif item.role == "assistant":
            messages.append(AIMessage(content=item.content))
        elif item.role == "system":
            messages.append(SystemMessage(content=item.content))

    messages.append(HumanMessage(content=user_message.strip()))

    try:
        result = build_graph().invoke({"messages": messages})

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except AuthenticationError as exc:
        raise HTTPException(
            status_code=503,
            detail="OpenAI rejected the configured API key. Verify OPENAI_API_KEY.",
        ) from exc

    except NotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="The configured OpenAI model was not found. Verify OPENAI_MODEL.",
        ) from exc

    except BadRequestError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "OpenAI rejected the request. Verify OPENAI_MODEL is supported "
                "by your account and the request is valid."
            ),
        ) from exc

    agents_used = result.get("agents_used", [])

    return ChatResponse(
        response=result.get("final_response", ""),
        agent_used=(
            result.get("agent_used")
            or (agents_used[0] if agents_used else None)
        ),
        agents_used=agents_used,
    )

from functools import lru_cache
from os import environ
from pathlib import Path

from dotenv import dotenv_values
from langchain_openai import ChatOpenAI


API_KEY_PLACEHOLDERS = {
    "your_openai_api_key_here",
    "your_openai_api_key",
    "sk-your-api-key-here",
}
MODEL_PLACEHOLDERS = {
    "your_supported_model_name",
    "your_openai_model",
}


def _read_project_config() -> tuple[Path, str, str]:
    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env"
    file_values = dotenv_values(env_path)
    api_key = environ.get("OPENAI_API_KEY", "").strip() or (
        file_values.get("OPENAI_API_KEY") or ""
    ).strip()
    model_name = environ.get("OPENAI_MODEL", "").strip() or (
        file_values.get("OPENAI_MODEL") or ""
    ).strip()
    return env_path, api_key, model_name


def get_llm_config_status() -> dict[str, bool]:
    env_path, api_key, model_name = _read_project_config()
    return {
        "env_file_found": env_path.is_file(),
        "api_key_configured": bool(api_key and api_key not in API_KEY_PLACEHOLDERS),
        "model_configured": bool(model_name and model_name not in MODEL_PLACEHOLDERS),
    }


@lru_cache(maxsize=1)
def get_chat_model() -> ChatOpenAI:
    _, api_key, model_name = _read_project_config()

    if not api_key or api_key in API_KEY_PLACEHOLDERS:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. Set it in the project .env file."
        )
    if not model_name or model_name in MODEL_PLACEHOLDERS:
        raise RuntimeError(
            "OPENAI_MODEL is not configured. Set it to a model available to your API account."
        )

    try:
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            timeout=30,
            max_retries=1,
        )
    except ValueError as exc:
        raise RuntimeError(
            "OPENAI_MODEL is invalid. Set it to a model available to your API account."
        ) from exc

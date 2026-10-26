from pathlib import Path
from unittest.mock import Mock

import pytest

from app import llm


@pytest.fixture(autouse=True)
def clear_model_cache():
    llm.get_chat_model.cache_clear()
    yield
    llm.get_chat_model.cache_clear()


def test_loads_project_env_when_started_from_backend_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    project_root = tmp_path
    backend_dir = project_root / "backend"
    app_dir = backend_dir / "app"
    app_dir.mkdir(parents=True)
    (project_root / ".env").write_text(
        "OPENAI_API_KEY=test-api-key\nOPENAI_MODEL=test-model\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(llm, "__file__", str(app_dir / "llm.py"))
    monkeypatch.chdir(backend_dir)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    mock_client = Mock()
    chat_openai = Mock(return_value=mock_client)
    monkeypatch.setattr(llm, "ChatOpenAI", chat_openai)

    result = llm.get_chat_model()

    assert result is mock_client
    chat_openai.assert_called_once_with(
        model="test-model",
        api_key="test-api-key",
        timeout=30,
        max_retries=1,
    )


def test_blank_environment_settings_fall_back_to_project_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    app_dir = tmp_path / "backend" / "app"
    app_dir.mkdir(parents=True)
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=test-api-key\nOPENAI_MODEL=test-model\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(llm, "__file__", str(app_dir / "llm.py"))
    monkeypatch.setenv("OPENAI_API_KEY", " ")
    monkeypatch.setenv("OPENAI_MODEL", "")
    chat_openai = Mock()
    monkeypatch.setattr(llm, "ChatOpenAI", chat_openai)

    llm.get_chat_model()

    chat_openai.assert_called_once()


def test_config_status_reports_only_non_secret_presence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    app_dir = tmp_path / "backend" / "app"
    app_dir.mkdir(parents=True)
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=test-api-key\nOPENAI_MODEL=test-model\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(llm, "__file__", str(app_dir / "llm.py"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    assert llm.get_llm_config_status() == {
        "env_file_found": True,
        "api_key_configured": True,
        "model_configured": True,
    }


def test_config_status_reports_missing_settings_without_exposing_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    app_dir = tmp_path / "backend" / "app"
    app_dir.mkdir(parents=True)
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.setattr(llm, "__file__", str(app_dir / "llm.py"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    assert llm.get_llm_config_status() == {
        "env_file_found": True,
        "api_key_configured": False,
        "model_configured": False,
    }


@pytest.mark.parametrize(
    ("env_contents", "missing_setting"),
    [
        ("OPENAI_MODEL=test-model\n", "OPENAI_API_KEY"),
        ("OPENAI_API_KEY=test-api-key\n", "OPENAI_MODEL"),
        (
            "OPENAI_API_KEY=sk-your-api-key-here\n"
            "OPENAI_MODEL=your_supported_model_name\n",
            "OPENAI_API_KEY",
        ),
    ],
)
def test_rejects_missing_or_placeholder_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    env_contents: str,
    missing_setting: str,
):
    app_dir = tmp_path / "backend" / "app"
    app_dir.mkdir(parents=True)
    (tmp_path / ".env").write_text(env_contents, encoding="utf-8")
    monkeypatch.setattr(llm, "__file__", str(app_dir / "llm.py"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    chat_openai = Mock()
    monkeypatch.setattr(llm, "ChatOpenAI", chat_openai)

    with pytest.raises(RuntimeError, match=missing_setting):
        llm.get_chat_model()

    chat_openai.assert_not_called()

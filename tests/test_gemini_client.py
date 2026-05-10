"""Tests for gemini_client.py."""
import sys
import types
from unittest.mock import MagicMock, patch


def _build_fake_modules():
    """Insert minimal stubs so gemini_client can be imported without real deps."""
    # Stub out config
    config_mod = types.ModuleType("config")
    config_mod.GOOGLE_API_KEY = "fake-api-key"
    sys.modules.setdefault("config", config_mod)

    # Stub out llama_index hierarchy
    for mod_name in [
        "llama_index",
        "llama_index.core",
        "llama_index.llms",
        "llama_index.llms.gemini",
    ]:
        sys.modules.setdefault(mod_name, types.ModuleType(mod_name))

    # Provide a Settings stub with a settable llm attribute
    settings_stub = MagicMock()
    sys.modules["llama_index.core"].Settings = settings_stub

    # Provide a Gemini class stub
    gemini_cls = MagicMock(name="Gemini")
    sys.modules["llama_index.llms.gemini"].Gemini = gemini_cls

    return settings_stub, gemini_cls


_SETTINGS_STUB, _GEMINI_CLS = _build_fake_modules()


def _reimport():
    """Remove cached module and re-import so patches take effect cleanly."""
    sys.modules.pop("gemini_client", None)
    import gemini_client  # noqa: PLC0415

    return gemini_client


def test_initialize_gemini_llm_returns_llm_instance():
    """initialize_gemini_llm should return the Gemini instance it creates."""
    _GEMINI_CLS.reset_mock()
    fake_llm = MagicMock(name="fake_llm")
    _GEMINI_CLS.return_value = fake_llm

    mod = _reimport()
    result = mod.initialize_gemini_llm()

    assert result is fake_llm


def test_initialize_gemini_llm_constructs_gemini_with_correct_args():
    """initialize_gemini_llm should call Gemini with expected kwargs."""
    _GEMINI_CLS.reset_mock()

    mod = _reimport()
    mod.initialize_gemini_llm()

    _GEMINI_CLS.assert_called_once()
    _, kwargs = _GEMINI_CLS.call_args
    assert kwargs.get("api_key") == "fake-api-key"
    assert kwargs.get("model") == "models/gemini-2.5-flash"
    assert kwargs.get("temperature") == 0.1


def test_initialize_gemini_llm_registers_with_settings():
    """initialize_gemini_llm should assign the llm to Settings.llm."""
    _GEMINI_CLS.reset_mock()
    fake_llm = MagicMock(name="fake_llm")
    _GEMINI_CLS.return_value = fake_llm

    mod = _reimport()
    mod.initialize_gemini_llm()

    # Settings is a MagicMock; verify assignment was attempted
    assert mod.Settings.llm == fake_llm

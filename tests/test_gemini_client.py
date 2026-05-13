import sys
import types
from unittest.mock import MagicMock, patch

import pytest

import gemini_client
from gemini_client import (
    MAX_BATCH_SIZE,
    MAX_PROMPT_LENGTH,
    _get_api_key,
    _sanitize,
    batch_generate,
    generate_response,
    generate_with_context,
    initialize_gemini_llm,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    gemini_client._llm_instance = None
    yield
    gemini_client._llm_instance = None


# ---------------------------------------------------------------------------
# _get_api_key
# ---------------------------------------------------------------------------

class TestGetApiKey:
    def test_returns_env_var(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
        assert _get_api_key() == "env-key"

    def test_falls_back_to_config(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        fake_config = types.ModuleType("config")
        fake_config.GOOGLE_API_KEY = "config-key"
        with patch.dict(sys.modules, {"config": fake_config}):
            assert _get_api_key() == "config-key"

    def test_raises_if_no_key_and_no_config(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with patch.dict(sys.modules, {"config": None}):
            with pytest.raises(ValueError, match="GOOGLE_API_KEY is not set"):
                _get_api_key()

    def test_raises_if_env_var_is_empty(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "")
        with patch.dict(sys.modules, {"config": None}):
            with pytest.raises(ValueError, match="GOOGLE_API_KEY is not set"):
                _get_api_key()

    def test_env_takes_priority_over_config(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "env-wins")
        fake_config = types.ModuleType("config")
        fake_config.GOOGLE_API_KEY = "config-key"
        with patch.dict(sys.modules, {"config": fake_config}):
            assert _get_api_key() == "env-wins"


# ---------------------------------------------------------------------------
# _sanitize
# ---------------------------------------------------------------------------

class TestSanitize:
    def test_passes_clean_text(self):
        assert _sanitize("Hello, world!") == "Hello, world!"

    def test_preserves_newlines_and_tabs(self):
        text = "line1\nline2\ttabbed"
        assert _sanitize(text) == text

    def test_empty_string(self):
        assert _sanitize("") == ""

    def test_raises_on_non_string_int(self):
        with pytest.raises(TypeError, match="Expected str, got int"):
            _sanitize(42)

    def test_raises_on_none(self):
        with pytest.raises(TypeError, match="Expected str, got NoneType"):
            _sanitize(None)

    def test_raises_if_exceeds_max_length(self):
        with pytest.raises(ValueError, match="exceeds max allowed length"):
            _sanitize("x" * (MAX_PROMPT_LENGTH + 1))

    def test_accepts_exactly_max_length(self):
        text = "x" * MAX_PROMPT_LENGTH
        assert _sanitize(text) == text

    def test_strips_rtl_override(self):
        text = "before‮after"
        result = _sanitize(text)
        assert "‮" not in result
        assert "before" in result and "after" in result

    def test_strips_ltr_embedding(self):
        text = "a‪b"
        result = _sanitize(text)
        assert "‪" not in result

    def test_strips_control_characters(self):
        result = _sanitize("hello\x01world")
        assert "\x01" not in result
        assert result == "helloworld"

    def test_strips_null_byte(self):
        result = _sanitize("hello\x00world")
        assert "\x00" not in result

    def test_unicode_letters_preserved(self):
        text = "Héllo wörld"
        assert _sanitize(text) == text


# ---------------------------------------------------------------------------
# initialize_gemini_llm
# ---------------------------------------------------------------------------

class TestInitializeGeminiLlm:
    def test_returns_gemini_instance(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        mock_llm = MagicMock()
        with patch("gemini_client.Gemini", return_value=mock_llm):
            with patch("gemini_client.Settings"):
                result = initialize_gemini_llm()
        assert result is mock_llm

    def test_passes_correct_args_to_gemini(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        with patch("gemini_client.Gemini") as mock_cls:
            with patch("gemini_client.Settings"):
                initialize_gemini_llm()
        mock_cls.assert_called_once_with(
            api_key="fake-key",
            model="models/gemini-2.5-flash",
            temperature=0.1,
        )

    def test_sets_settings_llm(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        mock_llm = MagicMock()
        with patch("gemini_client.Gemini", return_value=mock_llm):
            with patch("gemini_client.Settings") as mock_settings:
                initialize_gemini_llm()
        assert mock_settings.llm is mock_llm

    def test_singleton_called_once(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        with patch("gemini_client.Gemini") as mock_cls:
            with patch("gemini_client.Settings"):
                r1 = initialize_gemini_llm()
                r2 = initialize_gemini_llm()
        assert r1 is r2
        mock_cls.assert_called_once()

    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with patch.dict(sys.modules, {"config": None}):
            with pytest.raises(ValueError, match="GOOGLE_API_KEY is not set"):
                initialize_gemini_llm()


# ---------------------------------------------------------------------------
# generate_response
# ---------------------------------------------------------------------------

class TestGenerateResponse:
    def _mock_llm(self, text="response"):
        llm = MagicMock()
        llm.complete.return_value.text = text
        return llm

    def test_returns_response_text(self):
        llm = self._mock_llm("Hello!")
        assert generate_response("Hi", llm=llm) == "Hello!"

    def test_passes_sanitized_prompt_to_llm(self):
        llm = self._mock_llm()
        generate_response("test prompt", llm=llm)
        llm.complete.assert_called_once_with("test prompt")

    def test_raises_on_non_string_prompt(self):
        with pytest.raises(TypeError):
            generate_response(123, llm=MagicMock())

    def test_raises_on_too_long_prompt(self):
        with pytest.raises(ValueError, match="exceeds max allowed length"):
            generate_response("x" * (MAX_PROMPT_LENGTH + 1), llm=MagicMock())

    def test_auto_initializes_llm_when_none(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        mock_llm = self._mock_llm("auto")
        with patch("gemini_client.initialize_gemini_llm", return_value=mock_llm):
            result = generate_response("hello")
        assert result == "auto"

    def test_strips_direction_override_before_sending(self):
        llm = self._mock_llm()
        generate_response("clean‮injected", llm=llm)
        sent = llm.complete.call_args[0][0]
        assert "‮" not in sent


# ---------------------------------------------------------------------------
# batch_generate
# ---------------------------------------------------------------------------

class TestBatchGenerate:
    def _mock_llm(self, responses):
        llm = MagicMock()
        llm.complete.side_effect = [MagicMock(text=r) for r in responses]
        return llm

    def test_returns_list_of_responses(self):
        llm = self._mock_llm(["r1", "r2", "r3"])
        assert batch_generate(["a", "b", "c"], llm=llm) == ["r1", "r2", "r3"]

    def test_empty_list(self):
        llm = MagicMock()
        assert batch_generate([], llm=llm) == []
        llm.complete.assert_not_called()

    def test_raises_if_not_list(self):
        with pytest.raises(TypeError, match="prompts must be a list"):
            batch_generate("not a list", llm=MagicMock())

    def test_raises_if_batch_exceeds_max(self):
        with pytest.raises(ValueError, match="exceeds maximum"):
            batch_generate(["p"] * (MAX_BATCH_SIZE + 1), llm=MagicMock())

    def test_accepts_exact_max_batch_size(self):
        llm = MagicMock()
        llm.complete.return_value.text = "ok"
        result = batch_generate(["p"] * MAX_BATCH_SIZE, llm=llm)
        assert len(result) == MAX_BATCH_SIZE

    def test_raises_on_non_string_in_batch(self):
        with pytest.raises(TypeError):
            batch_generate(["valid", 42], llm=MagicMock())

    def test_raises_on_too_long_prompt_in_batch(self):
        with pytest.raises(ValueError, match="exceeds max allowed length"):
            batch_generate(["x" * (MAX_PROMPT_LENGTH + 1)], llm=MagicMock())

    def test_auto_initializes_llm_when_none(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        mock_llm = MagicMock()
        mock_llm.complete.return_value.text = "ok"
        with patch("gemini_client.initialize_gemini_llm", return_value=mock_llm):
            result = batch_generate(["hello"])
        assert result == ["ok"]


# ---------------------------------------------------------------------------
# generate_with_context
# ---------------------------------------------------------------------------

class TestGenerateWithContext:
    def _mock_llm(self, text="answer"):
        llm = MagicMock()
        llm.complete.return_value.text = text
        return llm

    def test_returns_response_text(self):
        llm = self._mock_llm("42")
        assert generate_with_context("What?", "Some context.", llm=llm) == "42"

    def test_prompt_has_xml_context_delimiters(self):
        llm = self._mock_llm()
        generate_with_context("Q", "CTX", llm=llm)
        sent = llm.complete.call_args[0][0]
        assert "<context>" in sent
        assert "</context>" in sent

    def test_prompt_contains_context_and_question(self):
        llm = self._mock_llm()
        generate_with_context("my question", "my context", llm=llm)
        sent = llm.complete.call_args[0][0]
        assert "my context" in sent
        assert "Question: my question" in sent

    def test_context_appears_before_question(self):
        llm = self._mock_llm()
        generate_with_context("Q", "CTX", llm=llm)
        sent = llm.complete.call_args[0][0]
        assert sent.index("CTX") < sent.index("Question: Q")

    def test_raises_on_non_string_prompt(self):
        with pytest.raises(TypeError):
            generate_with_context(123, "context", llm=MagicMock())

    def test_raises_on_non_string_context(self):
        with pytest.raises(TypeError):
            generate_with_context("prompt", 456, llm=MagicMock())

    def test_raises_on_too_long_prompt(self):
        with pytest.raises(ValueError, match="exceeds max allowed length"):
            generate_with_context("x" * (MAX_PROMPT_LENGTH + 1), "ctx", llm=MagicMock())

    def test_raises_on_too_long_context(self):
        with pytest.raises(ValueError, match="exceeds max allowed length"):
            generate_with_context("q", "x" * (MAX_PROMPT_LENGTH + 1), llm=MagicMock())

    def test_auto_initializes_llm_when_none(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        mock_llm = self._mock_llm("auto")
        with patch("gemini_client.initialize_gemini_llm", return_value=mock_llm):
            result = generate_with_context("q", "ctx")
        assert result == "auto"

    def test_sanitizes_injected_context(self):
        llm = self._mock_llm()
        generate_with_context("q", "ctx‮ injected", llm=llm)
        sent = llm.complete.call_args[0][0]
        assert "‮" not in sent

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from ingestion import ollama_client
from ingestion.ollama_client import GenerationError, generate_json


class DummySchema(BaseModel):
    foo: str


def make_response(content: str) -> MagicMock:
    response = MagicMock()
    response.message.content = content
    return response


def test_generate_json_succeeds_first_try(monkeypatch):
    mock_chat = MagicMock(return_value=make_response('{"foo": "bar"}'))
    monkeypatch.setattr(ollama_client._client, "chat", mock_chat)

    result = generate_json("prompt", DummySchema)

    assert result.foo == "bar"
    assert mock_chat.call_count == 1


def test_generate_json_retries_on_malformed_json_then_succeeds(monkeypatch):
    responses = [make_response("this is not json"), make_response('{"foo": "bar"}')]
    mock_chat = MagicMock(side_effect=responses)
    monkeypatch.setattr(ollama_client._client, "chat", mock_chat)

    result = generate_json("prompt", DummySchema)

    assert result.foo == "bar"
    assert mock_chat.call_count == 2
    retry_prompt = mock_chat.call_args_list[1].kwargs["messages"][0]["content"]
    assert "ONLY valid JSON" in retry_prompt


def test_generate_json_retries_on_schema_mismatch_then_succeeds(monkeypatch):
    responses = [make_response('{"wrong_field": "x"}'), make_response('{"foo": "bar"}')]
    mock_chat = MagicMock(side_effect=responses)
    monkeypatch.setattr(ollama_client._client, "chat", mock_chat)

    result = generate_json("prompt", DummySchema)

    assert result.foo == "bar"
    assert mock_chat.call_count == 2


def test_generate_json_raises_after_exhausting_retries(monkeypatch):
    mock_chat = MagicMock(return_value=make_response("still not json"))
    monkeypatch.setattr(ollama_client._client, "chat", mock_chat)

    with pytest.raises(GenerationError):
        generate_json("prompt", DummySchema)

    assert mock_chat.call_count == ollama_client.MAX_RETRIES + 1


def test_generate_json_passes_schema_as_format(monkeypatch):
    mock_chat = MagicMock(return_value=make_response('{"foo": "bar"}'))
    monkeypatch.setattr(ollama_client._client, "chat", mock_chat)

    generate_json("prompt", DummySchema)

    assert mock_chat.call_args.kwargs["format"] == DummySchema.model_json_schema()


def test_embed_returns_first_embedding(monkeypatch):
    mock_response = MagicMock()
    mock_response.embeddings = [[0.1, 0.2, 0.3]]
    mock_embed = MagicMock(return_value=mock_response)
    monkeypatch.setattr(ollama_client._client, "embed", mock_embed)

    vector = ollama_client.embed("some text")

    assert vector == [0.1, 0.2, 0.3]

from unittest.mock import MagicMock

from retrieval import search as search_module


def _mock_rpc_client(data: list[dict]) -> MagicMock:
    mock_response = MagicMock()
    mock_response.data = data
    mock_rpc = MagicMock()
    mock_rpc.execute.return_value = mock_response
    mock_client = MagicMock()
    mock_client.rpc.return_value = mock_rpc
    return mock_client


def test_search_embeds_query_with_the_ingestion_embedder(monkeypatch):
    mock_embed = MagicMock(return_value=[0.1] * 768)
    monkeypatch.setattr(search_module, "embed_text", mock_embed)
    monkeypatch.setattr(search_module, "_get_client", MagicMock(return_value=_mock_rpc_client([])))

    search_module.search("how do I export a video", "OpenShot")

    mock_embed.assert_called_once_with("how do I export a video")


def test_search_calls_rpc_with_expected_params(monkeypatch):
    monkeypatch.setattr(search_module, "embed_text", MagicMock(return_value=[0.1] * 768))
    mock_client = _mock_rpc_client([{"id": "1", "content_type": "procedural", "similarity": 0.9}])
    monkeypatch.setattr(search_module, "_get_client", MagicMock(return_value=mock_client))

    results = search_module.search("how do I export", "OpenShot", top_k=3, content_types=["procedural"])

    assert results == [{"id": "1", "content_type": "procedural", "similarity": 0.9}]
    mock_client.rpc.assert_called_once_with(
        "match_knowledge_chunks",
        {
            "query_embedding": [0.1] * 768,
            "match_app_name": "OpenShot",
            "match_content_types": ["procedural"],
            "match_count": 3,
        },
    )


def test_search_defaults_content_types_to_none(monkeypatch):
    monkeypatch.setattr(search_module, "embed_text", MagicMock(return_value=[0.0] * 768))
    mock_client = _mock_rpc_client([])
    monkeypatch.setattr(search_module, "_get_client", MagicMock(return_value=mock_client))

    search_module.search("query", "OpenShot")

    called_params = mock_client.rpc.call_args[0][1]
    assert called_params["match_content_types"] is None
    assert called_params["match_count"] == 5

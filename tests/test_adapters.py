"""Tests du client Qdrant local et du contrat HTTP des deux fournisseurs."""

import json
from uuid import uuid4

import httpx
import pytest
from qdrant_client import QdrantClient

from app.config import Settings
from app.embeddings import Embeddings
from app.errors import RagError
from app.generation import Generation
from app.store import Store


@pytest.mark.filterwarnings("ignore:Payload indexes have no effect")
def test_qdrant_persistence_reingestion_and_filters(tmp_path):
    settings = Settings(embed_dim=2)
    path = str(tmp_path / "index")
    store = Store(settings, QdrantClient(path=path))
    store.ensure()
    point = {
        "document_id": str(uuid4()), "chunk_id": str(uuid4()), "chunk_index": 0,
        "source_name": "guide.pdf", "source_type": "pdf", "page": 2, "text": "Accueil à neuf heures.",
        "metadata": {"equipe": "support", "public": True, "edition": 3},
    }
    store.upsert([point], [[1.0, 0.0]])
    store.upsert([point], [[1.0, 0.0]])
    store.close()

    # Une nouvelle instance relit le même répertoire de stockage.
    db = QdrantClient(path=path)
    store = Store(settings, db)
    try:
        store.ensure()
        filters = {"page": 2, "source_name": "guide.pdf", "metadata.public": True, "metadata.edition": 3}
        assert store.search([1.0, 0.0], 5, 0.9, filters) == [{**point, "score": 1.0}]
        assert store.search([0.0, 1.0], 5, 0.9, {}) == []
        assert store.search([1.0, 0.0], 5, None, {"metadata.equipe": "compta"}) == []

        for incompatible in ({"embed_dim": 3}, {"embed_model": "autre"}):
            with pytest.raises(RagError, match="réindexez") as error:
                Store(settings.model_copy(update=incompatible), db).ensure()
            assert error.value.status == 409
    finally:
        store.close()


def test_nomic_prefixes_batching_and_model_probe():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append((request.url.path, body))
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"details": {}})
        return httpx.Response(200, json={"embeddings": [[1.0, 0.5] for _ in body["input"]]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        embedder = Embeddings(Settings(embed_dim=2, embed_batch_size=2), client)
        embedder.check_model()
        assert len(embedder.embed(["a", "b", "c"], "document")) == 3
        assert len(embedder.embed(["q"], "query")) == 1
    assert calls[0] == ("/api/show", {"model": "nomic-embed-text"})
    assert [body["input"] for _, body in calls[1:]] == [
        ["search_document: a", "search_document: b"], ["search_document: c"], ["search_query: q"],
    ]
    assert all(path == "/api/embed" and body["truncate"] is False for path, body in calls[1:])


@pytest.mark.parametrize("embeddings", [[], [[1.0]], [[0.0, 0.0]], [[True, "a"]]])
def test_rejects_incomplete_or_invalid_vectors(embeddings):
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={"embeddings": embeddings}))
    with httpx.Client(transport=transport) as client:
        with pytest.raises(RagError) as error:
            Embeddings(Settings(embed_dim=2), client).embed(["passage"], "document")
        assert error.value.status == 502


@pytest.mark.parametrize("provider,base,key", [
    ("ollama", "http://ollama.test:11434/v1/", ""),
    ("openrouter", "https://openrouter.test/api/v1", "cle-de-test"),
])
def test_chat_completions_contract(provider, base, key):
    settings = Settings(llm_provider=provider, llm_api_key=key, llm_base_url=base, llm_model="modele-test")
    messages = [{"role": "system", "content": "Citer les sources."}, {"role": "user", "content": "Question ?"}]

    def handler(request):
        assert str(request.url) == base.rstrip("/") + "/chat/completions"
        assert request.headers.get("Authorization") == ("Bearer cle-de-test" if key else None)
        assert json.loads(request.content) == {"model": "modele-test", "messages": messages,
                                               "stream": False, "temperature": 0}
        return httpx.Response(200, json={"choices": [{"message": {"content": "Réponse [S1]."}}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert Generation(settings, client).chat(messages) == "Réponse [S1]."


def test_model_error_omits_raw_response_and_timeout_is_distinct():
    def fail(request):
        if request.url.path == "/api/embed":
            return httpx.Response(404, text="contenu privé du document")
        raise httpx.ReadTimeout("simulateur")

    with httpx.Client(transport=httpx.MockTransport(fail)) as client:
        with pytest.raises(RagError, match="404") as error:
            Embeddings(Settings(), client).embed(["contenu privé du document"], "document")
        assert "privé" not in error.value.detail
        with pytest.raises(RagError) as timeout:
            Generation(Settings(), client).chat([{"role": "user", "content": "Question"}])
        assert timeout.value.status == 504


def test_unreachable_qdrant_reported_through_real_http_client():
    def disconnected(request):
        raise httpx.ConnectError("connexion simulée", request=request)

    client = QdrantClient(url="http://qdrant.test", transport=httpx.MockTransport(disconnected),
                          check_compatibility=False)
    store = Store(Settings(), client)
    try:
        with pytest.raises(RagError, match="Qdrant est inaccessible") as failure:
            store.ensure()
        assert failure.value.status == 503
    finally:
        store.close()

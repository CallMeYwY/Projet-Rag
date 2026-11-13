"""Test du contrat HTTP complet, sans réseau ni modèle payant.

Qdrant exécute réellement la recherche en mémoire. Seuls les modèles sont
remplacés : les vecteurs prévisibles permettent de vérifier le classement.
"""
import warnings
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.config import Settings
from app.errors import RagError
from app.main import create_app
from app.store import Store


class FakeEmbeddings:
    def check_model(self):
        pass

    def embed(self, texts, kind):
        return [[float("vélo" in t.lower()), float("casier" in t.lower()), 0.01] for t in texts]

    def close(self):
        pass


class FakeGeneration:
    def __init__(self):
        self.calls = []

    def chat(self, messages):
        self.calls.append(messages)
        return "Le vélo se trouve dans le local bleu [S1]."

    def close(self):
        pass


@contextmanager
def service(**overrides):
    cfg = Settings(embed_dim=3, **overrides)
    # Le moteur en mémoire ne crée pas d'index de payload, mais applique bien les filtres.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Payload indexes have no effect.*")
        store = Store(cfg, client=QdrantClient(":memory:"))
        gen, embed = FakeGeneration(), FakeEmbeddings()
        with TestClient(create_app(cfg, store=store, embeddings=embed, generation=gen)) as cli:
            yield cli, gen, store, embed


def test_ingest_dedup_metadata_recherche_et_generation():
    with service() as (cli, gen, store, _):
        doc = {"text": "Le vélo se trouve dans le local bleu.", "source_name": "Accueil",
               "metadata": {"site": "Nord", "public": True}}
        ing = cli.post("/ingest", json=doc)
        assert ing.status_code == 200
        indexed = ing.json()
        assert indexed["chunks_indexed"] == 1
        updated = cli.post("/ingest", json={**doc, "metadata": {"site": "Sud"}}).json()
        assert indexed["chunk_ids"] == updated["chunk_ids"]
        assert updated["document_id"] == indexed["document_id"]
        assert store.client.count(store.collection).count == 1
        file = cli.post("/ingest", files={"file": ("note.md", b"Un casier Opale.")},
                        data={"metadata": '{"site":"Est"}'})
        assert file.status_code == 200 and file.json()["source_type"] == "md"
        req = {"query": "Où est le vélo ?", "filters": {"metadata.site": "Sud"}}
        test = cli.post("/search", json=req).json()
        assert test["mode"] == "test" and test["metric"] == "cosine" and test["status"] == "ok"
        assert not gen.calls
        hit, = test["results"]
        assert (hit["text"], hit["source_name"], hit["metadata"], hit["page"]) == (
            doc["text"], "Accueil", {"site": "Sud"}, None)
        assert hit["score"] > 0.99
        ref, = test["sources"]
        assert ref["chunk_id"] == indexed["chunk_ids"][0] and ref["citation"] == "S1"
        prod = cli.post("/search", json={**req, "mode": "prod"}).json()
        assert prod["sources"] == test["sources"] and prod["answer"].endswith("[S1].")
        assert (prod["provider"], prod["model"]) == ("ollama", "gemma3:4b")
        assert gen.calls == [test["prompt"]]
        # Même nom mais contenu différent : c'est une autre version, sans écrasement.
        new = cli.post("/ingest", json={**doc, "text": "Une autre version."}).json()
        assert new["document_id"] != indexed["document_id"]


def test_aucune_generation_si_seuil_ou_filtre_exclut_tout():
    with service() as (cli, gen, _, _):
        cli.post("/ingest", json={"text": "Un vélo."}).raise_for_status()
        for extra in [{"score_threshold": 0.9}, {"filters": {"source_name": "absente"}}]:
            for mode in ("prod", "test"):
                res = cli.post("/search", json={"query": "casier", "mode": mode, **extra}).json()
                assert res["status"] == "no_context" and res["sources"] == []
                assert res.get("answer") is None
        assert not gen.calls


def test_budget_ne_cite_que_ce_qui_est_envoye():
    with service(max_context_chars=110) as (cli, gen, _, _):
        for txt, name in [("vélo " * 15, "long"), ("casier court", "bref")]:
            cli.post("/ingest", json={"text": txt, "source_name": name}).raise_for_status()
        query = {"query": "vélo"}
        debug = cli.post("/search", json=query).json()
        prod = cli.post("/search", json={**query, "mode": "prod"}).json()
        assert [h["source_name"] for h in debug["results"]] == ["long", "bref"]
        assert [s["source_name"] for s in debug["sources"]] == ["bref"]
        assert debug["sources"] == prod["sources"]
        assert gen.calls == [debug["prompt"]]
        assert "casier court" in gen.calls[0][1]["content"] and "long" not in gen.calls[0][1]["content"]
        only_long = cli.post("/search", json={**query, "mode": "prod", "filters": {"source_name": "long"}})
        assert only_long.json()["status"] == "context_limit" and len(gen.calls) == 1


def test_service_degrade_et_erreur_fournisseur(monkeypatch):
    def unavailable(*args):
        raise RagError(503, "Service inaccessible.")
    with service() as (cli, gen, store, embed):
        assert cli.get("/health").json()["status"] == "ready"
        cli.post("/ingest", json={"text": "un vélo"}).raise_for_status()
        monkeypatch.setattr(gen, "chat", unavailable)
        assert cli.post("/search", json={"query": "vélo", "mode": "prod"}).status_code == 503
        assert cli.post("/search", json={"query": "vélo"}).status_code == 200
        monkeypatch.setattr(store, "ensure", unavailable)
        assert cli.post("/ingest", json={"text": "document"}).status_code == 503
        assert cli.post("/search", json={"query": "vélo"}).status_code == 503
        health = cli.get("/health")
        assert health.status_code == 503 and health.json()["qdrant"] == "unavailable"
        monkeypatch.setattr(embed, "check_model", unavailable)
        assert cli.get("/health").json()["embeddings"] == "unavailable"


@pytest.mark.parametrize("req", [
    {}, {"query": " "}, {"query": "a", "top_k": 31}, {"query": "a", "score_threshold": 1.1},
    {"query": "a", "filters": {"text": "b"}}, {"query": "a", "filters": {"page": True}},
])
def test_recherche_invalide(req):
    with service() as (cli, *_):
        assert cli.post("/search", json=req).status_code == 422


def test_entrees_invalides_et_limites():
    with service(max_file_bytes=64) as (cli, *_):
        bad = [
            ({"json": {"text": "   "}}, 422),
            ({"content": "{}", "headers": {"content-type": "text/plain"}}, 415),
            ({"content": "{", "headers": {"content-type": "application/json"}}, 422),
            ({"files": {"file": ("notes.md", b"ok")}, "data": {"metadata": "[]"}}, 422),
            ({"files": {"file": ("notes.md", b"ok")}, "data": {"metadata": "?"}}, 422),
            ({"files": {"file": ("video.mp4", b"data")}}, 415),
            ({"files": {"file": ("photo.png", b"pas une image")}}, 422),
            ({"files": {"file": ("grand.txt", b" " * 65)}}, 413),
            ({"json": {"text": "é" * 33}}, 413),
        ]
        for payload, expected in bad:
            assert cli.post("/ingest", **payload).status_code == expected


def test_reingestion_en_echec_conserve_les_donnees(monkeypatch):
    def timeout(*args):
        raise RagError(504, "Délai embeddings dépassé.")
    with service() as (cli, _, store, embed):
        doc = {"text": "Un vélo bleu", "source_name": "Accueil", "metadata": {"version": 1}}
        old = cli.post("/ingest", json=doc).json()
        monkeypatch.setattr(embed, "embed", timeout)
        assert cli.post("/ingest", json={**doc, "metadata": {"version": 2}}).status_code == 504
        point, = store.client.retrieve(store.collection, old["chunk_ids"])
        assert point.payload["metadata"] == {"version": 1}

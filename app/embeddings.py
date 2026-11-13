"""Vectorisation des passages et des questions avec l'API /api/embed d'Ollama."""

import math
from typing import TYPE_CHECKING, Literal

import httpx

from .errors import RagError
from .model_http import post_json

if TYPE_CHECKING:
    from .config import Settings


class Embeddings:
    """Convertit le texte en une liste de nombres appelée vecteur ou embedding.

    À l'ingestion, on conserve un vecteur par passage. À la recherche, on calcule
    celui de la question avec le même modèle pour pouvoir les comparer dans Qdrant.
    """

    def __init__(self, settings: "Settings", client: httpx.Client | None = None):
        self.settings = settings
        self.client = client if client is not None else httpx.Client()

    def _call(self, path: str, body: dict) -> dict:
        return post_json(self.client, self.settings.ollama_url.rstrip("/") + path, body,
                         service="Ollama (embeddings)", timeout=self.settings.timeout)

    def check_model(self) -> None:
        """Vérifie la présence du modèle, sans calculer de vecteur."""
        self._call("/api/show", {"model": self.settings.embed_model})

    def embed(self, texts: list[str], kind: Literal["document", "query"]) -> list[list[float]]:
        """Renvoie un vecteur par texte, dans l'ordre d'entrée."""
        # Nomic utilise le préfixe pour distinguer la question du passage susceptible
        # d'y répondre. Ce préfixe sert uniquement au calcul du vecteur.
        model = self.settings.embed_model
        prefixes = {"document": "search_document: ", "query": "search_query: "}
        prefix = prefixes[kind] if model.rsplit("/", 1)[-1].split(":")[0] == "nomic-embed-text" else ""
        batch_size = self.settings.embed_batch_size
        vectors = []
        for start in range(0, len(texts), batch_size):
            inputs = [prefix + text for text in texts[start:start + batch_size]]
            # Refuser la troncature permet de détecter un passage trop long. Le vecteur
            # doit représenter tout le texte que le lecteur retrouvera ensuite dans la citation.
            data = self._call("/api/embed", {"model": model, "input": inputs, "truncate": False})
            batch = data.get("embeddings")
            if not isinstance(batch, list) or len(batch) != len(inputs):
                raise RagError(502, "Ollama n’a pas renvoyé un vecteur par texte.")
            for vector in batch:
                # Chaque vecteur doit tenir dans la collection et avoir une direction
                # définie pour que Qdrant puisse calculer une similarité cosinus.
                if not isinstance(vector, list) or len(vector) != self.settings.embed_dim:
                    raise RagError(502, "La dimension renvoyée par Ollama ne correspond pas à EMBED_DIM.")
                if not all(type(value) in (int, float) and math.isfinite(value) for value in vector):
                    raise RagError(502, "Ollama a renvoyé un vecteur non numérique ou non fini.")
                if not any(vector):
                    raise RagError(502, "Ollama a renvoyé un vecteur nul, inutilisable pour la similarité cosinus.")
            vectors.extend(batch)
        return vectors

    def close(self) -> None:
        self.client.close()

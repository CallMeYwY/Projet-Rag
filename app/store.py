"""Accès à Qdrant : une collection correspond à un modèle d'embeddings."""

from contextlib import contextmanager
from math import ceil
from threading import Lock
from typing import TYPE_CHECKING

import httpx
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from .errors import RagError

if TYPE_CHECKING:
    from .config import Settings


@contextmanager
def _qdrant_errors():
    """Traduit les erreurs du client sans renvoyer le corps des requêtes."""
    try:
        yield
    except ResponseHandlingException as exc:
        if isinstance(exc.source, httpx.TimeoutException):
            raise RagError(504, "Qdrant a dépassé le délai maximal de réponse.") from exc
        raise RagError(503, "Qdrant est inaccessible. Vérifiez QDRANT_URL et son démarrage.") from exc
    except UnexpectedResponse as exc:
        if exc.status_code in (401, 403):
            raise RagError(503, "Qdrant a refusé l’accès. Vérifiez QDRANT_API_KEY.") from exc
        if exc.status_code == 413:
            raise RagError(413, "Le document vectorisé dépasse la taille de requête admise par Qdrant.") from exc
        raise RagError(503, f"Échec de la requête à Qdrant (HTTP {exc.status_code}).") from exc


class Store:
    """Relie chaque vecteur au passage qu'on voudra lire ou citer.

    Dans Qdrant, un « point » réunit un identifiant, un vecteur et un ensemble
    de données appelé payload. Ici, ce dernier contient le texte, sa source et
    sa page. Il permet de remonter du résultat numérique au document d'origine.
    """

    def __init__(self, settings: "Settings", client: QdrantClient | None = None):
        self.settings = settings
        self.collection = settings.collection
        self.client = client if client is not None else QdrantClient(
            url=settings.qdrant_url, api_key=settings.qdrant_api_key or None,
            timeout=ceil(settings.timeout), check_compatibility=False,
        )
        self._creation_lock = Lock()

    def ensure(self) -> None:
        """Crée la collection au premier appel et contrôle la compatibilité à chaque vérification."""
        with _qdrant_errors():
            if not self.client.collection_exists(self.collection):
                with self._creation_lock:
                    # Une autre requête a pu créer la collection pendant l'attente du verrou.
                    if not self.client.collection_exists(self.collection):
                        self._create()
            config = self.client.get_collection(self.collection).config
        params = config.params.vectors
        if (not isinstance(params, models.VectorParams)
                or params.size != self.settings.embed_dim or params.distance != models.Distance.COSINE):
            raise RagError(409, "Collection Qdrant incompatible avec EMBED_DIM ou la distance cosinus. "
                                "Choisissez une nouvelle COLLECTION et réindexez vos documents.")
        # Deux modèles de même dimension produisent des espaces vectoriels distincts.
        # Le ZIP ne gardait pas le modèle : on ne peut pas le deviner rétroactivement.
        if (config.metadata or {}).get("embedding_model") != self.settings.embed_model:
            raise RagError(409, "Le modèle d’embeddings de la collection est inconnu ou différent de EMBED_MODEL. "
                                "Choisissez une nouvelle COLLECTION et réindexez vos documents.")

    def _create(self) -> None:
        try:
            self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=self.settings.embed_dim, distance=models.Distance.COSINE),
                metadata={"embedding_model": self.settings.embed_model},
            )
        except UnexpectedResponse as exc:
            # Deux processus peuvent créer la collection au même instant.
            # Dans ce cas, ensure() vérifiera celle qui a gagné la course.
            if exc.status_code != 409:
                raise
        # Ces index permettent à Qdrant de restreindre rapidement une recherche
        # à un fichier, à un type de document ou à une page précise.
        for field in ("document_id", "source_name", "source_type"):
            self.client.create_payload_index(self.collection, field, models.PayloadSchemaType.KEYWORD)
        self.client.create_payload_index(self.collection, "page", models.PayloadSchemaType.INTEGER)

    def upsert(self, points: list[dict], vectors: list[list[float]]) -> None:
        """Associe chaque passage au vecteur de même rang et attend la fin de l'écriture."""
        if len(points) != len(vectors):
            raise RagError(502, "Le nombre d’embeddings ne correspond pas au nombre de passages.")
        # Les UUID de passage sont stables : répéter cette opération remplace les
        # points. Une erreur de vectorisation ne supprime jamais une version indexée.
        items = [
            models.PointStruct(id=point["chunk_id"], vector=vector, payload=point)
            for point, vector in zip(points, vectors)
        ]
        with _qdrant_errors():
            self.client.upsert(self.collection, points=items, wait=True)

    def search(self, vector: list[float], k: int, threshold: float | None, filters: dict) -> list[dict]:
        """Retrouve au plus k passages selon leur similarité cosinus avec la question.

        Un score plus élevé indique que les vecteurs ont des directions plus proches.
        Le seuil facultatif permet d'écarter les passages les moins proches.
        """
        # Une clé « metadata.projet » interroge le champ projet de l'objet metadata.
        # MatchValue conserve le type de la valeur, y compris les booléens.
        conditions = [
            models.FieldCondition(key=field, match=models.MatchValue(value=value))
            for field, value in filters.items()
        ]
        with _qdrant_errors():
            result = self.client.query_points(
                self.collection, query=vector, limit=k,
                query_filter=models.Filter(must=conditions) if conditions else None,
                score_threshold=threshold, with_payload=True, with_vectors=False,
            )
        # Le constructeur du prompt a besoin du texte et de sa provenance.
        # Les vecteurs, qui ont servi au classement, peuvent rester dans Qdrant.
        return [{**(point.payload or {}), "score": point.score} for point in result.points]

    def close(self) -> None:
        self.client.close()

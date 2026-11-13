"""Orchestration : chaque étape peut être testée avec des adaptateurs injectés."""
import hashlib
import json
import logging
import uuid

from .chunking import chunk_text
from .errors import RagError
from .extraction import extract_file, source_type
from .prompt import build_prompt
from .schemas import Hit, IngestResult, ProdResult, SearchInput, SourceInput, TestResult

log = logging.getLogger("uvicorn.error")


class RagService:
    """Point d'entrée pour suivre un document depuis son ingestion jusqu'à sa citation.

    Le stockage conserve les passages. Le modèle d'embeddings les situe dans un
    espace numérique ; le modèle de génération n'intervient qu'à la fin d'une recherche.
    """
    def __init__(self, settings, store, embeddings, generation):
        self.settings = settings
        self.store = store
        self.embeddings = embeddings
        self.generation = generation

    def ingest_file(self, name: str, data: bytes, attrs: SourceInput):
        """Ramène chaque format à une suite de couples (page, texte) avant de l'indexer."""
        kind = source_type(name)
        pages = extract_file(name, data, self.settings)
        return self.index(pages, attrs.source_name or name, kind, attrs.metadata)

    def index(self, pages, name: str, kind: str, metadata: dict):
        """Rend les passages recherchables et retourne leurs identifiants permanents."""
        cfg = self.settings
        if sum(len(text) for _, text in pages) > cfg.max_extracted_chars:
            raise RagError(413, "Le texte dépasse MAX_EXTRACTED_CHARS.")
        # Les pages restent séparées pendant le découpage : une citation PDF désigne
        # toujours la page qui contient l'intégralité du passage.
        chunks = []
        for page, text in pages:
            for passage in chunk_text(text, cfg.chunk_size, cfg.chunk_overlap):
                chunks.append((page, passage))
                if len(chunks) > cfg.max_chunks:
                    raise RagError(413, "Trop de passages : réduire le document ou augmenter CHUNK_SIZE.")
        if not chunks:
            raise RagError(422, "Aucun texte exploitable à indexer.")

        # Le contenu et le découpage identifient une version. On exclut les métadonnées :
        # corriger la catégorie d'un PDF remplace ses étiquettes sans dupliquer ses pages.
        identity = json.dumps([name, kind, cfg.chunk_size, cfg.chunk_overlap, chunks],
                              ensure_ascii=False, separators=(",", ":")).encode()
        doc_id = hashlib.sha256(identity).hexdigest()
        base = {"document_id": doc_id, "source_name": name, "source_type": kind, "metadata": metadata}
        # Deux passages identiques peuvent se trouver sur des pages différentes. Le rang i,
        # associé à l'empreinte du document, permet de citer précisément l'un des deux.
        points = [{**base, "chunk_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}#{i}")),
                   "chunk_index": i, "page": page, "text": passage}
                  for i, (page, passage) in enumerate(chunks)]
        self.store.ensure()
        vectors = self.embeddings.embed([p["text"] for p in points], "document")
        # On a tous les vecteurs avant la première écriture ; aucune suppression préalable.
        self.store.upsert(points, vectors)
        log.info("Indexation terminée : document=%s type=%s passages=%d", doc_id[:12], kind, len(points))
        return IngestResult(**base, chunks_indexed=len(points), chunk_ids=[p["chunk_id"] for p in points])

    def search(self, req: SearchInput):
        """Retrouve d'abord les extraits ; le mode choisi décide ensuite si le modèle répond."""
        cfg = self.settings
        if req.top_k > cfg.max_top_k:
            raise RagError(422, f"top_k doit être compris entre 1 et {cfg.max_top_k}.")
        threshold = cfg.score_threshold if req.score_threshold is None else req.score_threshold
        self.store.ensure()
        # Même modèle qu'à l'ingestion : la question et les passages sont comparables
        # même si la question ne reprend pas mot pour mot le texte du document.
        vector = self.embeddings.embed([req.query], "query")[0]
        hits = [Hit(**r) for r in self.store.search(vector, req.top_k, threshold, req.filters)]
        # hits = ce que Qdrant a retrouvé ; sources = ce qui tient dans le prompt.
        # Seul le second ensemble peut être présenté comme contexte envoyé au modèle.
        messages, sources = build_prompt(req.query, hits, cfg.max_context_chars)
        status, detail = "ok", None
        if not hits:
            status, detail = "no_context", "Aucun passage ne satisfait les critères de recherche."
        elif not sources:
            status, detail = "context_limit", "Aucun passage entier ne tient dans le budget de contexte."
        log.info("Recherche : mode=%s résultats=%d passages envoyables=%d statut=%s",
                 req.mode, len(hits), len(sources), status)
        common = dict(status=status, query=req.query, sources=sources, detail=detail)
        if req.mode == "test":
            return TestResult(**common, results=hits, prompt=messages)
        if not sources:
            return ProdResult(**common)
        answer = self.generation.chat(messages)
        return ProdResult(**common, answer=answer, provider=cfg.llm_provider, model=cfg.llm_model)

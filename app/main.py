"""Routes HTTP ; les opérations bloquantes (OCR, clients HTTP) passent dans un thread."""
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from python_multipart.exceptions import MultipartParseError

from .config import Settings
from .embeddings import Embeddings
from .errors import RagError
from .generation import Generation
from .schemas import Health, IngestResult, ProdResult, SearchInput, SourceInput, TestResult, TextInput
from .service import RagService
from .store import Store

log = logging.getLogger("uvicorn.error")
INGEST_BODY = {"required": True, "content": {
    "application/json": {"schema": TextInput.model_json_schema(),
                         "example": {"text": "Le local vélo ouvre à 8 h.", "source_name": "accueil",
                                     "metadata": {"site": "Lyon"}}},
    "multipart/form-data": {"schema": {"type": "object", "required": ["file"], "properties": {
        "file": {"type": "string", "format": "binary"},
        "metadata": {"type": "string", "description": 'Objet JSON, par exemple {"site":"Lyon"}'},
        "source_name": {"type": "string", "description": "Par défaut : nom du fichier."}}}}}}


def invalid_fields(errors):
    # Les erreurs Pydantic incluent par défaut l'objet d'entrée ; un document ne doit
    # pas être recopié dans les erreurs de validation ou les journaux.
    return "; ".join(f"{'.'.join(map(str, e['loc'])) or 'entrée'} : {e['msg']}" for e in errors[:4])


async def bounded_body(request: Request, limit: int) -> bytearray:
    body = bytearray()
    async for part in request.stream():
        if len(body) + len(part) > limit:
            raise RagError(413, "Requête trop volumineuse pour /ingest.")
        body.extend(part)
    return body


async def read_ingest(request: Request, cfg: Settings):
    """Sur la même route, accepter du texte JSON ou décoder le formulaire d'un fichier."""
    media = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media not in ("application/json", "multipart/form-data"):
        raise RagError(415, "Utiliser application/json ou multipart/form-data.")
    # Bornage AVANT le décodage JSON/multipart, y compris sans Content-Length.
    # Les 64 Kio supplémentaires couvrent les champs, noms et séparateurs multipart.
    body = await bounded_body(request, cfg.max_file_bytes + 65_536)
    try:
        if media == "application/json":
            doc = TextInput.model_validate_json(body)
            if len(doc.text.encode("utf-8")) > cfg.max_file_bytes:
                raise RagError(413, "Le texte dépasse MAX_FILE_BYTES.")
            return doc, None, None

        async def receive():
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        # Le flux réseau a déjà été consommé pour appliquer la limite de taille.
        # On remet les octets validés à disposition du décodeur multipart de Starlette.
        replay = Request(request.scope, receive)
        async with replay.form(max_files=1, max_fields=2, max_part_size=65_536) as form:
            keys = list(form.keys())
            if len(keys) != len(form.multi_items()) or set(keys) - {"file", "metadata", "source_name"}:
                raise RagError(422, "Champs de formulaire inconnus ou répétés.")
            file = form.get("file")
            if not isinstance(file, UploadFile) or not file.filename:
                raise RagError(422, "Le champ file doit contenir un fichier nommé.")
            # Ne jamais traiter le chemin fourni par le client comme un chemin local.
            name = file.filename.replace("\\", "/").rsplit("/", 1)[-1].strip()
            if not name or len(name) > 255:
                raise RagError(422, "Le nom du fichier est vide ou dépasse 255 caractères.")
            meta = json.loads(form["metadata"]) if "metadata" in form else {}
            attrs = SourceInput(source_name=form.get("source_name"), metadata=meta)
            data = await file.read(cfg.max_file_bytes + 1)
            if len(data) > cfg.max_file_bytes:
                raise RagError(413, "Le fichier dépasse MAX_FILE_BYTES.")
            return attrs, name, data
    except ValidationError as exc:
        raise RagError(422, invalid_fields(exc.errors())) from exc
    except (ValueError, TypeError, HTTPException, MultipartParseError) as exc:
        raise RagError(422, "JSON, métadonnées ou formulaire multipart invalide.") from exc


def create_app(settings=None, *, store=None, embeddings=None, generation=None):
    """Les tests fournissent leurs propres clients ; en service, le câblage est unique."""
    cfg = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        # Tous les appels partagent ces clients. yield sépare le démarrage de l'arrêt,
        # où les connexions réseau sont libérées, même en cas d'erreur.
        components = [store or Store(cfg), embeddings or Embeddings(cfg), generation or Generation(cfg)]
        app.state.rag = RagService(cfg, *components)
        try:
            yield
        finally:
            for component in reversed(components):
                component.close()

    api = FastAPI(title="Moteur RAG documentaire", version="1.0.0", lifespan=lifespan,
                  description="Indexer des sources, inspecter les passages et obtenir une réponse avec citations.")

    @api.exception_handler(RagError)
    async def rag_error(request, exc):
        log.warning("Échec de %s : HTTP %d", request.url.path, exc.status)
        return JSONResponse({"detail": exc.detail}, status_code=exc.status)

    @api.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        return JSONResponse({"detail": invalid_fields(exc.errors())}, status_code=422)

    @api.get("/health", response_model=Health, responses={503: {"model": Health}},
             summary="Vérifier la collection Qdrant et le modèle d’embeddings")
    def health(request: Request, response: Response):
        # Sonde les dépendances de recherche ; tester le générateur ici risquerait
        # de déclencher une requête payante à chaque vérification de santé.
        rag = request.app.state.rag
        qdrant, embeddings_status, details = "ok", "ok", []
        try:
            rag.store.ensure()
        except RagError as exc:
            qdrant = "incompatible" if exc.status == 409 else "unavailable"
            details.append(exc.detail)
        try:
            rag.embeddings.check_model()
        except RagError as exc:
            embeddings_status = "unavailable"
            details.append(exc.detail)
        if details:
            response.status_code = 503
        return Health(status="degraded" if details else "ready", qdrant=qdrant,
                      embeddings=embeddings_status, details=details)

    @api.post("/ingest", response_model=IngestResult, openapi_extra={"requestBody": INGEST_BODY},
              summary="Indexer du texte (JSON) ou un document (multipart)")
    async def ingest(request: Request):
        rag = request.app.state.rag
        attrs, name, data = await read_ingest(request, cfg)
        if name is None:
            return await run_in_threadpool(rag.index, [(None, attrs.text)], attrs.source_name or "texte",
                                           "text", attrs.metadata)
        return await run_in_threadpool(rag.ingest_file, name, data, attrs)

    @api.post("/search", response_model=TestResult | ProdResult,
              summary="Inspecter le prompt (test) ou interroger le modèle (prod)")
    def search(req: SearchInput, request: Request):
        return request.app.state.rag.search(req)

    return api


app = create_app()

"""Le contrat public de l'API. Les filtres utilisent seulement des égalités exactes."""
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StringConstraints, field_validator

SourceName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)]
Scalar = Annotated[str, StringConstraints(max_length=1024)] | Annotated[StrictInt, Field(ge=-(2**63), le=2**63-1)] | StrictBool
Metadata = Annotated[dict[Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")], Scalar], Field(max_length=32)]


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class SourceInput(Input):
    source_name: SourceName | None = None
    metadata: Metadata = Field(default_factory=dict)


class TextInput(SourceInput):
    text: str = Field(min_length=1)


class SearchInput(Input):
    model_config = ConfigDict(json_schema_extra={"examples": [
        {"query": "Où sont rangés les kits de mosaïque ?", "mode": "test", "top_k": 3},
        {"query": "Où sont rangés les kits de mosaïque ?", "mode": "prod",
         "filters": {"source_name": "atelier.md"}},
    ]})

    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]
    mode: Literal["test", "prod"] = "test"
    top_k: Annotated[StrictInt, Field(ge=1)] = 5
    score_threshold: float | None = Field(default=None, ge=-1, le=1)
    filters: dict[str, Scalar] = Field(default_factory=dict, max_length=32)

    @field_validator("filters")
    @classmethod
    def filter_keys(cls, filters):
        for key, value in filters.items():
            if key in ("source_name", "source_type", "document_id"):
                if not isinstance(value, str):
                    raise ValueError(f"{key} doit être une chaîne.")
            elif key == "page":
                if type(value) is not int or value < 1:
                    raise ValueError("page doit être un entier positif (première page : 1).")
            elif not re.fullmatch(r"metadata\.[A-Za-z_][A-Za-z0-9_-]{0,63}", key):
                raise ValueError("Filtre inconnu : utiliser source_name, source_type, document_id, page ou metadata.<clé>.")
        return filters


class Source(BaseModel):
    source_name: str
    source_type: str
    metadata: Metadata


class IngestResult(Source):
    document_id: str
    status: Literal["indexed"] = "indexed"
    chunks_indexed: int
    chunk_ids: list[str]


class SourceRef(Source):
    chunk_id: str
    document_id: str
    chunk_index: int
    page: int | None


class Hit(SourceRef):
    """Un passage classé par Qdrant. Le score cosinus n'est pas une probabilité."""
    text: str
    score: float


class Citation(SourceRef):
    """Lien entre un repère du prompt (S1, S2…) et le passage stocké dans Qdrant."""
    citation: str


class Message(BaseModel):
    role: Literal["system", "user"]
    content: str


class SearchResult(BaseModel):
    status: Literal["ok", "no_context", "context_limit"]
    query: str
    metric: Literal["cosine"] = "cosine"
    sources: list[Citation]
    detail: str | None = None


class TestResult(SearchResult):
    mode: Literal["test"] = "test"
    results: list[Hit]
    prompt: list[Message]


class ProdResult(SearchResult):
    mode: Literal["prod"] = "prod"
    answer: str | None = None
    provider: str | None = None
    model: str | None = None


class Health(BaseModel):
    status: Literal["ready", "degraded"]
    qdrant: Literal["ok", "unavailable", "incompatible"]
    embeddings: Literal["ok", "unavailable"]
    details: list[str] = Field(default_factory=list)

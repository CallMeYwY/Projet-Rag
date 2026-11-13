"""Configuration unique : les mêmes paramètres servent à indexer et à rechercher."""
import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Settings(BaseModel):
    """Réglages communs à toute une instance, chargés au démarrage de l'API."""
    model_config = ConfigDict(frozen=True, allow_inf_nan=False, hide_input_in_errors=True)
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = Field(default="", repr=False)
    collection: str = Field(default="documents_v2", min_length=1)
    # Ces paramètres décrivent l'espace de recherche. Changer de modèle nécessite
    # de recalculer les vecteurs, même si leur nombre de dimensions reste identique.
    ollama_url: str = "http://localhost:11434"
    embed_model: str = Field(default="nomic-embed-text", min_length=1)
    embed_dim: int = Field(default=768, gt=0)
    embed_batch_size: int = Field(default=32, gt=0)
    timeout: int = Field(default=120, gt=0)
    # Le générateur lit du texte : il peut changer sans réindexer les documents.
    llm_provider: Literal["ollama", "openrouter"] = "ollama"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = Field(default="gemma3:4b", min_length=1)
    llm_api_key: str = Field(default="", repr=False)
    llm_timeout: int = Field(default=180, gt=0)

    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)
    max_chunks: int = Field(default=2048, gt=0)
    max_top_k: int = Field(default=30, ge=5)
    score_threshold: float | None = Field(default=None, ge=-1, le=1)
    max_context_chars: int = Field(default=6000, gt=0)
    max_file_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    max_extracted_chars: int = Field(default=250_000, gt=0)
    max_pages: int = Field(default=100, gt=0)
    max_image_pixels: int = Field(default=20_000_000, gt=0)
    ocr_languages: str = Field(default="fra+eng", pattern=r"^[A-Za-z0-9_]+(\+[A-Za-z0-9_]+)*$")
    ocr_timeout: int = Field(default=30, gt=0)

    @model_validator(mode="after")
    def validate_settings(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP doit être inférieur à CHUNK_SIZE.")
        return self

    @classmethod
    def from_env(cls):
        # Les variables hors du projet (PATH, CI, etc.) n'entrent jamais dans le modèle.
        values = {name: os.environ[name.upper()] for name in cls.model_fields if name.upper() in os.environ}
        if values.get("score_threshold") == "":
            values.pop("score_threshold")
        return cls(**values)

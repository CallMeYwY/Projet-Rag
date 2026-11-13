"""Génération par Ollama ou OpenRouter via le même contrat Chat Completions."""

from typing import TYPE_CHECKING

import httpx

from .errors import RagError
from .model_http import post_json

if TYPE_CHECKING:
    from .config import Settings


class Generation:
    """Transmet au modèle de rédaction la question et les extraits choisis par le RAG."""

    def __init__(self, settings: "Settings", client: httpx.Client | None = None):
        self.settings = settings
        self.client = client if client is not None else httpx.Client()

    def chat(self, messages: list[dict]) -> str:
        """Envoie les messages construits par prompt.py et renvoie le texte de la réponse."""
        settings = self.settings
        if settings.llm_provider == "openrouter" and not settings.llm_api_key:
            raise RagError(503, "Configurez LLM_API_KEY pour générer une réponse avec OpenRouter.")
        headers = {"Authorization": f"Bearer {settings.llm_api_key}"} if settings.llm_api_key else {}
        # Les deux fournisseurs acceptent Chat Completions. Avec stream=False,
        # l'API attend la réponse complète avant de la retourner avec les sources.
        data = post_json(
            self.client, settings.llm_base_url.rstrip("/") + "/chat/completions",
            # Pour une question documentaire, privilégier une réponse stable
            # lorsque l'on rejoue la même recherche avec les mêmes passages.
            {"model": settings.llm_model, "messages": messages, "stream": False, "temperature": 0},
            headers=headers, service=settings.llm_provider, timeout=settings.llm_timeout,
        )
        try:
            answer = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RagError(502, "Le fournisseur n’a pas renvoyé de réponse de génération.") from exc
        if not isinstance(answer, str) or not answer.strip():
            raise RagError(502, "Le fournisseur a renvoyé une réponse vide.")
        return answer.strip()

    def close(self) -> None:
        self.client.close()

"""Appels JSON aux modèles, avec des erreurs lisibles pour les routes FastAPI."""

import httpx

from .errors import RagError


def post_json(client: httpx.Client, url: str, body: dict, *, service: str,
              timeout: float, headers: dict | None = None) -> dict:
    """Envoie un appel au modèle et traduit les échecs pour le client du RAG.

    Le statut 503 indique un problème de connexion, le 504 un délai dépassé.
    Un modèle joignable qui refuse la requête ou renvoie un JSON illisible
    produit un 502. On peut ainsi identifier l'étape à vérifier depuis l'API.
    """
    try:
        response = client.post(url, json=body, headers=headers, timeout=timeout)
        response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise RagError(504, f"Délai dépassé lors de l’appel à {service}.") from exc
    except httpx.RequestError as exc:
        raise RagError(503, f"{service} est inaccessible. Vérifiez son adresse et son démarrage.") from exc
    except httpx.HTTPStatusError as exc:
        # La réponse brute peut reprendre le document, le prompt ou des données de compte.
        status = exc.response.status_code
        explanations = {
            400: "Vérifiez le modèle et la longueur du texte envoyé.",
            401: "Vérifiez la clé d’API.",
            402: "Vérifiez les crédits disponibles auprès du fournisseur.",
            403: "Vérifiez les droits de la clé d’API.",
            404: "Vérifiez le nom du modèle, son installation et l’URL de base.",
            429: "La limite de requêtes du fournisseur est atteinte.",
        }
        detail = explanations.get(status, "Vérifiez l’état du service.")
        raise RagError(502, f"{service} a renvoyé HTTP {status}. {detail}") from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise RagError(502, f"{service} a renvoyé un JSON illisible.") from exc
    if not isinstance(data, dict):
        raise RagError(502, f"Format de réponse inattendu de {service}.")
    return data

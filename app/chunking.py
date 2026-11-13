from collections.abc import Iterator


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> Iterator[str]:
    """Produit les passages d'une seule page, sans jamais traverser la page suivante.

    On conserve une partie de la fenêtre précédente pour qu'une phrase près
    de la coupure reste compréhensible. Les distances sont en caractères, pas en tokens.
    """
    if not 0 <= overlap < size:
        raise ValueError("Il faut 0 <= overlap < size.")
    # Les PDF ajoutent souvent des retours à la ligne au milieu des phrases.
    text = " ".join(text.split())
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Une coupure trop tôt ferait tourner la boucle sur la zone de chevauchement.
            cut = text.rfind(" ", start + max(size // 2, overlap + 1), end)
            if cut != -1:
                end = cut
        passage = text[start:end].strip()
        if passage:
            yield passage
        if end == len(text):
            break
        start = end - overlap

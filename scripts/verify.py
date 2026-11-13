"""Vérification d'intégration sur l'API lancée. Aucune génération par défaut."""
import argparse
import json
from io import BytesIO
from pathlib import Path

import httpx
from docx import Document

DEMO = Path(__file__).resolve().parents[1] / "demo"
TAG = {"validation": "demo"}


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify(client, args):
    def call(method, route, **kwargs):
        res = client.request(method, route, **kwargs)
        res.raise_for_status()
        return res.json()

    def search(question, name, mode="test"):
        return call("POST", "/search", json={
            "query": question, "mode": mode,
            "filters": {"source_name": name, "metadata.validation": TAG["validation"]},
        })

    health = call("GET", "/health")
    check(health["status"] == "ready", f"API en mode dégradé : {health}")
    question = "Où sont rangés les kits de mosaïque ?"
    if args.persistence:
        # Lecture seule : ne pas réingérer ce que l'on cherche à retrouver après redémarrage.
        result = search(question, "atelier.md")
        check(any("Indigo" in hit["text"] for hit in result["results"]), "Document absent après redémarrage.")
        print("Persistance vérifiée :", result["sources"][0]["chunk_id"])
        return

    doc = {"source_name": "demo-agenda", "text": "La navette de la Verrière part du quai Azur à 9 heures.",
           "metadata": TAG}
    first = call("POST", "/ingest", json=doc)
    second = call("POST", "/ingest", json=doc)
    result = search("De quel quai part la navette ?", "demo-agenda")
    check(first == second and len(result["results"]) == 1
          and result["sources"][0]["chunk_id"] == first["chunk_ids"][0], "Déduplication incorrecte.")
    print("Texte JSON : un passage après deux ingestions.")

    # Le DOCX est fabriqué en mémoire : la vérification s'exécute sur n'importe quelle machine.
    word = Document()
    word.add_paragraph("Le parapluie de l'accueil est rangé dans le coffre Églantine.")
    buf = BytesIO()
    word.save(buf)
    samples = [
        ("atelier.md", (DEMO / "atelier.md").read_bytes(), question, "Indigo", None),
        ("consigne.png", (DEMO / "consigne.png").read_bytes(), "Où a lieu le rassemblement ?", "Jasmin", None),
        ("manuel.pdf", (DEMO / "manuel.pdf").read_bytes(), "Où se range la housse du projecteur ?", "Opale", 2),
        ("guide.docx", buf.getvalue(), "Où est le parapluie ?", "Églantine", None),
        ("accueil.txt", b"Le registre est sur la table Corail.", "Où est le registre ?", "Corail", None),
    ]
    for name, data, query, expected, page in samples:
        indexed = call("POST", "/ingest", files={"file": (name, data)}, data={"metadata": json.dumps(TAG)})
        result = search(query, name)
        check(any(expected.lower() in hit["text"].lower() and hit["page"] == page and hit["metadata"] == TAG
                  and hit["chunk_id"] in indexed["chunk_ids"] for hit in result["results"]),
              f"Passage ou source incorrect pour {name}.")
        print(f"{name} : « {expected} » retrouvé" + (f", page {page}." if page else "."))
    # Tous les documents de la démo sont en concurrence : aucun filtre sur le
    # fichier ou la page ne donne la bonne réponse d'avance au moteur.
    ranked = call("POST", "/search", json={"query": question, "top_k": 3,
                                           "filters": {"metadata.validation": TAG["validation"]}})
    check(ranked["results"] and ranked["results"][0]["source_name"] == "atelier.md",
          "Le document de l'atelier n'arrive pas en tête des démos.")
    print("Classement entre documents : atelier.md arrive en tête pour les kits de mosaïque.")
    empty = search(question, "source-absente-verification", mode="prod")
    check(empty["status"] == "no_context" and not empty["sources"] and empty["answer"] is None,
          "Le filtre sans résultat devrait renvoyer no_context.")
    if args.prod:
        result = search(question, "atelier.md", mode="prod")
        check(result["sources"] and result["sources"][0]["source_name"] == "atelier.md"
              and "indigo" in (result["answer"] or "").lower(), "Réponse de démonstration inattendue.")
        print(f"{result['provider']} / {result['model']} : {result['answer']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="URL de l'API RAG")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prod", action="store_true", help="Appeler le générateur configuré une fois.")
    modes.add_argument("--persistence", action="store_true", help="Relire uniquement le document déjà indexé.")
    options = parser.parse_args()
    with httpx.Client(base_url=options.url, timeout=300) as api:
        verify(api, options)

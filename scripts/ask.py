#!/usr/bin/env python3
"""Interroge le moteur RAG depuis le terminal, avec la bibliothèque standard."""

import argparse
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description="Poser une question au moteur RAG.")
    parser.add_argument("question", nargs="+", help="Question, avec ou sans guillemets")
    parser.add_argument("--url", default=os.getenv("VECTOR_API", "http://127.0.0.1:8000"),
                        help="URL de base de l’API (défaut : VECTOR_API ou http://127.0.0.1:8000)")
    parser.add_argument("--test", action="store_true", help="Afficher les passages et le prompt sans génération")
    parser.add_argument("--source", help="Limiter la recherche à un document, par son nom exact")
    parser.add_argument("--timeout", type=int, default=300, metavar="SECONDES",
                        help="Délai d’attente de la réponse (300 secondes par défaut)")
    args = parser.parse_args()
    question = " ".join(args.question).strip()
    if not question or args.timeout < 1:
        parser.error("La question doit contenir du texte et le délai doit être positif.")

    # Toute la recherche et la construction du prompt ont lieu dans l'API.
    body = {"query": question, "mode": "test" if args.test else "prod"}
    if args.source:
        # Utile quand deux manuels emploient le même terme pour des fonctionnalités voisines.
        body["filters"] = {"source_name": args.source}
    try:
        req = Request(args.url.rstrip("/") + "/search", data=json.dumps(body).encode("utf-8"),
                      headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=args.timeout) as response:
            result = json.load(response)
        if not isinstance(result, dict) or result.get("mode") != body["mode"]:
            raise ValueError("Format de réponse inattendu")
        if args.test:
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        if result["status"] == "ok":
            print(result["answer"])
            print(f"\nModèle : {result['provider']} / {result['model']}")
            print("Passages envoyés au modèle (les repères cités sont dans sa réponse) :")
            for source in result["sources"]:
                page = f", page {source['page']}" if source.get("page") is not None else ""
                print(f"[{source['citation']}] {source['source_name']}{page}")
        else:
            print(result["detail"])
        return 0
    except HTTPError as exc:
        try:
            detail = json.loads(exc.read(65536)).get("detail", exc.reason)
        except (ValueError, AttributeError):
            detail = exc.reason
        print(f"Erreur HTTP {exc.code} : {detail}", file=sys.stderr)
    except (URLError, OSError) as exc:
        print(f"Impossible de joindre le moteur RAG : {exc}", file=sys.stderr)
    except (ValueError, KeyError, TypeError):
        print("URL incorrecte ou réponse illisible du moteur RAG.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

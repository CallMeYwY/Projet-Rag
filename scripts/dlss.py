#!/usr/bin/env python3
"""Teste le RAG sur trois PDF NVIDIA originaux, sans fabriquer de texte de démonstration."""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import httpx

ROOT = Path(__file__).resolve().parents[1]
ORIGINAUX = ROOT / "demo" / "dlss" / "originaux"
REV = "9981c6350dc58c0ce21f024fd65e31e851c31689"
CORPUS = f"nvidia-dlss-{REV[:8]}"
FILES = {
    "DLSS_Programming_Guide_Release.pdf": "dca4d2e68ccedc04a431803595867a893474b1cce033e65a41e9eab9f6b436f8",
    "DLSS-RR Integration Guide.pdf": "cc53ddaab916e067525077655f947458fc1854abc15b9cc3e484ad8eb9e96b1a",
    "RTX UI Developer Guidelines.pdf": "33ece89ff05e2ac0ce747ec467d4780314711aee532673fe6dfcc6cecd40e7d2",
}
# Les faits de référence ont été vérifiés dans les PDF avant la recherche.
# Ces chaînes servent à évaluer la sortie ; elles ne sont jamais envoyées au moteur.
CASES = [
    ("dlaa", "En mode DLAA (Deep Learning Anti-Aliasing), quel est le rapport entre la résolution d’entrée et la résolution de sortie ?",
     r"(1\s*[:/]\s*1\b|\b1[.,]0\b|mêmes? (dimensions|résolutions?)|identique|égale|1 pour 1)",
     [r"\bdlaa\b", r"(ratio of 1\.0|same (value|resolution)|native resolution|equal to the output)"]),
    ("motion", "DLSS Ray Reconstruction : quels sont les deux formats de texture acceptés pour les Motion Vectors ?",
     r"(?s)(?=.*RG16_FLOAT)(?=.*RG32_FLOAT)", [r"RG16_FLOAT", r"RG32_FLOAT"]),
    ("6x", "En mode 6X de DLSS Multi Frame Generation, combien d’images supplémentaires sont générées pour une image rendue ?",
     r"(?s)(?:cinq|\b5\b)[^.!?]{0,65}(?:image|frame)|(?:image|frame)[^.!?]{0,65}(?:cinq|\b5\b)",
     [r"\b6x\b", r"five additional frames"]),
    ("reflets", "À quoi servent les Specular Motion Vectors de DLSS Ray Reconstruction pour les reflets en mouvement ?",
     r"(?s)(?=.*(?:reflets|réflexions))(?=.*(?:mouvement|déplacement))(?=.*(?:qualité|amélior))",
     [r"specular motion vectors", r"quality of reflections during motion"]),
]


def source_url(name):
    return f"https://raw.githubusercontent.com/NVIDIA/DLSS/{REV}/doc/{quote(name)}"


def get_pdf(name, local_guide):
    """Réutilise l'original local s'il a la bonne empreinte, sinon récupère la version figée."""
    path = local_guide if name == "DLSS_Programming_Guide_Release.pdf" and local_guide else ORIGINAUX / name
    if path.is_file():
        data = path.read_bytes()
    elif local_guide and path == local_guide:
        raise ValueError(f"Guide local introuvable : {path}")
    else:
        # On ne conserve jamais une réponse HTML ou un téléchargement incomplet en cache.
        with urlopen(source_url(name), timeout=60) as response:
            data = response.read(10 * 1024 * 1024 + 1)
    if hashlib.sha256(data).hexdigest() != FILES[name]:
        raise ValueError(f"{name} : fichier différent de la version NVIDIA attendue ({REV[:8]}).")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return data


def request(api, method, route, **kw):
    response = api.request(method, route, **kw)
    if response.is_error:
        raise ValueError(f"{route} : HTTP {response.status_code} : {response.text[:300]}")
    return response.json()


def ingest(api, local_guide):
    records = []
    for name in FILES:
        data = get_pdf(name, local_guide)
        # Les URL de provenance restent dans le payload du passage, au côté du numéro de page.
        meta = {"corpus": CORPUS, "editeur": "NVIDIA", "url": source_url(name), "sha256": FILES[name]}
        r = request(api, "POST", "/ingest", files={"file": (name, data, "application/pdf")},
                    data={"metadata": json.dumps(meta)})
        records.append({"file": name, "sha256": FILES[name], "document_id": r["document_id"],
                        "chunks_indexed": r["chunks_indexed"]})
        print(f"{name} : {r['chunks_indexed']} passages", flush=True)
    return records


def evaluate(api, prod, k):
    results = []
    for key, question, answer_pattern, evidence in CASES:
        # Filtrer le corpus exclut les anciennes démos mais laisse les trois guides et
        # toutes leurs pages en concurrence. La question ne donne aucun numéro de page.
        req = {"query": question, "filters": {"metadata.corpus": CORPUS}, "top_k": k}
        debug = request(api, "POST", "/search", json={**req, "mode": "test"})
        hits = {h["chunk_id"]: h for h in debug["results"]}
        support = []
        for ref in debug["sources"]:
            hit = hits[ref["chunk_id"]]
            normalized = " ".join(hit["text"].replace("\u200b", "").split())
            if all(re.search(pattern, normalized, re.I) for pattern in evidence):
                support.append(hit["chunk_id"])
        row = {"case": key, "question": question, "top_k": k, "evidence_in_prompt": bool(support),
               "ranking": [{"file": h["source_name"], "page": h["page"], "score": round(h["score"], 4),
                            "chunk_id": h["chunk_id"], "supports": h["chunk_id"] in support}
                           for h in hits.values()]}
        if prod:
            response = request(api, "POST", "/search", json={**req, "mode": "prod"})
            answer = response["answer"] or ""
            # Les modèles écrivent parfois « [S1, S4] » au lieu de « [S1] [S4] ».
            cited = {token for group in re.findall(r"\[(S\d+(?:\s*,\s*S\d+)*)\]", answer)
                     for token in re.findall(r"S\d+", group)}
            refs = [{"cite": s["citation"], "file": s["source_name"], "page": s["page"],
                     "supported": s["chunk_id"] in support}
                    for s in response["sources"] if s["citation"] in cited]
            correct = bool(re.search(answer_pattern, answer, re.I))
            if key == "6x" and re.search(r"(six|\b6\b)\s+images?\s+supplémentaires?", answer, re.I):
                # Citer correctement la ligne anglaise ne rattrape pas un total mal traduit.
                correct = False
            row.update(answer=answer, provider=response["provider"], model=response["model"], citations=refs,
                       expected_answer=correct,
                       cites_support=any(s["supported"] for s in refs))
        print(f"{key} : preuve retrouvée = {bool(support)}" +
              (f" ; réponse attendue = {row['expected_answer']} ; citation étayée = {row['cites_support']}"
               if prod else ""), flush=True)
        results.append(row)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.getenv("VECTOR_API", "http://127.0.0.1:8000"))
    parser.add_argument("--guide-local", type=Path, help="PDF de programmation personnel identique à la version testée")
    parser.add_argument("--prod", action="store_true", help="Appeler le modèle configuré pour chaque question")
    parser.add_argument("--top-k", type=int, default=5, help="Nombre de passages à retrouver (défaut : 5)")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--index-only", action="store_true", help="Indexer les PDF sans poser de question")
    modes.add_argument("--search-only", action="store_true", help="Interroger les PDF déjà indexés")
    parser.add_argument("--rapport", type=Path, help="Conserver le classement, les sources et les réponses en JSON")
    args = parser.parse_args()
    with httpx.Client(base_url=args.url, timeout=360) as api:
        health = request(api, "GET", "/health")
        report = {"revision": REV, "corpus": CORPUS, "health": health}
        if not args.search_only:
            report["documents"] = ingest(api, args.guide_local)
        results = [] if args.index_only else evaluate(api, args.prod, args.top_k)
        report["results"] = results
    if args.rapport:
        args.rapport.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return int(any(not r["evidence_in_prompt"] or (args.prod and not (r["expected_answer"] and r["cites_support"]))
                   for r in results))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, httpx.HTTPError) as error:
        sys.exit(f"Vérification interrompue : {error}")

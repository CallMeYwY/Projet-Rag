# Vérification du 12 novembre 2026

**Résultat :** l’API Docker, la recherche Qdrant, l’OCR et la génération par Ollama ont fonctionné ensemble. OpenRouter a été testé avec un serveur HTTP simulé ; aucune clé ni requête OpenRouter réelle n’a été utilisée.

## Point de départ

Le ZIP d’origine a été ouvert et son code relu : API FastAPI, modules de découpage, d’embeddings et de stockage, script de génération séparé, deux exemples de configuration et Compose à trois services. Dans une copie temporaire, `docker compose config --quiet` a réussi. La première tentative `docker compose up -d --build` n’a pas pu joindre le démon Docker, alors arrêté. Les essais suivants portent sur l’implémentation livrée, après démarrage de Docker Desktop et d’Ollama.

## Vérifications exécutées

| Commande ou scénario | Résultat observé |
|---|---|
| `docker compose config --quiet` ; idem avec `--env-file .env.openrouter.example` | Les deux configurations sont valides. |
| `docker build --target tests -t rag-tests .` puis `docker run --rm rag-tests` | **44 tests réussis, aucun ignoré**, dont l’OCR PNG, JPEG et PDF. Un avertissement de dépréciation concerne le client de test Starlette/httpx. |
| `.venv/bin/python -m pytest -q` ; `.venv/bin/python -m pip check` | 43 réussis, 1 ignoré (Tesseract absent sur l’hôte) ; dépendances cohérentes. |
| `docker compose up -d --build --wait` | API et Qdrant 1.20.0 démarrés, état Docker healthy. `GET /health` : 200, `ready`, `qdrant: ok`, `embeddings: ok`. |
| Ingestions JSON, TXT, Markdown, DOCX, PNG et PDF, avec embeddings Ollama `nomic-embed-text` | Le texte JSON est retrouvé après deux ingestions avec les mêmes UUID et un seul résultat. Extraits, métadonnées et pages correspondent aux fichiers : Indigo dans le Markdown, Jasmin dans l’image, Opale sur la page 2 scannée du PDF. |
| `python3 scripts/ask.py --test "Où sont rangés les kits de mosaïque ?"` | Sans filtre, `atelier.md` est arrivé premier parmi les documents indexés (cosinus ≈ 0,696 lors de cet essai). Ce score ne représente pas une probabilité. |
| Recherche en mode prod via Ollama `gemma3:4b` | Réponse observée : « Les kits de mosaïque sont rangés dans l’armoire Indigo, au fond de la salle [S1]. » La source associée à `S1` est `atelier.md`. |
| Recherche en mode prod avec filtre sur une source absente | `no_context`, sources vides, pas de réponse générée. L’absence d’appel est aussi contrôlée avec un générateur espion dans les tests automatisés. |
| Arrêt du service Qdrant du projet puis appel à `/health` | 503, `degraded`, Qdrant indisponible ; le modèle d’embeddings restait joignable. |
| Recréation des conteneurs puis `python scripts/verify.py --persistence` | L’armoire Indigo a été retrouvée sans nouvelle ingestion. Les services ont été laissés en fonctionnement. |

## À vérifier avec sa propre configuration

- **OpenRouter réel** : configurer sa clé et un modèle autorisé, relancer l’API puis faire une recherche en mode prod. Les tests contrôlent précisément l’URL, l’en-tête Bearer, le modèle, les messages et la lecture de la réponse ; ils ne valident pas les droits ni les crédits d’un compte.
- **Ollama dans Docker** : le profil alternatif est fourni. L’essai de génération réel ci-dessus utilise Ollama sur l’hôte, joignable depuis le conteneur API.
- Les vérifications portent sur le corpus de démonstration. Pour un corpus différent, vérifier les premiers résultats en mode test et choisir un seuil de similarité à partir de questions dont on connaît les réponses.

Pour démarrer ou relancer : `docker compose up -d --build`. Les commandes à reproduire figurent dans le [README](README.md).

## Essai sur les documents NVIDIA DLSS

Trois PDF originaux ont été indexés sur la même API : **301 + 112 + 38 = 451 passages**. Le guide Super Resolution lu depuis Téléchargements est identique à celui publié par NVIDIA ; les deux autres PDF proviennent directement du [même dépôt officiel](https://github.com/NVIDIA/DLSS/tree/9981c6350dc58c0ce21f024fd65e31e851c31689/doc). Le [protocole reproductible](demo/dlss/README.md) et le [classement avec les réponses observées](demo/dlss/resultats.json) sont fournis. La preuve de référence a été vérifiée dans les PDF avant le test.

Les questions ci-dessous ont été posées en français avec `nomic-embed-text`, puis `gemma3:4b` pour la génération, sans fournir au moteur un nom de fichier ni une page. Seul le corpus NVIDIA était filtré. Pour les quatre questions, **au moins un passage contenant la preuve figurait parmi les cinq résultats, puis dans le contexte envoyé au modèle**. Trois réponses sur quatre donnent le fait attendu avec au moins une citation qui l’étaye.

| Question résumée | Réponse vérifiée dans le PDF | Ce que le RAG a fait |
|---|---|---|
| Rapport entre résolution d’entrée et de sortie en DLAA ? | 1:1, les deux sont identiques. Guide Super Resolution, page PDF 21 (page imprimée 11). | Réponse correcte, citation de la page 21. |
| Formats des vecteurs de mouvement ? | `RG16_FLOAT` ou `RG32_FLOAT`. Guide Ray Reconstruction, page 16 ; mêmes formats dans le guide Super Resolution, page 26. | Les deux formats sont corrects ; le modèle cite la page 26 de Super Resolution, dont un passage étaye la réponse. La recherche limitée au guide Ray Reconstruction a aussi fourni les deux formats, avec sa page 16. |
| En mode 6X, combien d’images **supplémentaires** par image rendue ? | **5 supplémentaires**, soit 6 au total. Guide d’interface RTX, page 6. | **Échec en recherche générale** : la bonne ligne est dans le contexte, mais le modèle répond 6 et cite la page 4 parlant du réglage par défaut. En limitant le contexte à la page 6 du guide, il répond 5. |
| Que font les vecteurs de mouvement spéculaires pour les reflets ? | Ils améliorent la qualité des reflets en mouvement. Guide Ray Reconstruction, page 17. | Réponse correcte, citation de la page 17. |

Le premier essai sur le mode 6X confondait déjà les images supplémentaires avec le total. Plusieurs formulations des consignes ont été comparées avec les mêmes passages : un essai a donné 5, mais la correction n’a pas été constante. La version livrée demande une courte preuve tirée d’un passage, utilise une température de génération de 0 et ne contient aucune règle propre à NVIDIA. **La recherche et la génération sont donc évaluées séparément** : le test documentaire signale un échec (code de sortie 1) quand le modèle lit mal un chiffre, même si les tests logiciels passent.

Pour reproduire le contrôle ciblé de la page du tableau après ingestion :

```sh
curl -fsS http://localhost:8000/search -H 'Content-Type: application/json' \
  -d '{"mode":"prod","query":"En mode 6X de DLSS Multi Frame Generation, combien d’images supplémentaires sont générées pour une image rendue ?","filters":{"metadata.corpus":"nvidia-dlss-9981c635","source_name":"RTX UI Developer Guidelines.pdf","page":6}}'
```

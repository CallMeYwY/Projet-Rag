# Moteur RAG documentaire

Pose une question à tes documents et retrouve le passage qui a servi à répondre. **Le mode `test` montre les extraits, leurs scores et le prompt. Le mode `prod` demande en plus une réponse à Ollama ou à OpenRouter.** L’indexation et la recherche utilisent Ollama et Qdrant dans les deux modes.

## Essayer avec l’exemple fourni

Prérequis : Docker avec Compose v2 et Ollama lancé sur la machine. Les commandes suivantes s’exécutent depuis le dossier du projet. Sur macOS, ouvrir Docker Desktop et Ollama ; sur un système sans service Ollama démarré, lancer `ollama serve` dans un autre terminal.

```sh
ollama pull nomic-embed-text
ollama pull gemma3:4b
docker compose up -d --build
curl -sS http://localhost:8000/health
```

Les modèles ne sont téléchargés que s’ils manquent. Au premier démarrage, attendre que `/health` renvoie HTTP 200 avec `status: "ready"`. Si le service n’est pas prêt, consulter `docker compose ps` et `docker compose logs --tail=50 api`. La [documentation interactive](http://localhost:8000/docs) permet ensuite d’essayer les trois routes.

Indexer le document de démonstration, puis lui poser une question :

```sh
curl -fsS http://localhost:8000/ingest -F 'file=@demo/atelier.md' -F 'metadata={"site":"Verrière"}'

curl -fsS http://localhost:8000/search -H 'Content-Type: application/json' \
  -d '{"query":"Où sont rangés les kits de mosaïque ?","mode":"test","filters":{"source_name":"atelier.md"}}'

curl -fsS http://localhost:8000/search -H 'Content-Type: application/json' \
  -d '{"query":"Où sont rangés les kits de mosaïque ?","mode":"prod","filters":{"source_name":"atelier.md"}}'
```

Le passage attendu mentionne **l’armoire Indigo**. Dans le dernier résultat, `answer` contient la réponse et `sources` relie `S1` à `atelier.md` et à son identifiant de passage. Tu peux aussi utiliser `python3 scripts/ask.py "Où sont rangés les kits de mosaïque ?"` depuis le terminal ; ajouter `--test` pour voir le détail de la recherche.

Pour indexer **du texte directement**, sans créer de fichier :

```sh
curl -fsS http://localhost:8000/ingest -H 'Content-Type: application/json' \
  -d '{"text":"La navette part du quai Azur à 9 heures.","source_name":"horaires","metadata":{"site":"Verrière","annee":2026}}'
```

## Essayer sur de vrais PDF NVIDIA DLSS

Le [test DLSS](demo/dlss/README.md) utilise trois guides officiels en anglais : il permet de poser des questions en français et de contrôler les pages citées. Sur une machine neuve, le script récupère les originaux depuis la révision NVIDIA documentée, puis les indexe. Sur une installation déjà alimentée :

```sh
python3 scripts/ask.py "En mode DLAA, quel est le rapport entre la résolution d’entrée et celle de sortie ?"
python3 scripts/ask.py --source "RTX UI Developer Guidelines.pdf" "En mode 6X, combien d’images supplémentaires sont générées pour une image rendue ?"
```

Pour **relire le code** avec ces exemples, suis [`RagService.index`](app/service.py) : le PDF y arrive sous forme de couples *page, texte*, puis chaque passage reçoit un identifiant et un vecteur. Passe ensuite à `RagService.search` dans le même fichier : la question devient un vecteur, Qdrant sélectionne les passages et [`build_prompt`](app/prompt.py) les associe aux repères comme `[S1]`. La page affichée vient du PDF ; elle n’est pas devinée par le générateur.

## Ce qui vient du prototype

Le ZIP séparait le service vectoriel (`app/`) du script de génération (`engine/`). Il savait découper une chaîne, appeler `/api/embed` avec les préfixes Nomic, indexer des UUIDv5 dans une collection Qdrant en cosinus et interroger un modèle compatible Chat Completions. Sa route de santé était statique ; il ne lisait pas les fichiers et supprimait une source **avant** de calculer ses nouveaux vecteurs.

Le projet reprend le découpage avec chevauchement, les préfixes, les UUIDv5, les opérations `upsert`/`query_points`, les consignes de citation et le client Chat Completions du ZIP. La génération se fait maintenant dans l’API : le client terminal et les deux modes de recherche utilisent le même prompt. La nouvelle collection par défaut s’appelle `documents_v2` pour ne pas mélanger les anciens points et les nouveaux. Les anciennes requêtes `{question,k,source}` deviennent `{query,top_k,filters:{source_name:…}}` ; à l’ingestion, `source` devient `source_name`.

Pour relire le code, commencer par [app/service.py](app/service.py) :

| Fichier | Rôle |
|---|---|
| [main.py](app/main.py), [schemas.py](app/schemas.py) | Routes, validation, formats de réponse, limites des requêtes |
| [extraction.py](app/extraction.py) | Lecture des formats, rendu des pages PDF et OCR |
| [chunking.py](app/chunking.py), [embeddings.py](app/embeddings.py) | Découpage par page et calcul des vecteurs |
| [store.py](app/store.py) | Vérification de la collection, écriture, filtres et recherche |
| [prompt.py](app/prompt.py), [generation.py](app/generation.py) | Sélection des sources selon le budget puis appel au fournisseur |
| [config.py](app/config.py) | Valeurs par défaut et validation des paramètres |

## Configuration des modèles

Les réglages Docker par défaut correspondent à [l’exemple Ollama](.env.example). Pour les personnaliser, copier `.env.example` en `.env`, modifier les valeurs puis relancer `docker compose up -d`. Les clés ne sont pas nécessaires pour Ollama local. L’API et Qdrant n’exposent leurs ports que sur la boucle locale.

- **Ollama sur l’hôte** : `OLLAMA_URL=http://host.docker.internal:11434` pour les embeddings, `LLM_BASE_URL=http://host.docker.internal:11434/v1` pour la génération. Le nom `host.docker.internal` est déclaré aussi pour Linux. Sur Linux, si Ollama écoute uniquement sur `127.0.0.1`, le conteneur ne pourra pas le joindre : le rendre accessible sur l’interface du pont Docker, ou utiliser le profil ci-dessous. Choisir le modèle génératif avec `LLM_MODEL` et l’installer avec `ollama pull <modèle>`.
- **Ollama dans Docker** : créer `.env` à partir de `.env.example` et remplacer les deux URL par `http://ollama:11434` et `http://ollama:11434/v1`. Suivre les commandes ci-dessous. Les modèles sont conservés dans un volume distinct. Le profil n’exige pas qu’Ollama soit installé sur l’hôte.
- **OpenRouter** : partir du [second exemple complet](.env.openrouter.example), en copiant ses réglages dans `.env`. Renseigner sa clé dans `LLM_API_KEY` et choisir un identifiant de modèle disponible sur son compte (`LLM_MODEL`). Les autres valeurs sont `LLM_PROVIDER=openrouter` et `LLM_BASE_URL=https://openrouter.ai/api/v1`. Relancer `docker compose up -d`. Les embeddings restent locaux : garder Ollama et `nomic-embed-text` disponibles. Seul le mode `prod` avec des extraits sélectionnés transmet la question et ces extraits au fournisseur ; il peut être facturé. [Contrat Chat Completions d’OpenRouter](https://openrouter.ai/docs/api-reference/overview).

```sh
# Variante avec Ollama dans Docker, après modification de .env
docker compose --profile ollama up -d --build
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull gemma3:4b
```

| Réglage | Valeur par défaut | Effet |
|---|---:|---|
| `QDRANT_URL`, `COLLECTION` | `http://qdrant:6333`, `documents_v2` | Serveur et collection. `QDRANT_API_KEY` est facultative. |
| `EMBED_MODEL`, `EMBED_DIM`, `EMBED_BATCH_SIZE` | `nomic-embed-text`, 768, 32 | Modèle d’embeddings, dimension et nombre de textes par appel |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | 800, 150 | Fenêtre en caractères et chevauchement maximal ; le chevauchement doit être inférieur à la fenêtre |
| `MAX_TOP_K`, `SCORE_THRESHOLD` | 30, vide | Plafond des résultats ; sans seuil tous les scores sont admissibles |
| `MAX_CONTEXT_CHARS` | 6 000 | Budget des extraits sérialisés, identifiants et séparateurs compris ; hors question et consignes |
| `MAX_FILE_BYTES`, `MAX_EXTRACTED_CHARS`, `MAX_CHUNKS` | 10 Mio, 250 000, 2 048 | Limites d’une ingestion |
| `MAX_PAGES`, `MAX_IMAGE_PIXELS` | 100, 20 000 000 | Pages par PDF, pixels par image |
| `OCR_LANGUAGES`, `OCR_TIMEOUT` | `fra+eng`, 30 s | Langues et délai de chaque appel au rendu ou à l’OCR |
| `TIMEOUT`, `LLM_TIMEOUT` | 120 s, 180 s | Délais pour Qdrant/embeddings et pour la génération |
| `API_PORT`, `QDRANT_PORT` | 8000, 6333 | Ports publiés sur l’hôte |

## Formats et citations

- **JSON**, **TXT**, **Markdown** (`.md`, `.markdown`) : fichiers encodés en UTF-8, BOM accepté. Le nom par défaut est `texte` en JSON, ou le nom du fichier en multipart.
- **PDF** non chiffrés : extraction native par page ; quand une page ne contient pas de texte extractible, Poppler la rend en image et Tesseract la transcrit. La première page porte le numéro 1. Une page contenant à la fois du texte natif et une photo n’est pas retraitée par OCR.
- **DOCX** : texte des paragraphes et tableaux du corps. La pagination dépend du rendu dans Word ; aucune page n’est inventée. Les en-têtes, zones de dessin, images intégrées et notes ne sont pas indexés. Le contenu décompressé du DOCX est limité à quatre fois le plafond de taille des fichiers.
- **PNG**, **JPEG** (`.jpg`, `.jpeg`) : recherche dans le texte reconnu par Tesseract, pas dans le contenu visuel. L’image Docker contient le français et l’anglais. Un scan flou ou manuscrit peut être mal lu ; un fichier sans texte exploitable est refusé. L’audio n’est pas pris en charge.

Les métadonnées sont un objet JSON plat (32 clés maximum, noms ASCII commençant par une lettre ou `_`, puis lettres/chiffres/`_`/`-`). Les valeurs acceptées sont les chaînes, les entiers signés de 64 bits et les booléens. En multipart, envoyer `metadata` comme champ texte contenant cet objet. Le champ `source_name` peut remplacer le nom de fichier visible, sans changer le lecteur choisi par l’extension.

### Comprendre une recherche

`query` est obligatoire. `top_k` vaut 5 par défaut. Les filtres se combinent par **ET** et comparent exactement `source_name`, `source_type`, `document_id`, `page` ou `metadata.<clé>`. Par exemple, `{"filters":{"source_type":"pdf","page":2,"metadata.site":"Verrière"}}`. Les types de sources sont `text` (JSON), `txt`, `md`, `pdf`, `docx`, `png` ou `jpeg`.

- En **test**, `results` contient tous les passages retrouvés avec leur texte et leur score. `prompt` contient les messages exacts destinés au modèle. `sources` ne contient que les passages entiers qui tiennent dans le budget de contexte, avec une étiquette `S1`, `S2`, etc. Les documents sont sérialisés comme données dans un message distinct des consignes système.
- En **prod**, la même sélection est envoyée au fournisseur. `answer` est la réponse du modèle ; `sources` décrit le contexte envoyé. Une citation dans une réponse générée reste à vérifier dans le passage d’origine.
- Un **score cosinus** mesure une proximité vectorielle, pas une probabilité de réponse correcte. Le seuil peut être réglé entre −1 et 1. Sans seuil, même un passage peu pertinent peut être retourné : calibrer le seuil sur des questions représentatives.
- Si aucun passage ne satisfait les critères, le statut est `no_context`. Si des passages sont retrouvés mais qu’aucun ne tient dans le budget, il est `context_limit`. Dans les deux cas, le mode prod renvoie `answer: null` et n’appelle pas le générateur.

### Déduplication, persistance et changement de modèle

L’identifiant de document dépend du nom, du type, des paramètres de découpage et des passages normalisés avec leurs pages. L’UUID de chaque passage dépend ensuite du document et de sa position. Réingérer le même document sous le même nom **met les métadonnées à jour sans créer de doublons**. Modifier le contenu, le nom ou le découpage crée une nouvelle version ; les anciennes versions restent consultables. Utiliser `filters.document_id` pour cibler une version.

Qdrant vérifie la dimension, le cosinus et le nom du modèle enregistré sur la collection. Un changement de modèle (même à dimension égale) demande de **créer une nouvelle collection**, régler `EMBED_MODEL` et `EMBED_DIM`, relancer l’API et réingérer les originaux. Suivre le même processus si les poids d’un modèle changent sous un tag identique : le nom seul ne permet pas de le détecter. Une collection du ZIP sans information de modèle sera signalée comme incompatible.

Le volume Docker `qdrant_data` conserve les passages et vecteurs après un arrêt ou une recréation des conteneurs. `docker compose down` conserve ce volume ; l’option `-v` le supprime. Les fichiers originaux ne sont pas archivés par le service : les conserver pour une future réindexation. Déplacer le dossier du projet permet de relancer un moteur vide sur une autre machine ; migrer aussi le stockage de Qdrant ou réingérer ses sources pour retrouver les données.

## Vérifier le projet

**Tests automatisés, OCR compris** — depuis la racine, sans clé, modèle ni serveur Qdrant :

```sh
docker build --target tests -t rag-tests .
docker run --rm rag-tests
```

Les tests de routes utilisent un client Qdrant en mémoire et des modèles simulés ; les tests de stockage rouvrent aussi un index local sur disque. Les contrats HTTP Ollama et OpenRouter, les deux modes, les limites du contexte, la déduplication, les filtres, les erreurs et les lecteurs de fichiers sont vérifiés. Dans Docker, Tesseract et Poppler sont réellement exécutés sur l’image, sa variante JPEG et le PDF mixte.

Pour exécuter les tests sur l’hôte : créer un environnement **Python 3.10+**, puis `python -m pip install -r requirements-dev.txt` et `python -m pytest -q`. Le test OCR réel est ignoré si les commandes `tesseract` et `pdftoppm` ne sont pas présentes. Si elles sont installées, les langues `fra` et `eng` doivent l’être aussi.

**Parcours d’intégration réel** — l’API doit tourner et les dépendances Python doivent être installées. Le script ingère les démos et les retrouve ; `--prod` ajoute un appel au générateur actif :

```sh
python scripts/verify.py
python scripts/verify.py --prod
docker compose up -d --force-recreate --wait
python scripts/verify.py --persistence
```

Pour tester une indisponibilité sur cette installation de démonstration : `docker compose stop qdrant`, puis `curl -i http://localhost:8000/health` (attendu : 503, `degraded`). Relancer avec `docker compose start qdrant`. La route de santé vérifie l’accès à la collection et la présence du modèle d’embeddings ; elle n’appelle jamais le générateur, dont la disponibilité est vérifiée lors d’une recherche en production.

Les erreurs d’entrée renvoient 422 (contenu invalide), 415 (format non pris en charge) ou 413 (limite dépassée). Une collection incompatible renvoie 409 lors d’une opération ; un service inaccessible renvoie 503, une réponse fournisseur incorrecte 502, un délai dépassé 504. Les journaux indiquent le type d’opération, les nombres de passages et les statuts sans recopier le document ni la clé.

Voir [le compte rendu de validation](VALIDATION.md) pour les essais réellement effectués sur cette installation.

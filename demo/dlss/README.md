# Poser des questions aux guides DLSS de NVIDIA

Le corpus comprend **trois PDF originaux en anglais**, téléchargés depuis [le dépôt NVIDIA/DLSS, révision 9981c635](https://github.com/NVIDIA/DLSS/tree/9981c6350dc58c0ce21f024fd65e31e851c31689/doc). Les questions sont posées en français. Aucune fiche de synthèse ni réponse préparée n'est indexée.

| PDF d’origine | Pages du fichier | Passages indexés |
|---|---:|---:|
| [DLSS Super Resolution – Programming Guide](https://github.com/NVIDIA/DLSS/blob/9981c6350dc58c0ce21f024fd65e31e851c31689/doc/DLSS_Programming_Guide_Release.pdf) | 84 | 301 |
| [DLSS Ray Reconstruction – Integration Guide](https://github.com/NVIDIA/DLSS/blob/9981c6350dc58c0ce21f024fd65e31e851c31689/doc/DLSS-RR%20Integration%20Guide.pdf) | 38 | 112 |
| [RTX UI Developer Guidelines](https://github.com/NVIDIA/DLSS/blob/9981c6350dc58c0ce21f024fd65e31e851c31689/doc/RTX%20UI%20Developer%20Guidelines.pdf) | 15 | 38 |

Les originaux ont été conservés dans `demo/dlss/originaux/` sur la machine de test. Le guide de programmation dans Téléchargements est identique à la copie officielle (même empreinte SHA-256). Le test a ingéré le fichier local une seule fois. Le ZIP du projet contient le script pour récupérer les PDF à l’identique, pas une nouvelle distribution de ces PDF.

## Rejouer l'essai

Depuis la racine du projet, lancer `docker compose up -d --build`. Les commandes ci-dessous utilisent l’environnement Python du projet (voir le [README](../../README.md) pour l’installation). La première vérifie les empreintes, télécharge les fichiers s’ils manquent et indexe les PDF ; la deuxième teste la recherche sans générer de réponse ; la troisième interroge effectivement le modèle configuré.

```sh
.venv/bin/python scripts/dlss.py --index-only
.venv/bin/python scripts/dlss.py --search-only
.venv/bin/python scripts/dlss.py --search-only --prod --rapport /tmp/resultats-dlss.json
```

Pour utiliser le fichier personnel plutôt que sa copie publique lors de l’indexation :

```sh
.venv/bin/python scripts/dlss.py --index-only --guide-local "$HOME/Downloads/DLSS_Programming_Guide_Release.pdf"
```

Pour poser d'autres questions sans réindexer :

```sh
python3 scripts/ask.py "En mode DLAA, quel est le rapport entre la résolution d’entrée et la résolution de sortie ?"
python3 scripts/ask.py --source "DLSS-RR Integration Guide.pdf" "Quels sont les deux formats de texture acceptés pour les Motion Vectors ?"
```

Ajouter `--test` à `ask.py` affiche les textes retrouvés, leur score, leur source et le prompt. Dans [/docs](http://localhost:8000/docs), on peut poser la même question à `POST /search`. Pour limiter les résultats aux trois guides, utiliser `"filters":{"metadata.corpus":"nvidia-dlss-9981c635"}`. La recherche dans le script applique uniquement ce filtre : elle ne connaît pas la bonne page ou le bon fichier d’avance.

## Comment lire le résultat

Chaque test distingue trois contrôles : présence de la preuve dans le prompt, contenu de la réponse et citation d’au moins un passage à l’appui. Les motifs du script sont des contrôles ciblés, pas un évaluateur général de vérité : relire la réponse et la page reste utile. Les étiquettes `S1`, `S2`, etc. sont propres à chaque question. Les numéros de page commencent à 1 pour la première page du PDF, ce qui peut différer du numéro imprimé.

Les résultats réellement obtenus, y compris le premier essai incorrect sur le mode 6X, figurent dans le [compte rendu](../../VALIDATION.md#essai-sur-les-documents-nvidia-dlss).

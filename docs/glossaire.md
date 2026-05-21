<!--
TEMPLATE — Glossaire et conventions
====================================
Public cible : (1) une IA qui doit nommer correctement un nouveau type / une nouvelle
table / une nouvelle classe sans réinventer une convention, (2) un humain qui croise
un acronyme et veut savoir ce qu'il signifie.

C'est la SOURCE DE VÉRITÉ du vocabulaire. Toute convention de nommage, tout terme métier,
tout acronyme spécifique au projet doit s'y rattacher. Si un terme manque ici, c'est qu'il
ne devrait pas exister dans le code.

Garde-fous :
- Distinguer "validé" / "(à confirmer)" / "(legacy — à éviter)".
- Tout terme cite au moins un endroit où il apparaît dans le code, la base, ou la doc
  fonctionnelle. Sinon, c'est un terme qui n'a pas sa place ici.
- Le glossaire référence ; il ne RE-DÉFINIT PAS les concepts qui sont décrits ailleurs.

Bloc « Mode d'emploi » en fin de fichier.
-->

# Glossaire et conventions de nommage — RAG

| Champ | Valeur |
|---|---|
| **Dernière mise à jour** | 2026-05-21 |
| **Mise à jour par** | agent IA (doc-patcher) |
| **PR de référence** | 6d1386c |
| **Périmètre** | rag_engine, gemini_client, pdf_loader, text_processor, main |

> Source de vérité du vocabulaire projet. Termes marqués **(à confirmer)** = déductions à valider. Marqués **(legacy)** = conservés pour compatibilité, à ne pas reproduire.

---

## 1. Vocabulaire métier

### 1.1 Concepts centraux

| Terme | Définition | Représentation code | Apparaît dans |
|---|---|---|---|
| **RAG (Retrieval-Augmented Generation)** | Pipeline qui enrichit une requête utilisateur en récupérant d'abord des passages pertinents d'un corpus, puis en les transmettant à un LLM pour générer une réponse contextualisée. | `RAGEngine` | `rag_engine.py`, `main.py` |
| **Index vectoriel** | Représentation du corpus sous forme de vecteurs d'embeddings permettant la recherche par similarité sémantique. Persisté sur disque par URL hash. | `VectorStoreIndex` | `rag_engine.py` |
| **Embedding** | Représentation numérique dense d'un texte produite par un modèle de sentence-transformer. Utilisé pour la recherche sémantique. | `HuggingFaceEmbedding` (modèle `BAAI/bge-small-en-v1.5`) | `text_processor.py` |
| **Chunk / nœud** | Fragment de texte issu du découpage d'un document PDF, de taille fixe (512 tokens, overlap 50). Unité de base de l'index. | `SentenceSplitter` | `text_processor.py` |
| **Cache d'index** | Répertoire `./storage/index_<md5>` où l'index vectoriel est persisté pour éviter un re-calcul des embeddings à chaque démarrage. | `_get_index_path()` | `rag_engine.py` |
| **Reranking** | Étape optionnelle du pipeline RAG qui réordonne les passages récupérés par l'index vectoriel selon leur pertinence effective vis-à-vis de la requête, en interrogeant le LLM. Améliore la précision de la réponse finale. | `rerank_passages()` | `gemini_client.py` |

### 1.2 Règles de gestion (noms courts)

> Non applicable — le système RAG n'a pas de règles métier nommées au sens workflow/processus.

### 1.3 États et transitions nommés

> Non applicable — pas de machine à états dans ce projet.

---

## 2. Acronymes et abréviations

### 2.1 Acronymes métier

| Acronyme | Signification | Confiance | Origine |
|---|---|---|---|
| **RAG** | Retrieval-Augmented Generation | ✓ | README.md, rag_engine.py |
| **LLM** | Large Language Model | ✓ | README.md, gemini_client.py |
| **PDF** | Portable Document Format — format source des documents indexés | ✓ | pdf_loader.py |

### 2.2 Acronymes techniques

| Acronyme | Signification | Domaine |
|---|---|---|
| **CLI** | Command-Line Interface — interface interactive en ligne de commande | main.py |
| **API** | Application Programming Interface — ici l'API Google Gemini (`GEMINI_API_KEY`) | gemini_client.py |
| **BGE** | BAAI General Embedding — famille de modèles d'embeddings de BAAI | text_processor.py |
| **BAAI** | Beijing Academy of Artificial Intelligence — producteur du modèle `BAAI/bge-small-en-v1.5` | text_processor.py |

---

## 3. Conventions de nommage — Code

### 3.1 Packages / modules

```
<responsabilité>.py  (flat layout à la racine — pas de sous-packages)
```

| Module | Responsabilité |
|---|---|
| `main.py` | Point d'entrée — boucle interactive Q&A |
| `rag_engine.py` | Orchestration RAG — indexation, cache, requête |
| `pdf_loader.py` | Téléchargement et extraction de PDFs |
| `text_processor.py` | Configuration embeddings et découpage en chunks |
| `gemini_client.py` | Initialisation du LLM Gemini |

### 3.2 Classes

- **Casse** : `PascalCase` (Python)
- **Suffixes par rôle** :

| Rôle | Suffixe | Exemple |
|---|---|---|
| Moteur / orchestrateur RAG | `Engine` | `RAGEngine` |

### 3.3 Méthodes et fonctions

- **Casse** : `snake_case` (Python)
- **Verbes d'action** : `load_*`, `create_*`, `setup_*`, `initialize_*`, `query`
- **Méthodes privées** : préfixe `_` — ex. `_get_index_path`, `_save_index`, `_load_index`
- **Booléens** : préfixe `is_` / `has_` / `can_` ; jamais de double négation

### 3.4 Variables et constantes

- **Variables locales** : `snake_case` — courtes mais explicites, pas d'abréviations cryptiques
- **Constantes d'environnement** : `UPPER_SNAKE_CASE` — ex. `GEMINI_API_KEY`
- **Identifiants** : conserver le terme métier (ex. `pdf_url`, `embed_model`, `query_engine`)

### 3.5 Tests

| Type | Répertoire | Pattern de fichier |
|---|---|---|
| Unitaire | `tests/` | `test_*.py` |

Configuration dans `pyproject.toml` `[tool.pytest.ini_options]`.

### 3.6 Fichiers, branches, commits

- **Fichiers Python** : `snake_case.py`
- **Branches** : `feat/...`, `fix/...`, `chore/...` (à confirmer)
- **Commits** : Conventional Commits — `feat(scope): description` (à confirmer)

---

## 4. Conventions de nommage — Données

> Non applicable — ce projet n'utilise pas de base de données relationnelle ni de collections nommées. Le stockage est un répertoire de fichiers vectoriels sur disque (`./storage/index_<md5hash>`).

---

## 5. Conventions de nommage — Interfaces

> Non applicable — ce projet n'expose pas d'API HTTP et n'utilise pas de topics/queues. L'interface unique est le CLI interactif (`main.py`).

---

## 6. Vocabulaire technique

### 6.1 Stack

| Terme | Définition | Lien |
|---|---|---|
| **LlamaIndex** | Framework RAG Python — orchestre chargement, indexation, retrieval et génération | [llama_index.core](https://docs.llamaindex.ai) |
| **Google Gemini** | LLM utilisé pour la génération de réponses — modèle `gemini-2.5-flash` | [llama_index.llms.gemini](https://docs.llamaindex.ai/en/stable/examples/llm/gemini/) |
| **HuggingFace Embeddings** | Modèle d'embeddings `BAAI/bge-small-en-v1.5` via `llama_index.embeddings.huggingface` | [HuggingFaceEmbedding](https://docs.llamaindex.ai) |
| **PDFReader** | Chargeur de fichiers PDF (`llama_index.readers.file`) | `pdf_loader.py` |
| **SentenceSplitter** | Découpeur de texte en chunks de 512 tokens avec overlap 50 | `text_processor.py` |
| **VectorStoreIndex** | Index vectoriel LlamaIndex persisté sur disque | `rag_engine.py` |

### 6.2 Patterns

| Pattern | Sens dans ce projet | Où il est appliqué |
|---|---|---|
| **RAG (Retrieval-Augmented Generation)** | Enrichissement du prompt par retrieval sémantique avant génération LLM | `rag_engine.py` — `RAGEngine.query()` |
| **Cache-on-disk by URL hash** | L'index vectoriel est persisté sous `./storage/index_<md5(url)>` ; les runs suivants le rechargent sans re-calculer les embeddings | `rag_engine.py` — `_get_index_path()`, `_save_index()`, `_load_index()` |

### 6.3 Démarche / process

| Terme | Définition |
|---|---|
| **Conventional Commits** | Format de message de commit — `feat(scope): description` (à confirmer) |
| **Ruff** | Linter et formateur Python configuré dans `pyproject.toml` |
| **Pyright** | Type-checker Python (mode `standard`, Python 3.12) |

---

## 7. Termes à éviter / pièges de vocabulaire

> Termes ambigus, faux amis, ou collisions avec des termes techniques. Cette section évite les confusions répétitives.

| Terme | Pourquoi l'éviter | Préférer |
|---|---|---|
| `iNitialize_gemini_llm` (legacy) | Casse incorrecte (mélange majuscule/minuscule) — présent dans l'historique git avant commit 6d1386c | `initialize_gemini_llm` |
| `GOOGLE_API_KEY` (legacy) | Ancienne variable d'environnement référencée dans `config.py` avant la correction — remplacée par `GEMINI_API_KEY` via `os.environ` | `GEMINI_API_KEY` |

---

## 9. Questions ouvertes

> Ambiguïtés à lever avec les experts. Une question marquée non résolue depuis longtemps est un signal — voir [index.md §5](index.md#5-signaux-de-dérive-à-surveiller).

| # | Question | Impact | Owner | Ouverte depuis |
|---|---|---|---|---|
| Q1 | Le README mentionne `config.py` avec `GOOGLE_API_KEY` mais le code actuel lit `os.environ["GEMINI_API_KEY"]` — le README est-il à jour ? | Cosmétique | (à confirmer) | 2026-05-21 |

---

## Références

- Architecture (où ces termes sont utilisés en pratique) : [architecture.md](architecture.md)
- Modèle de données (conventions colonnes détaillées) : [data-model.md](data-model.md)
- Contrats (conventions interfaces détaillées) : [contracts.md](contracts.md)
- Fonctionnel (règles métier nommées) : [fonctionnel.md](fonctionnel.md)
- ADR : (à confirmer — pas de dossier ADR détecté dans le dépôt)

---

<!--
MODE D'EMPLOI DU TEMPLATE
=========================

POUR L'IA QUI MET À JOUR CE FICHIER

Déclencheurs :

| Modification dans la PR | Sections à toucher |
|---|---|
| Nouveau concept métier introduit dans le code | §1.1 |
| Nouvelle règle de gestion nommée | §1.2 (avec lien vers fonctionnel.md) |
| Nouvel état dans une machine à états | §1.3 |
| Nouvel acronyme dans des noms de classe / table / variable | §2 |
| Nouvelle convention de package / classe / méthode | §3 |
| Nouvelle convention de table / colonne | §4 (et data-model.md §9) |
| Nouvelle convention d'endpoint / topic | §5 |
| Nouveau pattern adopté | §6.2 |
| Renommage / dépréciation d'un terme | §7 (« legacy ») |

Règles spéciales :
- Quand on RENOMME un terme, ne pas supprimer l'ancien : le déplacer en §7 « legacy »
  avec le pointeur vers le nouveau, pendant au moins 1 cycle de release.
- Une convention ne s'AJOUTE qu'avec un exemple vivant dans le code (lien direct).
- Les questions ouvertes §9 vieillissent — celles sans MAJ depuis 90 jours doivent être
  signalées dans la PR.

Auto-checks :
- [ ] Chaque concept §1.1 cite au moins une représentation code réelle.
- [ ] Aucun acronyme §2 marqué ✓ ne reste sans occurrence dans le code.
- [ ] Les liens §Références sont valides.
- [ ] Section §7 ne déprécie aucun terme encore activement utilisé.

POUR LE RELECTEUR HUMAIN

- Le glossaire vieillit mal si on n'élague pas : terme inutilisé → le retirer (ou le passer
  en legacy si renommé).
- Les § « (à confirmer) » doivent être levés ou explicitement assumés.
- Vérifier la cohérence avec data-model §9 et contracts §1 — pas de doublon, pas d'écart.

POUR ADAPTER À UN AUTRE PROJET

1. Le glossaire est le document le plus DÉPENDANT du domaine — repartir de zéro pour §1.
2. §3 et §4 sont les plus stables — les conventions Java / SQL standard se répliquent.
3. Si le projet a un seul domaine simple, fusionner §1.1 et §1.2.
4. Pour un projet multi-langage, dédoubler §3 par langage (§3.A Java, §3.B Python, etc.).
-->

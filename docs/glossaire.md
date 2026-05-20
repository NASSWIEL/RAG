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
| **Dernière mise à jour** | 2026-05-20 |
| **Mise à jour par** | agent IA (doc-sync) |
| **PR de référence** | (à confirmer) |
| **Périmètre** | Pipeline RAG — chargement PDF, découpage, embeddings, requête LLM |

> Source de vérité du vocabulaire projet. Termes marqués **(à confirmer)** = déductions à valider. Marqués **(legacy)** = conservés pour compatibilité, à ne pas reproduire.

---

## 1. Vocabulaire métier

### 1.1 Concepts centraux

| Terme | Définition | Représentation code | Apparaît dans |
|---|---|---|---|
| **RAG** | Pipeline qui récupère des passages pertinents dans un corpus documentaire (retrieval) avant de générer une réponse via un LLM (generation) | `RAGEngine` | `rag_engine.py`, `main.py` |
| **Vector Index** | Structure qui stocke les embeddings des chunks de texte et permet la recherche sémantique par similarité | `VectorStoreIndex` | `rag_engine.py` |
| **Embedding** | Représentation vectorielle dense d'un texte, utilisée pour mesurer la similarité sémantique | `HuggingFaceEmbedding` | `text_processor.py`, `rag_engine.py` |
| **Chunk** | Fragment de texte extrait d'un document PDF après découpage, taille fixée à 512 tokens avec chevauchement de 50 | paramètres `chunk_size`, `chunk_overlap` | `text_processor.py` |
| **Node** | Unité atomique indexée dans LlamaIndex — correspond à un chunk enrichi de métadonnées | `SentenceSplitter`, `Settings.node_parser` | `text_processor.py`, `rag_engine.py` |
| **Query Engine** | Composant qui orchestre la recherche sémantique et la génération de réponse pour une requête utilisateur | `self.query_engine` (`similarity_top_k=3`, `response_mode="compact"`) | `rag_engine.py` |

### 1.2 Règles de gestion (noms courts)

> Liste des règles métier nommées qui apparaissent dans le code, les commits, les tickets. Le détail vit dans [fonctionnel.md](fonctionnel.md).

| Nom court | Sens | Détail dans |
|---|---|---|
| _Aucune règle de gestion nommée identifiée (à confirmer)_ | — | — |

### 1.3 États et transitions nommés

> Si le métier a des états explicites (statuts d'une commande, phases d'un workflow), les nommer ici.

| État | Sens | Transitions sortantes |
|---|---|---|
| _Sans objet — pas de machine à états explicite dans ce projet (à confirmer)_ | — | — |

---

## 2. Acronymes et abréviations

### 2.1 Acronymes métier

| Acronyme | Signification | Confiance | Origine |
|---|---|---|---|
| **RAG** | Retrieval-Augmented Generation | ✓ | README.md |
| **LLM** | Large Language Model | ✓ | README.md |
| **PDF** | Portable Document Format | ✓ | `pdf_loader.py` |

### 2.2 Acronymes techniques

| Acronyme | Signification | Domaine |
|---|---|---|
| **BGE** | BAAI General Embedding (modèle `BAAI/bge-small-en-v1.5`) | Embeddings / HuggingFace — `text_processor.py` |
| **MD5** | Message Digest 5 — utilisé pour hasher l'URL du PDF afin de nommer le répertoire de cache | Stdlib Python — `rag_engine.py` |
| **CLI** | Command-Line Interface — interface interactive en ligne de commande | `main.py` |

---

## 3. Conventions de nommage — Code

### 3.1 Packages / modules

_Layout plat — pas de sous-packages. Modules à la racine du projet :_

| Module | Rôle |
|---|---|
| `rag_engine.py` | Orchestrateur principal (index, cache, requête) |
| `pdf_loader.py` | Téléchargement et extraction PDF |
| `text_processor.py` | Découpage et configuration des embeddings |
| `gemini_client.py` | Initialisation du LLM Gemini |
| `main.py` | Point d'entrée CLI |
| `config.py` | Clé API Google (à ne pas versionner) |

### 3.2 Classes

- **Casse** : `PascalCase` (Python)
- **Suffixes par rôle** :

| Rôle | Suffixe | Exemple |
|---|---|---|
| Moteur / orchestrateur | `Engine` | `RAGEngine` (`rag_engine.py`) |
| _Autres rôles non représentés dans ce projet (à confirmer)_ | — | — |

### 3.3 Méthodes et fonctions

- **Casse** : `snake_case` (Python)
- **Verbes d'action** : `load_*`, `setup_*`, `create_*`, `initialize_*` — observés dans `pdf_loader.py`, `text_processor.py`, `gemini_client.py`
- **Booléens** : préfixe `is_` / `has_` / `can_` ; jamais de double négation
- **Async** : sans objet — pipeline synchrone

### 3.4 Variables et constantes

- **Variables locales** : `snake_case` — ex. `pdf_url`, `url_hash`, `index_path`
- **Constantes** : `UPPER_SNAKE_CASE` — ex. `GOOGLE_API_KEY` (`config.py`)
- **Énumérations** : sans objet — pas d'enum dans ce projet
- **Identifiants** : conserver le terme métier (ex. `pdf_url` plutôt que `url`, `embed_model` plutôt que `model`)

### 3.5 Tests

| Type | Pattern de fichier | Pattern de fonction |
|---|---|---|
| Unitaire | `test_*.py` dans `tests/` | `test_*` | 
| _Intégration / E2E non configurés (à confirmer)_ | — | — |

### 3.6 Fichiers, branches, commits

- **Fichiers** : `snake_case.py` (ex. `rag_engine.py`, `pdf_loader.py`) ; docs en `kebab-case.md`
- **Branches** : `feat/...`, `fix/...`, `chore/...` (à confirmer — convention non explicite dans le repo)
- **Commits** : Conventional Commits — `feat(scope): description` (gitlint configuré dans `pyproject.toml`)

---

## 4. Conventions de nommage — Données

> Sync avec [data-model.md §9.1](data-model.md). Lister ici uniquement les conventions DE NOMMAGE ; les détails de typage vivent côté data-model.

_Sans objet — ce projet ne comporte pas de base de données relationnelle ni de schéma de colonnes. La persistance se limite à un répertoire de fichiers (`./storage/`) nommé par hash MD5 de l'URL du PDF (`rag_engine.py:_get_index_path`)._

---

## 5. Conventions de nommage — Interfaces

### 5.1 Endpoints HTTP

_Sans objet — application CLI sans serveur HTTP._

### 5.2 Topics / queues

_Sans objet — pas de messaging asynchrone dans ce projet._

### 5.3 Schémas d'événements

_Sans objet — pas d'événements publiés/consommés dans ce projet._

---

## 6. Vocabulaire technique

### 6.1 Stack

| Terme | Définition | Lien |
|---|---|---|
| **LlamaIndex** | Framework d'orchestration RAG : gère l'indexation, le stockage et la requête de documents | [llama-index](https://docs.llamaindex.ai) |
| **Gemini** | LLM de Google utilisé pour générer les réponses (`models/gemini-2.5-flash`) | `gemini_client.py` |
| **HuggingFaceEmbedding** | Modèle d'embeddings `BAAI/bge-small-en-v1.5` — transforme les chunks en vecteurs | `text_processor.py` |
| **PDFReader** | Lecteur de fichiers PDF fourni par `llama-index-readers-file` | `pdf_loader.py` |
| **SentenceSplitter** | Découpeur de texte basé sur les phrases, configuré à 512 tokens / 50 de chevauchement | `text_processor.py` |
| **requests** | Bibliothèque HTTP Python pour télécharger le PDF depuis une URL (timeout 30 s) | `pdf_loader.py` |

### 6.2 Patterns

| Pattern | Sens dans ce projet | Où il est appliqué |
|---|---|---|
| **RAG** | Retrieval-Augmented Generation — enrichit le prompt LLM avec des passages récupérés par similarité vectorielle | `rag_engine.py` (pipeline complet) |
| **Cache sur disque par hash** | Les embeddings générés sont persistés sous `./storage/index_<md5_url>/` pour éviter le re-calcul | `rag_engine.py:_get_index_path`, `_save_index`, `_load_index` |

### 6.3 Démarche / process

| Terme | Définition |
|---|---|
| **ruff** | Linter et formateur Python configuré dans `pyproject.toml` (ligne max 100, cible Python 3.12, docstyle Google) |
| **pyright** | Vérificateur de types statique en mode `standard`, cible `src/` et `tests/` |
| **bandit** | Analyseur de sécurité Python — skip B104 (bind 0.0.0.0) intentionnel |
| **pytest** | Framework de tests, répertoire `tests/`, fichiers `test_*.py`, fonctions `test_*` |
| **gitlint** | Vérification du format des messages de commit (Conventional Commits) |
| **ADR** | Architecture Decision Record — décisions traçables, répertoire (à confirmer) |

---

## 7. Termes à éviter / pièges de vocabulaire

> Termes ambigus, faux amis, ou collisions avec des termes techniques. Cette section évite les confusions répétitives.

| Terme | Pourquoi l'éviter | Préférer |
|---|---|---|
| `iNitialize_gemini_llm` (legacy) | Casse incohérente (`iN`) — nom défini dans `gemini_client.py`, mais importé sous `initialize_gemini_llm` dans `rag_engine.py` ⚠️ | `initialize_gemini_llm` (à corriger) |

---

## 8. Correspondance / mapping

> À utiliser si le projet a deux vocabulaires en parallèle (ex. ancien système → nouveau, métier → technique, API publique → modèle interne). Sinon, supprimer la section.

_Sans objet — vocabulaire unique dans ce projet._

---

## 9. Questions ouvertes

> Ambiguïtés à lever avec les experts. Une question marquée non résolue depuis longtemps est un signal — voir [index.md §5](index.md#5-signaux-de-dérive-à-surveiller).

| # | Question | Impact | Owner | Ouverte depuis |
|---|---|---|---|---|
| Q1 | Le nom de fonction `iNitialize_gemini_llm` (casse incorrecte) diverge de l'import `initialize_gemini_llm` dans `rag_engine.py` — quelle est la version officielle ? | Bloquant (risque d'ImportError) | (à confirmer) | 2026-05-20 |

---

## Références

- Architecture (où ces termes sont utilisés en pratique) : [architecture.md](architecture.md)
- Modèle de données (conventions colonnes détaillées) : [data-model.md](data-model.md)
- Contrats (conventions interfaces détaillées) : [contracts.md](contracts.md)
- Fonctionnel (règles métier nommées) : [fonctionnel.md](fonctionnel.md)
- ADR : (à confirmer — répertoire ADR non identifié dans le repo)

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

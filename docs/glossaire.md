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
| **Dernière mise à jour** | 2026-05-11 |
| **Mise à jour par** | agent IA (doc-patcher) |
| **PR de référence** | fcd7a06 |
| **Périmètre** | Pipeline RAG — indexation PDF, embeddings, requêtage LLM |

> Source de vérité du vocabulaire projet. Termes marqués **(à confirmer)** = déductions à valider. Marqués **(legacy)** = conservés pour compatibilité, à ne pas reproduire.

---

## 1. Vocabulaire métier

### 1.1 Concepts centraux

| Terme | Définition | Représentation code | Apparaît dans |
|---|---|---|---|
| **Document** | Unité de contenu extraite d'un PDF ; correspond à une page ou section indexée avant découpage en chunks. | `documents` (liste retournée par `load_documents_from_pdf`) | `pdf_loader.py`, `rag_engine.py` |
| **Chunk** | Fragment de texte issu du découpage d'un document, taille fixée à 512 tokens avec chevauchement de 50 tokens. Unité de base de l'indexation vectorielle. | `SentenceSplitter(chunk_size=512, chunk_overlap=50)` | `text_processor.py` |
| **Embedding** | Représentation vectorielle dense d'un chunk, produite par le modèle HuggingFace `BAAI/bge-small-en-v1.5`. Permet la recherche sémantique. | `HuggingFaceEmbedding` / `Settings.embed_model` | `text_processor.py`, `rag_engine.py` |
| **Index** | Structure vectorielle (`VectorStoreIndex`) contenant l'ensemble des embeddings d'un PDF donné. Persisté sur disque sous `./storage/index_<url_hash>/`. | `VectorStoreIndex` / `self.index` | `rag_engine.py` |
| **Query Engine** | Composant LlamaIndex qui orchestre la recherche sémantique (top-k=3) puis la génération de réponse via Gemini. | `self.query_engine` (`as_query_engine`) | `rag_engine.py` |
| **Cache d'index** | Mécanisme de persistance : si un index existe déjà pour un URL donné (identifié par son hash MD5), il est rechargé sans re-calcul des embeddings. | `_get_index_path` / `_save_index` / `_load_index` | `rag_engine.py` |

### 1.2 Règles de gestion (noms courts)

> Liste des règles métier nommées qui apparaissent dans le code, les commits, les tickets. Le détail vit dans [fonctionnel.md](fonctionnel.md).

| Nom court | Sens | Détail dans |
|---|---|---|
| **Cache-par-URL** | Un index n'est recalculé que si aucun répertoire `index_<md5(url)>` n'existe dans `storage_dir`. | [fonctionnel.md](fonctionnel.md) |
| **Top-K retrieval** | Lors d'une requête, les 3 chunks les plus proches sémantiquement sont sélectionnés (`similarity_top_k=3`) et passés à Gemini. | [fonctionnel.md](fonctionnel.md) |

### 1.3 États et transitions nommés

> Phases du pipeline RAG lors de l'initialisation du moteur.

| État | Sens | Transitions sortantes |
|---|---|---|
| `COLD_START` | Premier lancement pour un URL donné — aucun cache présent. | → `DOWNLOADING` |
| `DOWNLOADING` | Téléchargement du PDF via `load_pdf_from_url`. | → `INDEXING` |
| `INDEXING` | Découpage, embedding et construction du `VectorStoreIndex`. | → `PERSISTED` |
| `PERSISTED` | Index sauvegardé sur disque dans `./storage/index_<hash>/`. | → `READY` |
| `WARM_START` | Cache détecté — chargement direct depuis le disque via `_load_index`. | → `READY` |
| `READY` | `query_engine` disponible, le moteur répond aux requêtes. | — |

---

## 2. Acronymes et abréviations

### 2.1 Acronymes métier

| Acronyme | Signification | Confiance | Origine |
|---|---|---|---|
| **RAG** | Retrieval-Augmented Generation — technique combinant recherche sémantique et génération par LLM pour répondre à des questions sur un corpus documentaire. | ✓ | `README.md`, nom du projet (`pyproject.toml`) |

### 2.2 Acronymes techniques

| Acronyme | Signification | Domaine |
|---|---|---|
| **LLM** | Large Language Model — modèle de langage de grande taille. Ici : Google Gemini (`models/gemini-2.5-flash`). | IA / NLP |
| **NLP** | Natural Language Processing — traitement automatique du langage naturel. | IA |
| **PDF** | Portable Document Format — format source des documents indexés. | Format fichier |
| **CLI** | Command-Line Interface — interface d'entrée du système (`main.py`). | Interface |
| **BGE** | BAAI General Embedding — famille de modèles d'embeddings de l'Institut d'IA de Pékin (BAAI). Modèle utilisé : `BAAI/bge-small-en-v1.5`. | Embeddings / HuggingFace |
| **CQRS** | (à confirmer) — non appliqué explicitement dans ce projet. | Pattern (non utilisé) |

---

## 3. Conventions de nommage — Code

### 3.1 Packages / modules

```
<module>.py  (flat layout à la racine du projet — pas de sous-packages)
```

| Module | Rôle |
|---|---|
| `rag_engine.py` | Orchestrateur principal — indexation et requêtage |
| `pdf_loader.py` | Téléchargement et extraction de contenu PDF |
| `text_processor.py` | Configuration embeddings et découpage en chunks |
| `gemini_client.py` | Initialisation du LLM Gemini |
| `main.py` | Point d'entrée CLI interactif |
| `config.py` | Clé API Google (à confirmer — non versionné) |

### 3.2 Classes

- **Casse** : `PascalCase` (Python, conformément à PEP 8 / Ruff règle N)
- **Suffixes par rôle** :

| Rôle | Suffixe | Exemple |
|---|---|---|
| Moteur / orchestrateur | `Engine` | `RAGEngine` |
| Exception | `Exception` ou `Error` | (à confirmer) |

### 3.3 Méthodes et fonctions

- **Casse** : `snake_case` (Python)
- **Verbes d'action** : `load_*`, `create_*`, `setup_*`, `initialize_*`, `query`
- **Méthodes privées** : préfixe `_` (ex. `_get_index_path`, `_save_index`, `_load_index`)
- **Booléens** : préfixe `is_` / `has_` ; jamais de double négation
- **Async** : non utilisé dans ce projet (synchrone)

### 3.4 Variables et constantes

- **Variables locales** : `snake_case` — courtes mais explicites, pas d'abréviations cryptiques
- **Constantes** : `UPPER_SNAKE_CASE` (ex. `GOOGLE_API_KEY` dans `config.py`)
- **Énumérations** : non utilisées actuellement dans ce projet
- **Identifiants** : conserver le terme métier (ex. `pdf_url`, `embed_model`, `query_engine`)

### 3.5 Tests

| Type | Pattern de classe | Pattern de méthode |
|---|---|---|
| Unitaire | `<ClasseTestée>Test` | `should<Comportement>When<Condition>` |
| Intégration | `<ClasseTestée>IT` ou `*IntegrationTest` | idem |
| Contract | `<Interface>ContractTest` | idem |
| End-to-end | `<Parcours>E2ETest` | `<scénario fonctionnel>` |

### 3.6 Fichiers, branches, commits

- **Fichiers Python** : `snake_case.py` (ex. `rag_engine.py`, `pdf_loader.py`)
- **Fichiers doc** : `kebab-case.md` (ex. `glossaire.md`, `data-model.md`)
- **Branches** : `feat/...`, `fix/...`, `chore/...` (à confirmer — convention standard)
- **Commits** : Conventional Commits — `feat(scope): description` (gitlint-core configuré comme dépendance dev)

---

## 4. Conventions de nommage — Données

> Sync avec [data-model.md §9.1](data-model.md). Lister ici uniquement les conventions DE NOMMAGE ; les détails de typage vivent côté data-model.

### 4.1 Préfixes de colonnes / champs

| Préfixe | Sémantique | Exemple |
|---|---|---|
| `id_` | Identifiant technique | `id_order` |
| `ref_` | Référence métier alphanum | `ref_customer` |
| `cd_` | Code discret | `cd_status` |
| `dt_` | Date | `dt_created` |
| `is_` / `has_` | Booléen | `is_active` |
| `nb_` / `qty_` | Quantité | `nb_items` |
| `lib_` / `label_` | Libellé | `label_status` |

### 4.2 Suffixes temporels

| Suffixe | Sens | Exemple |
|---|---|---|
| `*_at` | Instant ponctuel (timestamp) | `created_at` |
| `*_on` | Date sans heure | `effective_on` |
| `*_deb` / `*_fin` | Bornes de période | `dt_deb_validite` |

### 4.3 Tables / collections

> Ce projet n'utilise pas de base de données relationnelle. Le stockage est un système de fichiers plat sous `./storage/`. Les conventions ci-dessous s'appliquent si une base est ajoutée ultérieurement.

- **Casse** : `snake_case` (recommandation pour cohérence avec le reste du code Python)
- **Singulier ou pluriel** : (à confirmer)
- **Tables de liaison N:N** : `<a>_<b>` ordonnés alphabétiquement
- **Tables d'audit** : Suffixe `_audit` ou `_history`

---

## 5. Conventions de nommage — Interfaces

### 5.1 Endpoints HTTP

> Ce projet n'expose pas d'API HTTP dans son état actuel. L'interface est une CLI interactive (`main.py`). Les conventions ci-dessous s'appliquent si une API est ajoutée ultérieurement.

- **Casse** : `kebab-case` dans les paths
- **Pluriel pour les collections** : `/documents`, `/queries`
- **Verbes** : à éviter dans les paths sauf actions hors CRUD
- **Versioning** : `/v1/...` (à confirmer)

### 5.2 Topics / queues

> Ce projet n'utilise pas de message broker dans son état actuel. Section à compléter si un système de queue est introduit.

```
<env>.<domaine>.<entité>.<événement>  (à confirmer)
```

| Segment | Valeurs |
|---|---|
| `<env>` | `dev`, `stg`, `prod` |
| `<domaine>` | `rag` |
| `<entité>` | `document`, `query` |
| `<événement>` | `indexed`, `queried` (à confirmer) |

### 5.3 Schémas d'événements

- **Champs systématiques** : `id`, `event_type`, `schema_version`, `occurred_at`, `producer`
- **Casse champs** : `snake_case` (cohérence avec Python)
- **Versioning** : champ `schema_version` (entier monotone)

---

## 6. Vocabulaire technique

### 6.1 Stack

| Terme | Définition | Lien |
|---|---|---|
| **LlamaIndex** | Framework d'orchestration RAG — fournit `VectorStoreIndex`, `SentenceSplitter`, `Settings`, `StorageContext`. | [llamaindex.ai](https://www.llamaindex.ai) |
| **HuggingFace Embeddings** | Bibliothèque d'embeddings — modèle `BAAI/bge-small-en-v1.5` utilisé pour vectoriser les chunks. | `llama-index-embeddings-huggingface` |
| **Google Gemini** | LLM de génération de réponses — modèle `models/gemini-2.5-flash`, température 0.1. | `llama-index-llms-gemini` |
| **PDFReader** | Lecteur PDF fourni par `llama-index-readers-file` ; charge les pages d'un PDF en objets `Document`. | `llama-index-readers-file` |
| **Ruff** | Linter et formateur Python — lint + format, longueur de ligne 100, convention docstring Google. | `pyproject.toml` |
| **Pyright** | Vérificateur de types statique Python (`typeCheckingMode = "standard"`). | `pyproject.toml` |
| **Bandit** | Analyseur de sécurité statique Python. | `pyproject.toml` |

### 6.2 Patterns

| Pattern | Sens dans ce projet | Où il est appliqué |
|---|---|---|
| **Pipeline RAG** | Chaîne linéaire : PDF → chunks → embeddings → index → retrieval → génération LLM. | Tout le projet |
| **Cache-on-disk** | L'index vectoriel est persisté sur disque et rechargé à chaud si disponible (identifié par hash MD5 de l'URL). | `rag_engine.py` |
| **Façade** | `RAGEngine` encapsule tous les sous-composants (loader, embedder, LLM, parser) derrière une interface `__init__` + `query`. | `rag_engine.py` |

### 6.3 Démarche / process

| Terme | Définition |
|---|---|
| **Trunk-based** | (à confirmer) — branche principale `main`, commits récents directs. |
| **Conventional Commits** | Format de message de commit imposé par `gitlint-core` (dépendance dev). |
| **ADR** | Architecture Decision Record — décisions traçables. Aucun dossier ADR identifié dans ce dépôt à ce jour (à confirmer). |

---

## 7. Termes à éviter / pièges de vocabulaire

> Termes ambigus, faux amis, ou collisions avec des termes techniques. Cette section évite les confusions répétitives.

| Terme | Pourquoi l'éviter | Préférer |
|---|---|---|
| « base de données » | Ambigu — ce projet ne possède pas de BDD ; le stockage est un répertoire de fichiers sérialisés par LlamaIndex. | « index » ou « storage » |
| « modèle » seul | Ambigu entre le modèle d'embedding (`embed_model`) et le LLM (`llm`). | « embedding model » ou « LLM » selon le contexte |

---

## 8. Correspondance / mapping

> À utiliser si le projet a deux vocabulaires en parallèle (ex. ancien système → nouveau, métier → technique, API publique → modèle interne). Sinon, supprimer la section.

| Terme README | Terme code | Note |
|---|---|---|
| « embeddings » | `embed_model` / `HuggingFaceEmbedding` | Le README parle d'embeddings de manière générale ; dans le code, c'est l'objet `embed_model` qui les produit. |
| « vector index » | `VectorStoreIndex` | Terminologie LlamaIndex pour l'index vectoriel. |
| « smart caching » | `_get_index_path` + `os.path.exists` | La vérification de cache repose sur l'existence du répertoire `index_<md5>`. |

---

## 9. Questions ouvertes

> Ambiguïtés à lever avec les experts. Une question marquée non résolue depuis longtemps est un signal — voir [index.md §5](index.md#5-signaux-de-dérive-à-surveiller).

| # | Question | Impact | Owner | Ouverte depuis |
|---|---|---|---|---|
| Q1 | Le fichier `config.py` contenant `GOOGLE_API_KEY` est-il versionné ou exclu du repo ? | Bloquant (sécurité) | (à confirmer) | 2026-05-11 |
| Q2 | Quelle version exacte de Gemini est ciblée — `gemini-2.5-flash` est-il stable ou preview ? | Cosmétique | (à confirmer) | 2026-05-11 |
| Q3 | La flat layout (modules à la racine) est-elle intentionnelle ou doit-on migrer vers `src/` ? | Cosmétique | (à confirmer) | 2026-05-11 |

---

## Références

- Architecture (où ces termes sont utilisés en pratique) : [architecture.md](architecture.md)
- Modèle de données (conventions colonnes détaillées) : [data-model.md](data-model.md)
- Contrats (conventions interfaces détaillées) : [contracts.md](contracts.md)
- Fonctionnel (règles métier nommées) : [fonctionnel.md](fonctionnel.md)
- ADR : (à confirmer — aucun dossier ADR identifié dans ce dépôt)

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

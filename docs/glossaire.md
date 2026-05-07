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
| **Dernière mise à jour** | 2026-05-07 |
| **Mise à jour par** | agent IA |
| **PR de référence** | (à confirmer) |
| **Périmètre** | `pdf_loader.py`, `rag_engine.py`, `text_processor.py` |

> Source de vérité du vocabulaire projet. Termes marqués **(à confirmer)** = déductions à valider. Marqués **(legacy)** = conservés pour compatibilité, à ne pas reproduire.

---

## 1. Vocabulaire métier

### 1.1 Concepts centraux

| Terme | Définition | Représentation code | Apparaît dans |
|---|---|---|---|
| **RAGEngine** | Orchestrateur principal du système : charge un PDF, construit ou recharge l'index vectoriel, et répond aux questions via un LLM. | Classe `RAGEngine` | `rag_engine.py` |
| **index vectoriel** | Structure de données qui stocke les embeddings des chunks de texte pour permettre la recherche sémantique. Persisté dans `./storage/<md5_of_url>/`. | `VectorStoreIndex` (llama-index) | `rag_engine.py` |
| **cache d'index** | Index préalablement calculé et sauvegardé sur disque ; permet de sauter la phase d'embedding lors d'un rechargement du même PDF. | `./storage/<md5_of_url>/` | `rag_engine.py` |
| **streaming download** | Téléchargement HTTP par transfert chunké (sans charger le fichier entier en RAM). | `requests.get(..., stream=True)` dans `load_pdf_from_url` | `pdf_loader.py` |
| **temp file** | Fichier PDF temporaire créé via `mkstemp`, géré par l'appelant qui en assume le cycle de vie (suppression après usage). | `tmp_path` dans `load_pdf_from_url` | `pdf_loader.py` |
| **page batch** | Groupe de `_PAGE_BATCH` (50) pages traitées consécutivement avant un appel à `gc.collect()`, pour borner la consommation mémoire. | `_PAGE_BATCH = 50` | `pdf_loader.py` |
| **chunk** | Fragment de texte extrait d'un PDF, de taille maximale `chunk_size` tokens, avec un chevauchement `chunk_overlap` avec le chunk précédent. | `SentenceSplitter(chunk_size=512, chunk_overlap=50)` | `text_processor.py` |

### 1.2 Règles de gestion (noms courts)

> Liste des règles métier nommées qui apparaissent dans le code, les commits, les tickets. Le détail vit dans [fonctionnel.md](fonctionnel.md).

| Nom court | Sens | Détail dans |
|---|---|---|
| **Batch mémoire** | Limiter le traitement à `_PAGE_BATCH` pages à la fois pour éviter un OOM lors de l'extraction PDF. | [fonctionnel.md](fonctionnel.md) |
| **Cache-avant-embedding** | Vérifier l'existence d'un index persisté avant de lancer l'embedding ; charger depuis le cache si présent. | [fonctionnel.md](fonctionnel.md) |

### 1.3 États et transitions nommés

> Cycle de vie d'une session RAGEngine.

| État | Sens | Transitions sortantes |
|---|---|---|
| `INIT` | RAGEngine instancié, modèles chargés, chemin de cache calculé | → `CACHE_HIT`, `DOWNLOADING` |
| `CACHE_HIT` | Index existant trouvé dans `./storage/` — rechargement direct | → `READY` |
| `DOWNLOADING` | Téléchargement du PDF par streaming | → `INDEXING` |
| `INDEXING` | Construction du `VectorStoreIndex` via embedding des chunks | → `READY`, `OOM_ERROR` |
| `READY` | Index disponible, requêtes utilisateur acceptées | — |
| `OOM_ERROR` | `MemoryError` levée pendant l'embedding — session invalide | — |

---

## 2. Acronymes et abréviations

### 2.1 Acronymes métier

| Acronyme | Signification | Confiance | Origine |
|---|---|---|---|
| **RAG** | Retrieval-Augmented Generation — architecture combinant recherche vectorielle et génération par LLM | ✓ | `rag_engine.py` — commentaire + classe `RAGEngine` |

### 2.2 Acronymes techniques

| Acronyme | Signification | Domaine |
|---|---|---|
| **OOM** | Out-Of-Memory — condition déclenchée quand l'embedding dépasse la RAM disponible ; interceptée via `MemoryError` + `gc.collect()` | Python / llama-index — `rag_engine.py` |
| **BGE** | BAAI General Embedding — famille de modèles d'embedding HuggingFace ; modèle utilisé : `BAAI/bge-small-en-v1.5` | NLP / HuggingFace — `text_processor.py` |
| **LLM** | Large Language Model — modèle de langage utilisé pour générer les réponses (Gemini via `gemini_client.py`) | IA générative — `rag_engine.py` |

---

## 3. Conventions de nommage — Code

### 3.1 Packages / modules

```
<nom_module>.py  (racine plate — pas de sous-packages pour l'instant)
```

| Segment | Valeurs autorisées | Exemple |
|---|---|---|
| `<nom_module>` | `pdf_loader`, `rag_engine`, `text_processor`, `gemini_client` | `rag_engine.py` |

### 3.2 Classes

- **Casse** : `PascalCase` (Python)
- **Suffixes par rôle** :

| Rôle | Suffixe | Exemple |
|---|---|---|
| Orchestrateur RAG | `Engine` | `RAGEngine` |
| Entité persistée | aucun suffixe | (à confirmer) |
| Service / utilitaire | aucun suffixe | (à confirmer) |
| Exception | `Error` ou `Exception` (standard Python) | `MemoryError`, `ImportError` |

### 3.3 Méthodes et fonctions

- **Casse** : `snake_case` (Python)
- **Verbes d'action** : `load_*`, `create_*`, `setup_*`, `initialize_*`
- **Booléens** : préfixe `is_` / `has_` / `can_` ; jamais de double négation
- **Async** : non utilisé dans cette base (synchrone)

### 3.4 Variables et constantes

- **Variables locales** : `snake_case` — courtes mais explicites, pas d'abréviations cryptiques
- **Constantes de module** : `UPPER_SNAKE_CASE` avec préfixe `_` si privées (ex. `_PAGE_BATCH`, `_DOWNLOAD_CHUNK`, `_REQUEST_TIMEOUT`)
- **Énumérations** : non utilisées à ce jour (à confirmer)
- **Identifiants** : conserver le terme métier (ex. `pdf_url` plutôt que `url` si le contexte est exclusivement PDF)

### 3.5 Tests

> Pas de suite de tests identifiée dans le dépôt à ce jour. (à confirmer)

| Type | Pattern de fichier | Pattern de fonction |
|---|---|---|
| Unitaire | `test_<module>.py` | `test_<comportement>` |

### 3.6 Fichiers, branches, commits

- **Fichiers** : `snake_case.py` pour les modules Python, `kebab-case.md` pour la documentation
- **Branches** : (à confirmer)
- **Commits** : (à confirmer)

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

> Pas de base de données relationnelle identifiée. Le stockage est le système de fichiers (`./storage/<md5>/`). (à confirmer)

---

## 5. Conventions de nommage — Interfaces

### 5.1 Endpoints HTTP

- **Casse** : `kebab-case` dans les paths
- **Pluriel pour les collections** : `/orders`, `/orders/{id}`
- **Verbes** : à éviter dans les paths sauf actions hors CRUD (`/orders/{id}:cancel`)
- **Versioning** : {{`/v1/...`}}

### 5.2 Topics / queues

> Pas de messagerie asynchrone identifiée dans ce projet. (à confirmer)

### 5.3 Schémas d'événements

> Non applicable à ce stade. (à confirmer)

---

## 6. Vocabulaire technique

### 6.1 Stack

| Terme | Définition | Lien |
|---|---|---|
| **llama-index** | Framework Python d'orchestration RAG : parsing, indexation vectorielle, requêtage via LLM. Fournit `VectorStoreIndex`, `SentenceSplitter`, `Settings`. | `rag_engine.py`, `text_processor.py` |
| **pypdf** | Bibliothèque Python d'extraction de texte page par page depuis des fichiers PDF. | `pdf_loader.py` |
| **HuggingFace / BGE** | Modèle d'embedding `BAAI/bge-small-en-v1.5` chargé via `llama_index.embeddings.huggingface.HuggingFaceEmbedding` ; convertit les chunks de texte en vecteurs denses. | `text_processor.py` |
| **Gemini** | LLM de Google utilisé pour la génération de réponses ; initialisé dans `gemini_client.py`. | `rag_engine.py` |
| **requests** | Bibliothèque HTTP Python utilisée pour le téléchargement en streaming du PDF. | `pdf_loader.py` |

### 6.2 Patterns

| Pattern | Sens dans ce projet | Où il est appliqué |
|---|---|---|
| **RAG (Retrieval-Augmented Generation)** | Les questions sont traitées en deux temps : (1) recherche des chunks les plus proches sémantiquement dans l'index vectoriel, (2) génération d'une réponse par le LLM enrichi de ce contexte. | `rag_engine.py` — classe `RAGEngine` |
| **Cache-aside** | L'index est cherché dans `./storage/` avant tout calcul ; s'il existe, on le charge directement sans re-embedding. | `rag_engine.py` — `_get_index_path()` |

### 6.3 Démarche / process

| Terme | Définition |
|---|---|
| **ADR** | Architecture Decision Record — décisions traçables, voir (à confirmer) |

---

## 7. Termes à éviter / pièges de vocabulaire

> Termes ambigus, faux amis, ou collisions avec des termes techniques. Cette section évite les confusions répétitives.

| Terme | Pourquoi l'éviter | Préférer |
|---|---|---|
| « document » (sens générique) | Collision avec `llama_index.core.Document` qui désigne spécifiquement un objet texte structuré issu du PDF | `chunk` pour un fragment de texte, `Document` (avec majuscule) pour l'objet llama-index |
| « index » (sans qualificatif) | Ambigu : peut désigner le `VectorStoreIndex` en mémoire ou le répertoire `./storage/` sur disque | `index vectoriel` (en mémoire) / `cache d'index` (sur disque) |

---

## 8. Correspondance / mapping

> Pas de double vocabulaire identifié à ce stade. (à confirmer)

---

## 9. Questions ouvertes

> Ambiguïtés à lever avec les experts. Une question marquée non résolue depuis longtemps est un signal — voir [index.md §5](index.md#5-signaux-de-dérive-à-surveiller).

| # | Question | Impact | Owner | Ouverte depuis |
|---|---|---|---|---|
| Q1 | Quel est le nom officiel du projet (utilisé dans les en-têtes, les tags, le README) ? | Cosmétique | (à confirmer) | 2026-05-07 |
| Q2 | `embed_batch_size` est fixé à 32 dans `text_processor.py` mais `Settings.embed_batch_size` est cité dans le diff comme paramètre ajustable — y a-t-il une valeur recommandée documentée ? | Bloquant si OOM | (à confirmer) | 2026-05-07 |

---

## Références

- Architecture (où ces termes sont utilisés en pratique) : [architecture.md](architecture.md)
- Modèle de données (conventions colonnes détaillées) : [data-model.md](data-model.md)
- Contrats (conventions interfaces détaillées) : [contracts.md](contracts.md)
- Fonctionnel (règles métier nommées) : [fonctionnel.md](fonctionnel.md)
- ADR : (à confirmer)

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

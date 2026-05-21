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
| **Mise à jour par** | agent IA doc-sync |
| **PR de référence** | (à confirmer) |
| **Périmètre** | `rag_engine.py`, `text_processor.py`, `gemini_client.py` |

> Source de vérité du vocabulaire projet. Termes marqués **(à confirmer)** = déductions à valider. Marqués **(legacy)** = conservés pour compatibilité, à ne pas reproduire.

---

## 1. Vocabulaire métier

### 1.1 Concepts centraux

| Terme | Définition | Représentation code | Apparaît dans |
|---|---|---|---|
| **Index vectoriel** | Représentation de documents sous forme de vecteurs numériques permettant la recherche sémantique par similarité. | `VectorStoreIndex` | `rag_engine.py` |
| **Chunk** | Fragment de texte issu du découpage d'un document PDF, unité de base de l'indexation. Taille fixée à 512 tokens, chevauchement à 50. | `SentenceSplitter(chunk_size=512, chunk_overlap=50)` | `text_processor.py` |
| **Node** | Objet LlamaIndex représentant un chunk enrichi de métadonnées, produit par le `node_parser`. | `Settings.node_parser` | `rag_engine.py`, `text_processor.py` |
| **Stockage persistant** | Cache des embeddings sur disque, identifié par le hash MD5 de l'URL du PDF. Évite de re-encoder à chaque démarrage. | `storage_dir` (défaut `./storage`) | `rag_engine.py` |

### 1.2 Règles de gestion (noms courts)

> Liste des règles métier nommées qui apparaissent dans le code, les commits, les tickets. Le détail vit dans [fonctionnel.md](fonctionnel.md).

| Nom court | Sens | Détail dans |
|---|---|---|
| **Cache-par-URL** | Les embeddings sont mis en cache par hash MD5 de l'URL PDF ; un même PDF ne génère des embeddings qu'une seule fois. | [fonctionnel.md](fonctionnel.md) |

### 1.3 États et transitions nommés

> Cycle de vie de l'index au démarrage du `RAGEngine`.

| État | Sens | Transitions sortantes |
|---|---|---|
| `INDEX_ABSENT` | Aucun cache trouvé pour l'URL donnée. | → `INDEX_BUILDING` |
| `INDEX_BUILDING` | Chargement PDF, découpage en chunks, génération des embeddings, persistance sur disque. | → `INDEX_READY` |
| `INDEX_CACHED` | Embeddings déjà présents sur disque, chargement direct depuis `storage_dir`. | → `INDEX_READY` |
| `INDEX_READY` | Index disponible, moteur de requête (`query_engine`) opérationnel avec `similarity_top_k=3`. | — |

---

## 2. Acronymes et abréviations

### 2.1 Acronymes métier

| Acronyme | Signification | Confiance | Origine |
|---|---|---|---|
| **RAG** | Retrieval-Augmented Generation | ✓ | Nom du module principal (`rag_engine.py`, classe `RAGEngine`) |
| **PDF** | Portable Document Format | ✓ | Format source des documents indexés (`pdf_loader.py`) |

### 2.2 Acronymes techniques

| Acronyme | Signification | Domaine |
|---|---|---|
| **BGE** | BAAI General Embedding | Modèle d'embedding HuggingFace (`BAAI/bge-small-en-v1.5`) — `text_processor.py` |
| **LLM** | Large Language Model | Désigne ici l'instance `Gemini` configurée via `gemini_client.py` |
| **API** | Application Programming Interface | Clé d'accès Gemini via `GOOGLE_API_KEY` (variable d'environnement) |

---

## 3. Conventions de nommage — Code

### 3.1 Packages / modules

```
<nom_fonctionnel>.py  (ex. rag_engine.py, text_processor.py, gemini_client.py)
```

Projet Python monolithique, pas de sous-packages. Un fichier = un rôle fonctionnel.

| Fichier | Rôle |
|---|---|
| `rag_engine.py` | Orchestration : chargement, indexation, requêtes |
| `text_processor.py` | Embedding et découpage en chunks |
| `gemini_client.py` | Initialisation du LLM Gemini |
| `pdf_loader.py` | Chargement et lecture des PDFs |

### 3.2 Classes

- **Casse** : `PascalCase`
- **Suffixes par rôle** :

| Rôle | Suffixe | Exemple |
|---|---|---|
| Moteur RAG (orchestrateur) | `Engine` | `RAGEngine` |
| Client LLM externe | aucun (nom du modèle) | `Gemini` (LlamaIndex) |
| Parseur de nœuds | `Splitter` | `SentenceSplitter` |
| Modèle d'embedding | `Embedding` | `HuggingFaceEmbedding` |

### 3.3 Méthodes et fonctions

- **Casse** : `snake_case` (Python)
- **Verbes d'action** : `load_*`, `create_*`, `setup_*`, `initialize_*`, `query`
- **Méthodes privées** : préfixe `_` (ex. `_get_index_path`, `_save_index`, `_load_index`)
- **Booléens** : préfixe `is_` / `has_` / `can_`

### 3.4 Variables et constantes

- **Variables locales** : `snake_case`
- **Constantes d'environnement** : `UPPER_SNAKE_CASE` (ex. `GOOGLE_API_KEY`)
- **Paramètres numériques clés** :

| Paramètre | Valeur | Fichier |
|---|---|---|
| `chunk_size` | 512 | `text_processor.py` |
| `chunk_overlap` | 50 | `text_processor.py` |
| `similarity_top_k` | 3 | `rag_engine.py` |

### 3.5 Tests

| Type | Pattern de fichier | Pattern de méthode |
|---|---|---|
| Unitaire | `test_<module>.py` | `test_<comportement>` |
| Intégration | `test_<module>_integration.py` | `test_<scénario>` |

### 3.6 Fichiers, branches, commits

- **Fichiers Python** : `snake_case.py`
- **Fichiers doc** : `kebab-case.md`
- **Branches** : `feat/...`, `fix/...`, `chore/...`
- **Commits** : Conventional Commits — `feat(scope): description`

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

> Ce projet utilise un stockage vectoriel sur fichier (LlamaIndex `StorageContext`), pas de base relationnelle. Les conventions de nommage de colonnes SQL des §4.1 et §4.2 sont conservées pour référence future si une base de données est ajoutée.

- **Répertoire de stockage** : `snake_case` (ex. `./storage/index_<md5hash>`)

---

## 5. Conventions de nommage — Interfaces

### 5.1 Endpoints HTTP

- **Casse** : `kebab-case` dans les paths
- **Pluriel pour les collections** : `/orders`, `/orders/{id}`
- **Verbes** : à éviter dans les paths sauf actions hors CRUD (`/orders/{id}:cancel`)
- **Versioning** : (à confirmer — pas d'API HTTP exposée dans l'état actuel du projet)

### 5.2 Topics / queues

> Ce projet n'utilise pas de système de messagerie asynchrone. Section conservée pour référence si un bus d'événements est introduit ultérieurement.

### 5.3 Schémas d'événements

> Non applicable dans l'état actuel du projet.

---

## 6. Vocabulaire technique

### 6.1 Stack

| Terme | Définition | Lien |
|---|---|---|
| **LlamaIndex** | Framework Python d'orchestration RAG : indexation, stockage, requêtes vectorielles. | [llamaindex.ai](https://www.llamaindex.ai/) |
| **HuggingFaceEmbedding** | Génère les embeddings texte via le modèle `BAAI/bge-small-en-v1.5`. | `text_processor.py` |
| **Gemini** | LLM Google utilisé pour la génération de réponses (`models/gemini-2.5-flash`, temp. 0.1). | `gemini_client.py` |
| **VectorStoreIndex** | Index LlamaIndex stockant les vecteurs en mémoire avec persistance disque optionnelle. | `rag_engine.py` |
| **SentenceSplitter** | Parseur LlamaIndex découpant le texte en chunks de 512 tokens avec chevauchement de 50. | `text_processor.py` |

### 6.2 Patterns

| Pattern | Sens dans ce projet | Où il est appliqué |
|---|---|---|
| **RAG (Retrieval-Augmented Generation)** | Les chunks pertinents sont récupérés par similarité vectorielle avant d'être injectés dans le prompt du LLM. | `rag_engine.py` — méthode `query()` |
| **Cache-par-empreinte** | Les embeddings sont persistés sur disque sous un nom dérivé du hash MD5 de l'URL source, évitant la ré-indexation. | `rag_engine.py` — `_get_index_path()` |

### 6.3 Démarche / process

| Terme | Définition |
|---|---|
| **Conventional Commits** | Format de message de commit structuré : `type(scope): description`. |
| **ADR** | Architecture Decision Record — décisions traçables, voir (à confirmer) |

---

## 7. Termes à éviter / pièges de vocabulaire

> Termes ambigus, faux amis, ou collisions avec des termes techniques. Cette section évite les confusions répétitives.

| Terme | Pourquoi l'éviter | Préférer |
|---|---|---|
| « embedding model » (générique) | Ambigu : le projet utilise un modèle BGE spécifique, pas un embedding quelconque. | `HuggingFaceEmbedding` avec `BAAI/bge-small-en-v1.5` |
| « database » / « base de données » | Ce projet n'a pas de base relationnelle ; terme confusant face au `StorageContext` LlamaIndex. | « index vectoriel », « stockage persistant » |

---

## 8. Correspondance / mapping

> Non applicable dans l'état actuel du projet — vocabulaire unique.

---

## 9. Questions ouvertes

> Ambiguïtés à lever avec les experts. Une question marquée non résolue depuis longtemps est un signal — voir [index.md §5](index.md#5-signaux-de-dérive-à-surveiller).

| # | Question | Impact | Owner | Ouverte depuis |
|---|---|---|---|---|
| Q1 | Le modèle Gemini (`gemini-2.5-flash`) et le modèle BGE (`bge-small-en-v1.5`) sont-ils les versions définitives pour la prod ? | Bloquant (coût, SLA) | (à confirmer) | 2026-05-21 |
| Q2 | La valeur `similarity_top_k=3` a-t-elle été calibrée sur un jeu de test ? | Cosmétique (qualité réponses) | (à confirmer) | 2026-05-21 |

---

## Références

- Architecture (où ces termes sont utilisés en pratique) : [architecture.md](architecture.md)
- Modèle de données (conventions colonnes détaillées) : [data-model.md](data-model.md)
- Contrats (conventions interfaces détaillées) : [contracts.md](contracts.md)
- Fonctionnel (règles métier nommées) : [fonctionnel.md](fonctionnel.md)
- ADR : (à confirmer — dossier ADR non trouvé dans le dépôt)

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

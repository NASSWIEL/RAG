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

# Glossaire et conventions de nommage — RAG PDF Q&A

| Champ | Valeur |
|---|---|
| **Dernière mise à jour** | 2026-05-10 |
| **Mise à jour par** | agent IA |
| **PR de référence** | (à confirmer) |
| **Périmètre** | rag_engine, text_processor, gemini_client, pdf_loader |

> Source de vérité du vocabulaire projet. Termes marqués **(à confirmer)** = déductions à valider. Marqués **(legacy)** = conservés pour compatibilité, à ne pas reproduire.

---

## 1. Vocabulaire métier

### 1.1 Concepts centraux

| Terme | Définition | Représentation code | Apparaît dans |
|---|---|---|---|
| **RAG** | Retrieval-Augmented Generation — paradigme combinant la recherche sémantique dans une base de documents et la génération de réponses par un LLM. | `RAGEngine` ([rag_engine.py](../rag_engine.py)) | rag_engine, glossaire §2.2 |
| **Index vectoriel** | Structure de données contenant les embeddings des chunks de texte, permettant la recherche par similarité sémantique. Persisté sur disque pour réutilisation. | `VectorStoreIndex` (llama_index) | rag_engine.py |
| **Embedding** | Représentation numérique dense d'un texte, produite par un modèle pré-entraîné, permettant de mesurer la similarité sémantique entre passages. | `HuggingFaceEmbedding` ([text_processor.py](../text_processor.py)) | text_processor.py, rag_engine.py |
| **Chunk** | Fragment de texte obtenu en découpant un document source, unité de base indexée. Taille : 512 tokens, chevauchement : 50 tokens. | `SentenceSplitter(chunk_size=512, chunk_overlap=50)` ([text_processor.py](../text_processor.py)) | text_processor.py |
| **Node** | Objet LlamaIndex encapsulant un chunk de texte avec ses métadonnées, résultat du parsing par `SentenceSplitter`. | `SentenceSplitter` → nodes | text_processor.py, rag_engine.py |

### 1.2 Règles de gestion (noms courts)

> Liste des règles métier nommées qui apparaissent dans le code, les commits, les tickets. Le détail vit dans [fonctionnel.md](fonctionnel.md).

| Nom court | Sens | Détail dans |
|---|---|---|
| **Cache index** | Si un index vectoriel existe déjà sur disque pour le hash MD5 de l'URL PDF, il est rechargé sans ré-indexation. | [rag_engine.py](../rag_engine.py) `_get_index_path` |
| **Top-K retrieval** | Les 3 chunks les plus similaires sémantiquement sont récupérés pour construire le contexte de la réponse (`similarity_top_k=3`). | [rag_engine.py](../rag_engine.py) `as_query_engine` |

### 1.3 États et transitions nommés

> Pipeline de traitement d'une requête.

| État | Sens | Transitions sortantes |
|---|---|---|
| `PDF_LOADED` | PDF téléchargé localement depuis l'URL | → `INDEXED` |
| `INDEXED` | Index vectoriel construit ou rechargé depuis le cache | → `READY` |
| `READY` | Query engine initialisé, prêt à répondre | → `QUERIED` |
| `QUERIED` | Question posée, réponse générée par le LLM | → `READY` |

---

## 2. Acronymes et abréviations

### 2.1 Acronymes métier

| Acronyme | Signification | Confiance | Origine |
|---|---|---|---|
| **RAG** | Retrieval-Augmented Generation | ✓ | Nom du module `RAGEngine` ([rag_engine.py](../rag_engine.py)) |
| **PDF** | Portable Document Format — format des documents sources ingérés | ✓ | [pdf_loader.py](../pdf_loader.py) |
| **Q&A** | Question & Answer — mode d'interaction utilisateur avec le système | ✓ | Docstring `rag_engine.py` |

### 2.2 Acronymes techniques

| Acronyme | Signification | Domaine |
|---|---|---|
| **LLM** | Large Language Model | IA générative — désigne ici Gemini (`gemini_client.py`) |
| **BGE** | BAAI General Embedding | Modèle d'embedding — `BAAI/bge-small-en-v1.5` dans `text_processor.py` |
| **BAAI** | Beijing Academy of Artificial Intelligence | Organisation productrice du modèle BGE |
| **API** | Application Programming Interface | Accès à Gemini via `GOOGLE_API_KEY` (`config.py`) |

---

## 3. Conventions de nommage — Code

### 3.1 Packages / modules

```
<fonctionnalité>.py  (modules plats, pas de hiérarchie de paquets)
```

| Module | Rôle |
|---|---|
| `rag_engine.py` | Orchestration — construction de l'index, requêtage |
| `text_processor.py` | Configuration des embeddings et du node parser |
| `gemini_client.py` | Initialisation du LLM Gemini |
| `pdf_loader.py` | Téléchargement et lecture du PDF source |
| `config.py` | Variables de configuration (clés API, etc.) |

### 3.2 Classes

- **Casse** : PascalCase (Python)
- **Suffixes par rôle** :

| Rôle | Suffixe | Exemple |
|---|---|---|
| Moteur RAG | `Engine` | `RAGEngine` |
| Aucun suffixe imposé pour les autres classes (à confirmer) | — | — |

### 3.3 Méthodes et fonctions

- **Casse** : snake_case (Python)
- **Verbes d'action observés** : `initialize_*`, `setup_*`, `create_*`, `load_*`
- **Méthodes privées** : préfixe `_` (ex. `_get_index_path`, `_save_index`, `_load_index`)
- **Booléens** : préfixe `is_` / `has_` / `can_` ; jamais de double négation

### 3.4 Variables et constantes

- **Variables locales** : snake_case — courtes mais explicites, pas d'abréviations cryptiques
- **Constantes** : UPPER_SNAKE_CASE (ex. `GOOGLE_API_KEY` dans `config.py`)
- **Identifiants de configuration** : conserver le terme officiel de l'API externe (ex. `GOOGLE_API_KEY`)

### 3.5 Tests

> Aucun fichier de test détecté dans le dépôt à ce jour. (à confirmer)

### 3.6 Fichiers, branches, commits

- **Fichiers Python** : snake_case.py (ex. `rag_engine.py`, `text_processor.py`, `gemini_client.py`, `pdf_loader.py`)
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

> Pas de base de données relationnelle dans ce projet. La persistance est assurée par le stockage de l'index vectoriel sur le système de fichiers (répertoire `./storage/<hash_md5>/`). (à confirmer)

---

## 5. Conventions de nommage — Interfaces

### 5.1 Endpoints HTTP

> Pas d'API HTTP exposée détectée dans le code source actuel. Le système est utilisé en mode bibliothèque / script. (à confirmer)

### 5.2 Topics / queues

> Pas de système de messagerie détecté. (à confirmer)

### 5.3 Schémas d'événements

> Pas d'événements publiés détectés. (à confirmer)

---

## 6. Vocabulaire technique

### 6.1 Stack

| Terme | Définition | Lien |
|---|---|---|
| **LlamaIndex** | Framework Python orchestrant le pipeline RAG : chargement de documents, indexation, requêtage. | [llama_index](https://docs.llamaindex.ai) |
| **HuggingFaceEmbedding** | Classe LlamaIndex encapsulant le modèle `BAAI/bge-small-en-v1.5` pour la génération d'embeddings. | [text_processor.py](../text_processor.py) |
| **SentenceSplitter** | Node parser LlamaIndex découpant le texte en chunks de 512 tokens avec chevauchement de 50. | [text_processor.py](../text_processor.py) |
| **VectorStoreIndex** | Index en mémoire/disque stockant les vecteurs d'embeddings pour la recherche par similarité. | [rag_engine.py](../rag_engine.py) |
| **Gemini** | LLM Google (modèle `models/gemini-2.5-flash`, température 0.1) utilisé pour la génération des réponses. | [gemini_client.py](../gemini_client.py) |
| **PDFReader** | Lecteur LlamaIndex (`llama_index.readers.file`) extrayant le texte brut d'un fichier PDF local. | [pdf_loader.py](../pdf_loader.py) |

### 6.2 Patterns

| Pattern | Sens dans ce projet | Où il est appliqué |
|---|---|---|
| **RAG (Retrieval-Augmented Generation)** | Recherche sémantique sur les chunks indexés, puis génération de réponse par le LLM à partir du contexte récupéré. | `RAGEngine.query()` ([rag_engine.py](../rag_engine.py)) |
| **Cache sur hash d'URL** | L'index vectoriel est persisté sous `./storage/index_<md5_url>/`. Si le répertoire existe, il est rechargé sans ré-indexation. | `RAGEngine._get_index_path()` |

### 6.3 Démarche / process

| Terme | Définition |
|---|---|
| **ADR** | Architecture Decision Record — décisions traçables, voir (à confirmer) |

---

## 7. Termes à éviter / pièges de vocabulaire

> Termes ambigus, faux amis, ou collisions avec des termes techniques. Cette section évite les confusions répétitives.

| Terme | Pourquoi l'éviter | Préférer |
|---|---|---|
| « base de données » | Ambigu — ce projet ne possède pas de BDD relationnelle ; la persistance est un répertoire d'index vectoriel sur disque. | « index vectoriel persisté » / « cache d'index » |

---

## 8. Correspondance / mapping

> À utiliser si le projet a deux vocabulaires en parallèle (ex. ancien système → nouveau, métier → technique, API publique → modèle interne). Sinon, supprimer la section.

| Terme LlamaIndex | Sens dans ce projet | Note |
|---|---|---|
| `Document` | Contenu textuel extrait d'une page PDF | Produit par `PDFReader` |
| `Node` | Chunk de texte issu du découpage d'un `Document` | Produit par `SentenceSplitter` |
| `Index` | Structure de recherche vectorielle | `VectorStoreIndex` persisté sur disque |
| `QueryEngine` | Interface de requêtage combinant retrieval et LLM | Configuré avec `similarity_top_k=3`, mode `compact` |

---

## 9. Questions ouvertes

> Ambiguïtés à lever avec les experts. Une question marquée non résolue depuis longtemps est un signal — voir [index.md §5](index.md#5-signaux-de-dérive-à-surveiller).

| # | Question | Impact | Owner | Ouverte depuis |
|---|---|---|---|---|
| Q1 | Quel est le nom officiel du projet (NOM_PROJET) ? | Cosmétique | (à confirmer) | 2026-05-10 |
| Q2 | Le modèle `gemini-2.5-flash` sera-t-il mis à jour ? Faut-il versionner le modèle dans la config ? | Bloquant | (à confirmer) | 2026-05-10 |
| Q3 | Des tests unitaires ou d'intégration sont-ils prévus ? | Cosmétique | (à confirmer) | 2026-05-10 |

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

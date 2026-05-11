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
| **Mise à jour par** | agent IA |
| **PR de référence** | fcd7a06 |
| **Périmètre** | rag_engine, gemini_client, text_processor, pdf_loader, main |

> Source de vérité du vocabulaire projet. Termes marqués **(à confirmer)** = déductions à valider. Marqués **(legacy)** = conservés pour compatibilité, à ne pas reproduire.

---

## 1. Vocabulaire métier

### 1.1 Concepts centraux

| Terme | Définition | Représentation code | Apparaît dans |
|---|---|---|---|
| **RAG** | Retrieval-Augmented Generation — technique combinant recherche sémantique dans une base de documents et génération de réponses par un LLM. Le moteur indexe un PDF, retrouve les passages pertinents, puis génère une réponse contextualisée. | `RAGEngine` ([rag_engine.py](../rag_engine.py)) | `rag_engine.py`, `main.py` |
| **Index vectoriel** | Représentation numérique du contenu d'un PDF sous forme de vecteurs d'embedding, permettant la recherche sémantique par similarité. Persisté sur disque pour éviter de recalculer les embeddings à chaque lancement. | `VectorStoreIndex` (LlamaIndex), stocké dans `./storage/index_<hash>` | `rag_engine.py` |
| **Embedding** | Vecteur de nombres réels représentant le sens sémantique d'un fragment de texte. Calculé par le modèle `BAAI/bge-small-en-v1.5` via HuggingFace. | `HuggingFaceEmbedding` dans `setup_advanced_text_processing()` ([text_processor.py](../text_processor.py)) | `text_processor.py`, `rag_engine.py` |
| **Chunk / nœud** | Fragment de texte issu du découpage du PDF source, unité de base de l'indexation. Taille : 512 tokens, chevauchement : 50 tokens. | `SentenceSplitter(chunk_size=512, chunk_overlap=50)` ([text_processor.py](../text_processor.py)) | `text_processor.py`, `rag_engine.py` |

### 1.2 Règles de gestion (noms courts)

> Liste des règles métier nommées qui apparaissent dans le code, les commits, les tickets. Le détail vit dans [fonctionnel.md](fonctionnel.md).

| Nom court | Sens | Détail dans |
|---|---|---|
| **Cache par URL** | L'index vectoriel d'un PDF est persisté et réutilisé si un hash MD5 de l'URL existe déjà dans `./storage`. Évite de recalculer les embeddings à chaque exécution. | [fonctionnel.md](fonctionnel.md) |

### 1.3 États et transitions nommés

> Ce projet ne définit pas de machine à états métier explicite. Le cycle de vie de l'index vectoriel est documenté à titre indicatif.

| État | Sens | Transitions sortantes |
|---|---|---|
| `ABSENT` | Aucun index persisté pour l'URL donnée (`./storage/index_<hash>` inexistant) | → `INDEXATION` |
| `INDEXATION` | Téléchargement du PDF, découpage en chunks, calcul des embeddings, persistance sur disque | → `PRET` |
| `PRET` | Index chargé depuis le cache ou fraîchement construit ; moteur de requête disponible | → `INTERROGATION` |
| `INTERROGATION` | Exécution d'une requête utilisateur : recherche sémantique + génération de réponse par Gemini | → `PRET` |

---

## 2. Acronymes et abréviations

### 2.1 Acronymes métier

| Acronyme | Signification | Confiance | Origine |
|---|---|---|---|
| **RAG** | Retrieval-Augmented Generation | ✓ | Littérature académique ; implémenté dans `RAGEngine` ([rag_engine.py](../rag_engine.py)) |

### 2.2 Acronymes techniques

| Acronyme | Signification | Domaine |
|---|---|---|
| **LLM** | Large Language Model | IA générative — ici Gemini 2.5 Flash via `llama_index.llms.gemini` |
| **API** | Application Programming Interface | Accès à Gemini via `GOOGLE_API_KEY` (config.py) |
| **PDF** | Portable Document Format | Format source des documents indexés |
| **BGE** | BAAI General Embedding | Famille de modèles d'embedding HuggingFace ; modèle utilisé : `BAAI/bge-small-en-v1.5` |

---

## 3. Conventions de nommage — Code

### 3.1 Packages / modules

```
<module>.py  (fichiers plats à la racine — pas de sous-packages)
```

| Module | Rôle | Fichier |
|---|---|---|
| `rag_engine` | Moteur RAG central : indexation et requêtage | [rag_engine.py](../rag_engine.py) |
| `gemini_client` | Initialisation du LLM Gemini | [gemini_client.py](../gemini_client.py) |
| `text_processor` | Embedding et découpage en chunks | [text_processor.py](../text_processor.py) |
| `pdf_loader` | Téléchargement et lecture de PDF | [pdf_loader.py](../pdf_loader.py) |
| `main` | Point d'entrée interactif | [main.py](../main.py) |

### 3.2 Classes

- **Casse** : `PascalCase` (Python)
- **Suffixes par rôle** :

| Rôle | Suffixe | Exemple |
|---|---|---|
| Moteur applicatif | `Engine` | `RAGEngine` |
| Autres classes | aucun suffixe imposé | (à confirmer) |

### 3.3 Méthodes et fonctions

- **Casse** : `snake_case` (Python)
- **Verbes d'action** : `initialize_*`, `load_*`, `create_*`, `setup_*`, `query` (exemples tirés du code)
- **Booléens** : préfixe `is_` / `has_` / `can_` ; jamais de double négation
- **Async** : synchrone uniquement dans la base actuelle (à confirmer)

### 3.4 Variables et constantes

- **Variables locales** : `snake_case` — courtes mais explicites, pas d'abréviations cryptiques
- **Constantes** : `UPPER_SNAKE_CASE` (ex. `GOOGLE_API_KEY` dans `config.py`)
- **Énumérations** : aucune dans la base actuelle (à confirmer)
- **Identifiants** : conserver le terme métier (ex. `pdf_url` et non `url`)

### 3.5 Tests

| Type | Pattern de classe | Pattern de méthode |
|---|---|---|
| Unitaire | `<ClasseTestée>Test` | `should<Comportement>When<Condition>` |
| Intégration | `<ClasseTestée>IT` ou `*IntegrationTest` | idem |
| Contract | `<Interface>ContractTest` | idem |
| End-to-end | `<Parcours>E2ETest` | `<scénario fonctionnel>` |

### 3.6 Fichiers, branches, commits

- **Fichiers** : `snake_case.py` (modules Python), `kebab-case.md` (documentation)
- **Branches** : (à confirmer)
- **Commits** : Conventional Commits configurés via `gitlint-core` (voir `pyproject.toml`)

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

Ce projet ne dispose pas de base de données relationnelle. Le stockage est fichier-système (`./storage/index_<md5_url>/`). Les conventions de colonnes §4.1 et §4.2 s'appliquent si une base est ajoutée ultérieurement.

---

## 5. Conventions de nommage — Interfaces

### 5.1 Endpoints HTTP

Ce projet n'expose pas d'API HTTP. L'interface est en ligne de commande interactive (`main.py`). Cette section est réservée à une éventuelle future exposition via API (à confirmer).

### 5.2 Topics / queues

Ce projet n'utilise pas de messagerie asynchrone. Section non applicable dans l'état actuel.

### 5.3 Schémas d'événements

- **Champs systématiques** : (à confirmer — pas d'événements dans la base actuelle)
- **Casse champs** : `snake_case` par convention Python
- **Versioning** : (à confirmer)

---

## 6. Vocabulaire technique

### 6.1 Stack

| Terme | Définition | Lien |
|---|---|---|
| **LlamaIndex** | Framework d'orchestration RAG : gestion des index, des embeddings, des node parsers et des query engines | [llama-index.ai](https://www.llamaindex.ai/) |
| **Gemini** | LLM de Google utilisé pour la génération de réponses. Modèle : `models/gemini-2.5-flash`, température : 0.1 | [ai.google.dev](https://ai.google.dev/) |
| **HuggingFace Embeddings** | Bibliothèque fournissant le modèle d'embedding `BAAI/bge-small-en-v1.5` via `llama_index.embeddings.huggingface` | [huggingface.co](https://huggingface.co/) |
| **PDFReader** | Lecteur de fichiers PDF fourni par `llama_index.readers.file` | [llama-index.ai](https://www.llamaindex.ai/) |
| **SentenceSplitter** | Découpeur de texte en chunks chevauchants (`chunk_size=512`, `chunk_overlap=50`) de LlamaIndex | [llama-index.ai](https://www.llamaindex.ai/) |

### 6.2 Patterns

| Pattern | Sens dans ce projet | Où il est appliqué |
|---|---|---|
| **RAG (Retrieval-Augmented Generation)** | Enrichissement du prompt LLM avec des passages retrouvés par recherche vectorielle avant génération de la réponse | `RAGEngine.query()` ([rag_engine.py](../rag_engine.py)) |
| **Cache-aside** | L'index vectoriel est chargé depuis le disque si disponible ; sinon recalculé et persisté | `RAGEngine.__init__()` ([rag_engine.py](../rag_engine.py)) |

### 6.3 Démarche / process

| Terme | Définition |
|---|---|
| **Conventional Commits** | Convention de messages de commit appliquée via `gitlint-core` (voir `pyproject.toml`) |
| **Ruff** | Linter et formateur Python configuré dans `pyproject.toml` ; cible Python 3.12 |
| **Pyright** | Vérificateur de types statique Python (mode `standard`) |
| **ADR** | Architecture Decision Record — décisions traçables (à confirmer — dossier ADR non présent dans le repo) |

---

## 7. Termes à éviter / pièges de vocabulaire

> Termes ambigus, faux amis, ou collisions avec des termes techniques. Cette section évite les confusions répétitives.

| Terme | Pourquoi l'éviter | Préférer |
|---|---|---|
| « requête » seul | Ambigu entre requête SQL, requête HTTP et question posée au moteur RAG | Préciser : « question utilisateur » pour l'input de `query()`, « recherche vectorielle » pour la phase de retrieval |
| « document » seul | Dans LlamaIndex, `Document` désigne un objet chargé depuis le PDF, pas forcément un fichier entier | Préciser : « objet `Document` LlamaIndex » vs « fichier PDF source » |

---

## 8. Correspondance / mapping

> Mapping entre le vocabulaire utilisateur (question / réponse) et le vocabulaire technique LlamaIndex.

| Vocabulaire utilisateur | Vocabulaire technique LlamaIndex | Note |
|---|---|---|
| Question posée | `question` (arg de `RAGEngine.query()`) | Chaîne libre en langage naturel |
| Réponse générée | `Response` (objet LlamaIndex converti via `str()`) | Inclut les passages sources retrouvés |
| Passage source | `Node` / `TextNode` | Fragment de chunk utilisé pour générer la réponse |
| Base de connaissances | `VectorStoreIndex` persisté dans `./storage` | Index vectoriel calculé depuis le PDF |

---

## 9. Questions ouvertes

> Ambiguïtés à lever avec les experts. Une question marquée non résolue depuis longtemps est un signal — voir [index.md §5](index.md#5-signaux-de-dérive-à-surveiller).

| # | Question | Impact | Owner | Ouverte depuis |
|---|---|---|---|---|
| Q1 | Le modèle d'embedding `BAAI/bge-small-en-v1.5` est-il le choix définitif ou un placeholder ? Un modèle multilingue est-il envisagé ? | Cosmétique / qualité des résultats | (à confirmer) | 2026-05-11 |
| Q2 | La clé `GOOGLE_API_KEY` est-elle chargée depuis un `.env`, une variable d'environnement système ou un fichier `config.py` statique ? | Sécurité | (à confirmer) | 2026-05-11 |

---

## Références

- Architecture (où ces termes sont utilisés en pratique) : [architecture.md](architecture.md)
- Modèle de données (conventions colonnes détaillées) : [data-model.md](data-model.md)
- Contrats (conventions interfaces détaillées) : [contracts.md](contracts.md)
- Fonctionnel (règles métier nommées) : [fonctionnel.md](fonctionnel.md)
- ADR : (à confirmer — dossier ADR non présent dans le repo)

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

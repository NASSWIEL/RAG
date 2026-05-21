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
| **Mise à jour par** | agent IA (template-fill) |
| **PR de référence** | (à confirmer) |
| **Périmètre** | rag_engine, pdf_loader, text_processor, gemini_client, main |

> Source de vérité du vocabulaire projet. Termes marqués **(à confirmer)** = déductions à valider. Marqués **(legacy)** = conservés pour compatibilité, à ne pas reproduire.

---

## 1. Vocabulaire métier

### 1.1 Concepts centraux

| Terme | Définition | Représentation code | Apparaît dans |
|---|---|---|---|
| **Document** | Unité de contenu source — un PDF téléchargé depuis une URL, découpé en chunks avant indexation | `pdf_loader.py` | `rag_engine.py`, `pdf_loader.py` |
| **Chunk** | Fragment de texte issu du découpage sémantique d'un document ; unité de base de l'index vectoriel | `text_processor.py` | `rag_engine.py`, `text_processor.py` |
| **Index** | Structure vectorielle persistée sur disque (`./storage/`) qui stocke les embeddings des chunks pour permettre la recherche sémantique | `rag_engine.py` | `rag_engine.py` |
| **Query** | Question posée par l'utilisateur en langage naturel ; déclenche une recherche sémantique dans l'index puis une génération de réponse | `main.py` | `rag_engine.py`, `main.py` |
| **Contexte récupéré** | Ensemble des top-k chunks les plus proches sémantiquement de la query ; transmis au LLM pour générer la réponse | `rag_engine.py` | `rag_engine.py`, `gemini_client.py` |
| **Génération directe** | Appel au LLM sans phase de récupération vectorielle — utilisé pour des tâches comme la reformulation ou le résumé autonome | `gemini_client.generate_text()` | `gemini_client.py` |
| **Résumé** | Condensation d'un texte brut vers une cible de mots définie, sans contexte RAG | `gemini_client.summarize()` | `gemini_client.py` |
| **Reranking** | Réordonnancement de passages récupérés par pertinence vis-à-vis de la query, en utilisant le LLM comme reranker zero-shot | `gemini_client.rerank_passages()` | `gemini_client.py` |

### 1.2 Règles de gestion (noms courts)

> Liste des règles métier nommées qui apparaissent dans le code, les commits, les tickets. Le détail vit dans [fonctionnel.md](fonctionnel.md).

| Nom court | Sens | Détail dans |
|---|---|---|
| **Cache hit** | Si un index existe déjà pour l'URL hashée, les embeddings sont chargés depuis `./storage/` sans re-traitement du PDF | [fonctionnel.md](fonctionnel.md) |
| **top-k retrieval** | Sélection des k chunks les plus proches sémantiquement de la query pour constituer le contexte envoyé au LLM | [fonctionnel.md](fonctionnel.md) |

### 1.3 États et transitions nommés

> Si le métier a des états explicites (statuts d'une commande, phases d'un workflow), les nommer ici.

| État | Sens | Transitions sortantes |
|---|---|---|
| `UNINDEXED` | PDF non encore téléchargé ni découpé | → `INDEXED` |
| `INDEXED` | Embeddings générés et persistés dans `./storage/` | → `QUERYING` |
| `QUERYING` | Session interactive en cours ; l'utilisateur pose des questions | → `QUERYING` (boucle), fin sur `quit` |

---

## 2. Acronymes et abréviations

### 2.1 Acronymes métier

| Acronyme | Signification | Confiance | Origine |
|---|---|---|---|
| **RAG** | Retrieval-Augmented Generation | ✓ | README.md, nom du projet |
| **LLM** | Large Language Model | ✓ | README.md |
| **PDF** | Portable Document Format | ✓ | README.md, `pdf_loader.py` |

### 2.2 Acronymes techniques

| Acronyme | Signification | Domaine |
|---|---|---|
| **CLI** | Command-Line Interface | Interface utilisateur (`main.py`) |
| **HF** | HuggingFace | Librairie d'embeddings (`sentence-transformers`) |
| **API** | Application Programming Interface | Google Gemini (`gemini_client.py`) |

---

## 3. Conventions de nommage — Code

### 3.1 Packages / modules

```
<composant>.py  (à la racine du projet — pas de package hiérarchique)
```

| Segment | Valeurs autorisées | Exemple |
|---|---|---|
| `<composant>` | `rag_engine`, `pdf_loader`, `text_processor`, `gemini_client`, `main`, `config` | `rag_engine.py` |

### 3.2 Classes

- **Casse** : PascalCase (Python, conventions standard)
- **Suffixes par rôle** :

| Rôle | Suffixe | Exemple |
|---|---|---|
| Entité persistée | aucun suffixe | (à confirmer) |
| Orchestrateur RAG | `Engine` | `RAGEngine` (à confirmer) |
| Client LLM | `Client` | `GeminiClient` (à confirmer) |
| Chargeur de données | `Loader` | `PDFLoader` (à confirmer) |
| Processeur de texte | `Processor` | `TextProcessor` (à confirmer) |
| Exception | `Exception` ou `Error` | (à confirmer) |

### 3.3 Méthodes et fonctions

- **Casse** : snake_case (Python)
- **Verbes d'action** : `find*`, `create*`, `update*`, `delete*`, `is*`, `has*`, `can*`, `should*`
- **Booléens** : préfixe `is_` / `has_` / `can_` ; jamais de double négation
- **Async** : suffixe `*Async` (TypeScript) ou retourne `Mono<>` / `Future<>` (typage suffit)

### 3.4 Variables et constantes

- **Variables locales** : snake_case — courtes mais explicites, pas d'abréviations cryptiques
- **Constantes** : UPPER_SNAKE_CASE (ex. `GOOGLE_API_KEY` dans `config.py`)
- **Énumérations** : `enum {{NomEnum}} { VALEUR_A, VALEUR_B }`
- **Identifiants** : conserver le terme métier (ex. `idEnvelope` plutôt que `envId` si « envelope » est le terme officiel)

### 3.5 Tests

| Type | Pattern de classe | Pattern de méthode |
|---|---|---|
| Unitaire | `<ClasseTestée>Test` | `should<Comportement>When<Condition>` |
| Intégration | `<ClasseTestée>IT` ou `*IntegrationTest` | idem |
| Contract | `<Interface>ContractTest` | idem |
| End-to-end | `<Parcours>E2ETest` | `<scénario fonctionnel>` |

### 3.6 Fichiers, branches, commits

- **Fichiers** : `snake_case.py` (Python), `kebab-case.md` (docs)
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

- **Casse** : snake_case (à confirmer — pas de base SQL identifiée dans le projet)
- **Singulier ou pluriel** : (à confirmer)
- **Tables de liaison N:N** : (à confirmer)
- **Tables d'audit** : (à confirmer)

---

## 5. Conventions de nommage — Interfaces

### 5.1 Endpoints HTTP

- **Casse** : `kebab-case` dans les paths
- **Pluriel pour les collections** : `/orders`, `/orders/{id}`
- **Verbes** : à éviter dans les paths sauf actions hors CRUD (`/orders/{id}:cancel`)
- **Versioning** : (à confirmer — pas d'API HTTP exposée dans le projet actuel)

### 5.2 Topics / queues

```
(à confirmer — pas de message broker identifié dans le projet actuel)
```

| Segment | Valeurs |
|---|---|
| `<env>` | `dev`, `stg`, `prod` |
| `<domaine>` | (à confirmer) |
| `<entité>` | (à confirmer) |
| `<événement>` | (à confirmer) |

### 5.3 Schémas d'événements

- **Champs systématiques** : `id`, `event_type`, `schema_version`, `occurred_at`, `producer`
- **Casse champs** : snake_case (à confirmer)
- **Versioning** : champ `schema_version` (entier monotone)

---

## 6. Vocabulaire technique

### 6.1 Stack

| Terme | Définition | Lien |
|---|---|---|
| **LlamaIndex** | Framework d'orchestration RAG — gère l'indexation, le stockage et la query pipeline | [llamaindex.ai](https://www.llamaindex.ai) |
| **HuggingFace sentence-transformers** | Librairie de génération d'embeddings sémantiques pour les chunks | [sbert.net](https://www.sbert.net) |
| **Google Gemini** | LLM utilisé pour générer les réponses à partir du contexte récupéré | [ai.google.dev](https://ai.google.dev) |
| **pypdf** | Extraction du texte brut depuis les fichiers PDF | [pypdf.readthedocs.io](https://pypdf.readthedocs.io) |

### 6.2 Patterns

| Pattern | Sens dans ce projet | Où il est appliqué |
|---|---|---|
| **RAG (Retrieval-Augmented Generation)** | Récupération de contexte documentaire pertinent avant génération LLM — évite les hallucinations | `rag_engine.py` |
| **Cache par hash d'URL** | L'index est persisté sous une clé dérivée du hash de l'URL PDF ; rechargé si présent | `rag_engine.py` |
| **Singleton `_state`** | Dictionnaire module-level `{"llm": None}` dans `gemini_client.py` — garantit qu'une seule instance Gemini est créée et réutilisée par les appelants ; réinitialisable via `reset_llm()` | `gemini_client.py` |

### 6.3 Démarche / process

| Terme | Définition |
|---|---|
| **Conventional Commits** | Convention de message de commit : `feat(scope): description`, `fix(...)`, `chore(...)` |
| **Code review** | (à confirmer — règles internes non documentées) |
| **ADR** | Architecture Decision Record — décisions traçables, voir (à confirmer — dossier ADR non encore créé) |

---

## 7. Termes à éviter / pièges de vocabulaire

> Termes ambigus, faux amis, ou collisions avec des termes techniques. Cette section évite les confusions répétitives.

| Terme | Pourquoi l'éviter | Préférer |
|---|---|---|
| « recherche » seul | Ambiguïté entre recherche sémantique (vectorielle) et recherche textuelle exacte | « recherche sémantique » / « top-k retrieval » |
| « modèle » seul | Collision entre le modèle d'embeddings (HF) et le modèle de génération (Gemini) | « modèle d'embeddings » ou « LLM Gemini » selon le contexte |

---

## 8. Correspondance / mapping

> À utiliser si le projet a deux vocabulaires en parallèle (ex. ancien système → nouveau, métier → technique, API publique → modèle interne). Sinon, supprimer la section.

| Terme README | Terme technique | Note |
|---|---|---|
| « embeddings » | vecteurs d'embedding générés par `sentence-transformers` | Synonymes dans ce projet |
| « storage » | répertoire `./storage/` | Chemin de persistance de l'index LlamaIndex |

---

## 9. Questions ouvertes

> Ambiguïtés à lever avec les experts. Une question marquée non résolue depuis longtemps est un signal — voir [index.md §5](index.md#5-signaux-de-dérive-à-surveiller).

| # | Question | Impact | Owner | Ouverte depuis |
|---|---|---|---|---|
| Q1 | Quel modèle HuggingFace exact est utilisé pour les embeddings ? (`sentence-transformers/...`) | Bloquant pour doc complète | (à confirmer) | 2026-05-21 |
| Q2 | ~~Quelle version de Gemini est configurée dans `gemini_client.py` ?~~ Résolue : `models/gemini-2.5-flash` (`_DEFAULT_MODEL` dans `gemini_client.py`) | Cosmétique | agent IA | 2026-05-21 |
| Q3 | La valeur de k dans le top-k retrieval est-elle paramétrable ou fixée ? | Cosmétique | (à confirmer) | 2026-05-21 |

---

## Références

- Architecture (où ces termes sont utilisés en pratique) : [architecture.md](architecture.md)
- Modèle de données (conventions colonnes détaillées) : [data-model.md](data-model.md)
- Contrats (conventions interfaces détaillées) : [contracts.md](contracts.md)
- Fonctionnel (règles métier nommées) : [fonctionnel.md](fonctionnel.md)
- ADR : (à confirmer — dossier ADR non encore créé)

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

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
| **Dernière mise à jour** | 2026-05-24 |
| **Mise à jour par** | agent IA (doc-patcher) |
| **PR de référence** | (à confirmer) |
| **Périmètre** | Pipeline RAG — ingestion PDF, embeddings, requêtage Gemini |

> Source de vérité du vocabulaire projet. Termes marqués **(à confirmer)** = déductions à valider. Marqués **(legacy)** = conservés pour compatibilité, à ne pas reproduire.

---

## 1. Vocabulaire métier

### 1.1 Concepts centraux

| Terme | Définition | Représentation code | Apparaît dans |
|---|---|---|---|
| **RAG** | Retrieval-Augmented Generation — approche qui combine la recherche dans un index vectoriel de documents avec la génération de réponses par un LLM. | `RAGEngine` dans `rag_engine.py` | `rag_engine.py`, `README.md` |
| **Index vectoriel** | Structure de données qui stocke les embeddings des chunks de texte pour permettre la recherche sémantique par similarité cosinus. | `VectorStoreIndex` (LlamaIndex) dans `rag_engine.py` | `rag_engine.py` |
| **Chunk** | Segment de texte produit par découpage d'un document PDF, de taille fixe (`chunk_size=512` tokens, chevauchement `chunk_overlap=50`). | `SentenceSplitter` dans `text_processor.py` | `text_processor.py` |
| **Embedding** | Représentation vectorielle dense d'un chunk de texte, calculée par le modèle `BAAI/bge-small-en-v1.5`. | `HuggingFaceEmbedding` dans `text_processor.py` | `text_processor.py` |
| **Node** | Unité d'indexation dans LlamaIndex, correspondant à un chunk enrichi de métadonnées (position dans le PDF, score de similarité). | `SentenceSplitter` → nœuds dans `rag_engine.py` | `rag_engine.py`, `text_processor.py` |

### 1.2 Règles de gestion (noms courts)

> Liste des règles métier nommées qui apparaissent dans le code, les commits, les tickets. Le détail vit dans [fonctionnel.md](fonctionnel.md).

| Nom court | Sens | Détail dans |
|---|---|---|
| (à confirmer) | Aucune règle de gestion nommée identifiée dans le code actuel. | — |

### 1.3 États et transitions nommés

> Si le métier a des états explicites (statuts d'une commande, phases d'un workflow), les nommer ici.

| État | Sens | Transitions sortantes |
|---|---|---|
| (à confirmer) | Aucune machine à états explicite identifiée dans le code actuel. | — |

---

## 2. Acronymes et abréviations

### 2.1 Acronymes métier

| Acronyme | Signification | Confiance | Origine |
|---|---|---|---|
| **RAG** | Retrieval-Augmented Generation | ✓ | `README.md`, `rag_engine.py` |
| **PDF** | Portable Document Format — format source des documents ingérés | ✓ | `pdf_loader.py`, `rag_engine.py` |

### 2.2 Acronymes techniques

| Acronyme | Signification | Domaine |
|---|---|---|
| **LLM** | Large Language Model — modèle de langage utilisé pour la génération de réponses (ici, Gemini 2.5 Flash) | IA / NLP |
| **API** | Application Programming Interface — interface d'accès à Gemini via clé `GOOGLE_API_KEY` | Intégration externe |
| **BGE** | BAAI General Embedding — famille de modèles d'embedding de BAAI (`bge-small-en-v1.5`) | NLP / Embeddings |

---

## 3. Conventions de nommage — Code

### 3.1 Packages / modules

```
<module>.py  (modules à la racine du projet — pas de sous-package imposé)
```

| Segment | Valeurs autorisées | Exemple |
|---|---|---|
| module racine | `rag_engine`, `text_processor`, `gemini_client`, `pdf_loader` | `rag_engine.py` |

### 3.2 Classes

- **Casse** : PascalCase (Python, conforme ruff/N801)
- **Suffixes par rôle** :

| Rôle | Suffixe | Exemple |
|---|---|---|
| Entité persistée | aucun suffixe | `RAGEngine` |
| Engine (orchestrateur) | `Engine` | `RAGEngine` dans `rag_engine.py` |
| Client externe | `Client` ou `initialize_*` (fonction) | `initialize_gemini_llm()` dans `gemini_client.py` |
| Utilitaires | `*_processor`, `*_loader` | `text_processor.py`, `pdf_loader.py` |
| Exception | `Exception` ou `Error` | (à confirmer — aucune classe d'exception personnalisée dans le code actuel) |

### 3.3 Méthodes et fonctions

- **Casse** : snake_case (Python, conforme ruff/N802)
- **Verbes d'action** : `load_*`, `setup_*`, `create_*`, `initialize_*`, `query`
- **Booléens** : préfixe `is_` / `has_` / `can_` ; jamais de double négation
- **Async** : suffixe `*Async` (TypeScript) ou retourne `Mono<>` / `Future<>` (typage suffit)

### 3.4 Variables et constantes

- **Variables locales** : snake_case — courtes mais explicites, pas d'abréviations cryptiques
- **Constantes** : UPPER_SNAKE_CASE
- **Énumérations** : (à confirmer — aucune enum personnalisée dans le code actuel)
- **Identifiants** : conserver le terme métier (ex. `idEnvelope` plutôt que `envId` si « envelope » est le terme officiel)

### 3.5 Tests

| Type | Pattern de classe | Pattern de méthode |
|---|---|---|
| Unitaire | `<ClasseTestée>Test` | `should<Comportement>When<Condition>` |
| Intégration | `<ClasseTestée>IT` ou `*IntegrationTest` | idem |
| Contract | `<Interface>ContractTest` | idem |
| End-to-end | `<Parcours>E2ETest` | `<scénario fonctionnel>` |

### 3.6 Fichiers, branches, commits

- **Fichiers** : `snake_case.py` (modules Python), `kebab-case.md` (docs)
- **Branches** : `feat/...`, `fix/...`, `chore/...` (à confirmer)
- **Commits** : Conventional Commits — `feat(scope): description` (à confirmer)

---

## 4. Conventions de nommage — Données

> Sync avec [data-model.md §9.1](data-model.md). Lister ici uniquement les conventions DE NOMMAGE ; les détails de typage vivent côté data-model.

### 4.1 Préfixes de colonnes / champs

Non applicable — ce projet ne possède pas de base de données relationnelle. Le seul stockage persistant est l'index vectoriel LlamaIndex sur disque (`./storage/index_<md5>/`).

### 4.2 Suffixes temporels

Non applicable — pas de base de données relationnelle dans ce projet.

### 4.3 Tables / collections

Non applicable — pas de base de données relationnelle dans ce projet.

---

## 5. Conventions de nommage — Interfaces

### 5.1 Endpoints HTTP

Non applicable — ce projet est un pipeline CLI sans API HTTP exposée.

### 5.2 Topics / queues

Non applicable — pas de broker de messages dans ce projet.

### 5.3 Schémas d'événements

Non applicable — pas d'événements asynchrones dans ce projet.

---

## 6. Vocabulaire technique

### 6.1 Stack

| Terme | Définition | Lien |
|---|---|---|
| **LlamaIndex** | Framework d'orchestration RAG — gère l'indexation, le stockage et le requêtage du vector store. | `rag_engine.py`, `text_processor.py`, `gemini_client.py` |
| **HuggingFace Embedding (`bge-small-en-v1.5`)** | Modèle d'embedding utilisé pour vectoriser les chunks de texte. `chunk_size=512`, `chunk_overlap=50`. | `text_processor.py` |
| **Gemini 2.5 Flash** | LLM Google utilisé pour la génération de réponses. Accédé via `llama_index.llms.gemini`, clé `GOOGLE_API_KEY`. | `gemini_client.py` |
| **PDFReader (LlamaIndex)** | Lecteur de documents PDF — extrait le texte brut via `llama_index.readers.file.PDFReader`. | `pdf_loader.py` |

### 6.2 Patterns

| Pattern | Sens dans ce projet | Où il est appliqué |
|---|---|---|
| **RAG (Retrieval-Augmented Generation)** | Récupération de contexte pertinent dans un index vectoriel avant génération LLM. | `RAGEngine` dans `rag_engine.py` |
| **Cache sur disque (index vectoriel)** | L'index est persisté sur disque (hash MD5 de l'URL PDF) pour éviter le recalcul des embeddings. | `RAGEngine._get_index_path()`, `_save_index()`, `_load_index()` |

### 6.3 Démarche / process

| Terme | Définition |
|---|---|
| **Conventional Commits** | Format de commit `type(scope): description` — enforced via `gitlint-core` (voir `pyproject.toml`) (à confirmer) |
| **ADR** | Architecture Decision Record — décisions traçables. Pas de dossier ADR identifié dans ce dépôt (à confirmer). |

---

## 7. Termes à éviter / pièges de vocabulaire

> Termes ambigus, faux amis, ou collisions avec des termes techniques. Cette section évite les confusions répétitives.

| Terme | Pourquoi l'éviter | Préférer |
|---|---|---|
| « recherche sémantique » (sens vague) | Ambigu — peut désigner la recherche par similarité vectorielle ou une recherche full-text enrichie. | Préciser : « recherche par similarité vectorielle (top-k cosinus) » |
| « modèle » seul | Ambigu entre le modèle d'embedding (`bge-small-en-v1.5`) et le LLM (`gemini-2.5-flash`). | `embed_model` ou `llm` selon le contexte, comme dans `RAGEngine`. |

---

## 8. Correspondance / mapping

> À utiliser si le projet a deux vocabulaires en parallèle (ex. ancien système → nouveau, métier → technique, API publique → modèle interne). Sinon, supprimer la section.

Non applicable — pas de double vocabulaire parallèle dans ce projet.

---

## 9. Questions ouvertes

> Ambiguïtés à lever avec les experts. Une question marquée non résolue depuis longtemps est un signal — voir [index.md §5](index.md#5-signaux-de-dérive-à-surveiller).

| # | Question | Impact | Owner | Ouverte depuis |
|---|---|---|---|---|
| Q1 | Le dossier ADR existe-t-il ? Où sont documentées les décisions d'architecture (choix de LlamaIndex, Gemini, BGE) ? | cosmétique | (à confirmer) | 2026-05-24 |
| Q2 | Le workflow git suit-il Conventional Commits ? `gitlint-core` est en dev-dep mais aucune config `.gitlint` n'a été trouvée. | cosmétique | (à confirmer) | 2026-05-24 |

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

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
| **Dernière mise à jour** | 2026-05-10 |
| **Mise à jour par** | agent IA |
| **PR de référence** | (à confirmer) |
| **Périmètre** | rag_engine, pdf_loader, text_processor, gemini_client, main |

> Source de vérité du vocabulaire projet. Termes marqués **(à confirmer)** = déductions à valider. Marqués **(legacy)** = conservés pour compatibilité, à ne pas reproduire.

---

## 1. Vocabulaire métier

### 1.1 Concepts centraux

| Terme | Définition | Représentation code | Apparaît dans |
|---|---|---|---|
| **Document** | Unité de contenu extraite d'un PDF et stockée sous forme de chunks sémantiques dans l'index vectoriel. | `Document` (llama-index) | `pdf_loader.py`, `rag_engine.py` |
| **Chunk** | Fragment de texte issu du découpage d'un document, unité atomique de recherche sémantique. | nœud de l'index vectoriel | `text_processor.py`, `rag_engine.py` |
| **Embedding** | Représentation vectorielle d'un chunk, générée par un modèle HuggingFace sentence-transformer, permettant la recherche par similarité. | tenseur float stocké dans `./storage/` | `text_processor.py`, `rag_engine.py` |
| **Index vectoriel** | Structure de données (persistée sur disque) associant chaque chunk à son embedding pour la recherche top-k. | `VectorStoreIndex` (llama-index) | `rag_engine.py` |
| **Requête** | Question posée par l'utilisateur en langage naturel, transformée en embedding pour interroger l'index. | paramètre `query` de `RAGEngine.query()` | `rag_engine.py`, `main.py` |
| **Contexte récupéré** | Ensemble des chunks top-k renvoyés par la recherche sémantique et fournis au LLM pour génération. | `retrieved_nodes` | `rag_engine.py` |

### 1.2 Règles de gestion (noms courts)

> Liste des règles métier nommées qui apparaissent dans le code, les commits, les tickets. Le détail vit dans [fonctionnel.md](fonctionnel.md).

| Nom court | Sens | Détail dans |
|---|---|---|
| **Cache hit** | Si un index pour l'URL donnée (identifiée par hash) existe déjà dans `./storage/`, le système le charge sans re-traiter le PDF. | [fonctionnel.md](fonctionnel.md) |
| **Cache miss** | Aucun index trouvé pour l'URL — le PDF est téléchargé, découpé, embarqué et persisté. | [fonctionnel.md](fonctionnel.md) |

### 1.3 États et transitions nommés

> Si le métier a des états explicites (statuts d'une commande, phases d'un workflow), les nommer ici.

| État | Sens | Transitions sortantes |
|---|---|---|
| `NON_INDEXÉ` | Le PDF n'a pas encore été traité — aucun embedding disponible. | → `INDEXÉ` (après premier lancement) |
| `INDEXÉ` | Les embeddings sont persistés dans `./storage/` et prêts à être chargés. | → `EN_REQUÊTE` (à chaque question) |
| `EN_REQUÊTE` | Une question est en cours de traitement — recherche sémantique puis appel Gemini. | → `INDEXÉ` (réponse fournie) |

---

## 2. Acronymes et abréviations

### 2.1 Acronymes métier

| Acronyme | Signification | Confiance | Origine |
|---|---|---|---|
| **RAG** | Retrieval-Augmented Generation | ✓ | README.md |
| **LLM** | Large Language Model | ✓ | README.md |
| **PDF** | Portable Document Format | ✓ | README.md, `pdf_loader.py` |

### 2.2 Acronymes techniques

| Acronyme | Signification | Domaine |
|---|---|---|
| **CLI** | Command-Line Interface | Architecture / interface utilisateur |
| **API** | Application Programming Interface | Intégration Gemini / llama-index |
| **k** | Nombre de chunks les plus proches retournés (top-k) | Recherche vectorielle |

---

## 3. Conventions de nommage — Code

### 3.1 Packages / modules

```
<module>.py  — fichiers Python plats à la racine du projet (pas de package hiérarchique)
```

| Segment | Valeurs autorisées | Exemple |
|---|---|---|
| `<module>` | `rag_engine`, `pdf_loader`, `text_processor`, `gemini_client`, `main`, `config` | `rag_engine.py` |

### 3.2 Classes

- **Casse** : PascalCase pour les classes, snake_case pour les fonctions et variables (Python)
- **Suffixes par rôle** :

| Rôle | Suffixe | Exemple |
|---|---|---|
| Moteur / orchestrateur | `Engine` | `RAGEngine` (à confirmer) |
| Chargeur de données | `Loader` | `PDFLoader` (à confirmer) |
| Processeur / transformateur | aucun ou `Processor` | `TextProcessor` (à confirmer) |
| Client externe | `Client` | `GeminiClient` (à confirmer) |
| Exception | `Exception` ou `Error` | (à confirmer) |

### 3.3 Méthodes et fonctions

- **Casse** : snake_case (Python)
- **Verbes d'action** : `load_*`, `process_*`, `query_*`, `index_*`, `get_*`, `is_*`
- **Booléens** : préfixe `is_` / `has_` / `can_` ; jamais de double négation
- **Async** : suffixe `*Async` (TypeScript) ou retourne `Mono<>` / `Future<>` (typage suffit)

### 3.4 Variables et constantes

- **Variables locales** : snake_case — courtes mais explicites, pas d'abréviations cryptiques
- **Constantes** : UPPER_SNAKE_CASE (ex. `GOOGLE_API_KEY` dans `config.py`)
- **Identifiants** : conserver le terme métier (ex. `pdf_url` plutôt que `url` si le concept est bien un PDF)

### 3.5 Tests

| Type | Pattern de classe | Pattern de méthode |
|---|---|---|
| Unitaire | `<ClasseTestée>Test` | `should<Comportement>When<Condition>` |
| Intégration | `<ClasseTestée>IT` ou `*IntegrationTest` | idem |
| Contract | `<Interface>ContractTest` | idem |
| End-to-end | `<Parcours>E2ETest` | `<scénario fonctionnel>` |

### 3.6 Fichiers, branches, commits

- **Fichiers** : `snake_case.py` pour les modules Python, `kebab-case.md` pour la documentation
- **Branches** : `feat/...`, `fix/...`, `chore/...` (à confirmer)
- **Commits** : Conventional Commits — `feat(scope): description` (à confirmer)

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

- **Casse** : snake_case (à confirmer — pas de base relationnelle identifiée dans le projet actuel)
- **Stockage** : répertoire `./storage/` sur le système de fichiers local (index vectoriel llama-index)
- **Nommage des répertoires de cache** : basé sur le hash de l'URL du PDF source (à confirmer)

---

## 5. Conventions de nommage — Interfaces

### 5.1 Endpoints HTTP

- **Casse** : `kebab-case` dans les paths
- **Pluriel pour les collections** : `/orders`, `/orders/{id}`
- **Verbes** : à éviter dans les paths sauf actions hors CRUD (`/orders/{id}:cancel`)
- **Versioning** : (à confirmer — pas d'API HTTP exposée dans la version actuelle)

### 5.2 Topics / queues

```
(à confirmer — pas de messaging asynchrone identifié dans la version actuelle)
```

| Segment | Valeurs |
|---|---|
| (aucun segment défini) | (à confirmer) |

### 5.3 Schémas d'événements

- **Champs systématiques** : `id`, `event_type`, `schema_version`, `occurred_at`, `producer`
- **Casse champs** : snake_case (à confirmer)
- **Versioning** : champ `schema_version` (entier monotone)

---

## 6. Vocabulaire technique

### 6.1 Stack

| Terme | Définition | Lien |
|---|---|---|
| **llama-index** | Framework d'indexation et de requêtage RAG — orchestre le découpage, l'embedding et la recherche vectorielle. | [llamaindex.ai](https://www.llamaindex.ai) |
| **HuggingFace sentence-transformers** | Bibliothèque de modèles d'embedding sémantique utilisée pour vectoriser les chunks. | [huggingface.co](https://huggingface.co/sentence-transformers) |
| **Google Gemini** | LLM utilisé pour la génération des réponses à partir du contexte récupéré. | [ai.google.dev](https://ai.google.dev) |
| **pypdf** | Bibliothèque Python d'extraction de texte depuis des fichiers PDF. | [pypdf.readthedocs.io](https://pypdf.readthedocs.io) |

### 6.2 Patterns

| Pattern | Sens dans ce projet | Où il est appliqué |
|---|---|---|
| **RAG (Retrieval-Augmented Generation)** | Enrichissement de la génération LLM par récupération de contexte pertinent dans un index vectoriel avant l'appel au modèle. | `rag_engine.py` |
| **Cache-aside** | L'index vectoriel est chargé depuis `./storage/` s'il existe (hit), sinon reconstruit et persisté (miss). | `rag_engine.py` |

### 6.3 Démarche / process

| Terme | Définition |
|---|---|
| **Conventional Commits** | Convention de messages de commit — `feat(scope): description` (à confirmer) |
| **Code review** | (à confirmer — règles internes non documentées) |
| **ADR** | Architecture Decision Record — décisions traçables, voir (à confirmer) |

---

## 7. Termes à éviter / pièges de vocabulaire

> Termes ambigus, faux amis, ou collisions avec des termes techniques. Cette section évite les confusions répétitives.

| Terme | Pourquoi l'éviter | Préférer |
|---|---|---|
| « search » seul | Ambigu entre recherche vectorielle sémantique et recherche plein-texte classique | « recherche sémantique » ou « semantic search » |
| « model » seul | Collision entre le modèle d'embedding et le LLM de génération | « embedding model » ou « LLM » selon le contexte |

---

## 8. Correspondance / mapping

> À utiliser si le projet a deux vocabulaires en parallèle (ex. ancien système → nouveau, métier → technique, API publique → modèle interne). Sinon, supprimer la section.

| Terme README | Terme code | Note |
|---|---|---|
| « PDF document » | `Document` (llama-index) | Le README parle de PDF ; le code manipule des objets `Document` après extraction. |
| « embeddings » | `VectorStoreIndex` | Les embeddings sont matérialisés dans l'index vectoriel persisté. |
| « storage » | `./storage/` | Répertoire de persistance du cache d'index. |

---

## 9. Questions ouvertes

> Ambiguïtés à lever avec les experts. Une question marquée non résolue depuis longtemps est un signal — voir [index.md §5](index.md#5-signaux-de-dérive-à-surveiller).

| # | Question | Impact | Owner | Ouverte depuis |
|---|---|---|---|---|
| Q1 | Quel modèle HuggingFace sentence-transformer est utilisé par défaut dans `text_processor.py` ? | Cosmétique | (à confirmer) | 2026-05-10 |
| Q2 | Quelle est la valeur de top-k utilisée lors de la recherche vectorielle ? | Cosmétique | (à confirmer) | 2026-05-10 |
| Q3 | Le hash d'URL servant de clé de cache est-il un SHA ou un hash applicatif ? | Cosmétique | (à confirmer) | 2026-05-10 |

---

## Références

- Architecture (où ces termes sont utilisés en pratique) : [architecture.md](architecture.md)
- Modèle de données (conventions colonnes détaillées) : [data-model.md](data-model.md)
- Contrats (conventions interfaces détaillées) : [contracts.md](contracts.md)
- Fonctionnel (règles métier nommées) : [fonctionnel.md](fonctionnel.md)
- ADR : (à confirmer — pas de dossier ADR identifié dans le dépôt)

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

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
| **Mise à jour par** | agent IA |
| **PR de référence** | e6e4931 |
| **Périmètre** | rag_engine, text_processor, pdf_loader, gemini_client |

> Source de vérité du vocabulaire projet. Termes marqués **(à confirmer)** = déductions à valider. Marqués **(legacy)** = conservés pour compatibilité, à ne pas reproduire.

---

## 1. Vocabulaire métier

### 1.1 Concepts centraux

| Terme | Définition | Représentation code | Apparaît dans |
|---|---|---|---|
| **RAG** | Retrieval-Augmented Generation — paradigme combinant recherche sémantique dans une base de documents et génération de texte par un LLM pour répondre à des questions. | `RAGEngine` (`rag_engine.py`) | `rag_engine.py`, `main.py`, `README.md` |
| **Index vectoriel** | Représentation numérique d'un corpus de documents sous forme de vecteurs d'embeddings, permettant la recherche par similarité sémantique. | `VectorStoreIndex` (LlamaIndex) | `rag_engine.py` |
| **Chunk / nœud** | Segment de texte produit par découpage d'un document PDF, unité élémentaire indexée et récupérée lors d'une requête. | `SentenceSplitter`, `chunk_size=512` | `text_processor.py` |
| **Embedding** | Vecteur numérique de haute dimension représentant le sens d'un texte, produit par un modèle de langue (ici BGE via HuggingFace). | `HuggingFaceEmbedding` | `text_processor.py` |
| **Query engine** | Composant qui orchestre la récupération des chunks pertinents et la génération de la réponse finale par le LLM. | `self.query_engine` (`RAGEngine`) | `rag_engine.py` |

### 1.2 Règles de gestion (noms courts)

> Liste des règles métier nommées qui apparaissent dans le code, les commits, les tickets. Le détail vit dans [fonctionnel.md](fonctionnel.md).

| Nom court | Sens | Détail dans |
|---|---|---|
| **Cache par hash URL** | Les embeddings d'un PDF sont persistés sur disque sous un chemin dérivé du MD5 de l'URL ; les runs suivants chargent l'index sans regénérer les vecteurs. | [fonctionnel.md](fonctionnel.md) |
| **Top-k retrieval** | Lors d'une requête, les 3 chunks les plus proches sémantiquement sont récupérés (`similarity_top_k=3`) et transmis au LLM. | [fonctionnel.md](fonctionnel.md) |

### 1.3 États et transitions nommés

> Ce projet est un pipeline batch/interactif sans machine à états explicite — section non applicable.

---

## 2. Acronymes et abréviations

### 2.1 Acronymes métier

| Acronyme | Signification | Confiance | Origine |
|---|---|---|---|
| **RAG** | Retrieval-Augmented Generation | ✓ | README.md, `rag_engine.py` |
| **PDF** | Portable Document Format — format source des documents indexés | ✓ | `pdf_loader.py` |

### 2.2 Acronymes techniques

| Acronyme | Signification | Domaine |
|---|---|---|
| **LLM** | Large Language Model | IA générative |
| **BGE** | BAAI General Embedding (`BAAI/bge-small-en-v1.5`) | HuggingFace / embeddings |
| **CLI** | Command-Line Interface | Interface utilisateur |
| **API** | Application Programming Interface | Google Gemini |

---

## 3. Conventions de nommage — Code

### 3.1 Packages / modules

```
<responsabilité>.py  (modules plats à la racine du projet)
```

| Fichier | Responsabilité |
|---|---|
| `rag_engine.py` | Orchestration du pipeline RAG |
| `text_processor.py` | Configuration embeddings et découpage |
| `pdf_loader.py` | Chargement et extraction PDF |
| `gemini_client.py` | Initialisation du LLM Gemini |
| `main.py` | Point d'entrée CLI |

### 3.2 Classes

- **Casse** : PascalCase
- **Suffixes par rôle** :

| Rôle | Suffixe | Exemple |
|---|---|---|
| Moteur / orchestrateur | `Engine` | `RAGEngine` |
| Autres rôles | aucun suffixe imposé (à confirmer) | — |

### 3.3 Méthodes et fonctions

- **Casse** : snake_case
- **Verbes d'action** : `load_*`, `create_*`, `setup_*`, `initialize_*`, `query`
- **Booléens** : préfixe `is_` / `has_` / `can_` ; jamais de double négation
- **Async** : non applicable — pipeline synchrone

### 3.4 Variables et constantes

- **Variables locales** : snake_case — courtes mais explicites, pas d'abréviations cryptiques
- **Constantes** : UPPER_SNAKE_CASE (ex. `GOOGLE_API_KEY` dans `config.py`)
- **Énumérations** : non utilisées dans le projet actuel
- **Identifiants** : conserver le terme métier (ex. `pdf_url`, `embed_model`, `query_engine`)

### 3.5 Tests

| Type | Pattern de fichier | Pattern de fonction |
|---|---|---|
| Unitaire | `test_<module>.py` | `test_<comportement>` |

Exemple existant : `tests/test_gemini_client.py`

### 3.6 Fichiers, branches, commits

- **Fichiers** : `snake_case.py` pour le code Python, `kebab-case.md` pour la documentation
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

> Ce projet n'utilise pas de base de données relationnelle. Le stockage est un répertoire `./storage/<hash_url>/` contenant les fichiers de persistance LlamaIndex.

---

## 5. Conventions de nommage — Interfaces

> Ce projet n'expose pas d'API HTTP et ne publie pas d'événements — §5 non applicable dans l'état actuel du code.

---

## 6. Vocabulaire technique

### 6.1 Stack

| Terme | Définition | Lien |
|---|---|---|
| **LlamaIndex** | Framework d'orchestration RAG : chargement, découpage, indexation, requêtage | [`rag_engine.py`](../rag_engine.py) |
| **HuggingFaceEmbedding** | Modèle d'embedding local (BGE) utilisé pour vectoriser les chunks | [`text_processor.py`](../text_processor.py) |
| **SentenceSplitter** | Node parser LlamaIndex découpant le texte en chunks de 512 tokens avec chevauchement de 50 | [`text_processor.py`](../text_processor.py) |
| **Gemini** | LLM Google (`gemini-2.5-flash`) utilisé pour la génération de réponses | [`gemini_client.py`](../gemini_client.py) |
| **PDFReader** | Lecteur LlamaIndex extrayant le contenu textuel d'un fichier PDF local | [`pdf_loader.py`](../pdf_loader.py) |
| **VectorStoreIndex** | Index LlamaIndex stockant et interrogeant les vecteurs d'embeddings | [`rag_engine.py`](../rag_engine.py) |

### 6.2 Patterns

| Pattern | Sens dans ce projet | Où il est appliqué |
|---|---|---|
| **Pipeline séquentiel** | Enchaînement linéaire : téléchargement PDF → découpage → embedding → stockage → requête | `RAGEngine.__init__` |
| **Cache sur disque par hash** | Persistance des embeddings sous `./storage/<md5(url)>/` pour éviter le recalcul | `RAGEngine._get_index_path`, `_save_index`, `_load_index` |

### 6.3 Démarche / process

| Terme | Définition |
|---|---|
| **Conventional Commits** | Convention de messages de commit (`feat(scope): description`) — utilisée dans ce dépôt (à confirmer) |
| **ADR** | Architecture Decision Record — décisions traçables (à confirmer — dossier non encore créé) |

---

## 7. Termes à éviter / pièges de vocabulaire

> Termes ambigus, faux amis, ou collisions avec des termes techniques. Cette section évite les confusions répétitives.

| Terme | Pourquoi l'éviter | Préférer |
|---|---|---|
| « document » | Ambigu entre le fichier PDF source et l'objet `Document` LlamaIndex (chunk extrait) | Préciser « fichier PDF » ou « document LlamaIndex » selon le contexte |
| « modèle » | Collision entre le modèle LLM (Gemini) et le modèle d'embedding (BGE) | `llm` / `embed_model` comme dans le code |

---

## 8. Correspondance / mapping

> Mapping entre le vocabulaire utilisateur et les termes techniques du code.

| Vocabulaire utilisateur | Code / LlamaIndex | Note |
|---|---|---|
| « document / fichier PDF » | `pdf_url`, fichier téléchargé | Source brute |
| « passage / extrait » | `Node` / chunk (`SentenceSplitter`) | Unité d'indexation |
| « recherche » | `similarity_top_k=3` sur `VectorStoreIndex` | Recherche cosinus sur embeddings |
| « réponse » | sortie de `query_engine.query()` | Générée par Gemini |

---

## 9. Questions ouvertes

> Ambiguïtés à lever avec les experts. Une question marquée non résolue depuis longtemps est un signal — voir [index.md §5](index.md#5-signaux-de-dérive-à-surveiller).

| # | Question | Impact | Owner | Ouverte depuis |
|---|---|---|---|---|
| Q1 | Le projet utilisera-t-il une interface HTTP/API à terme ? (impacte §5) | Cosmétique | (à confirmer) | 2026-05-20 |
| Q2 | Existe-t-il un dossier ADR prévu ? | Cosmétique | (à confirmer) | 2026-05-20 |

---

## Références

- Architecture (où ces termes sont utilisés en pratique) : [architecture.md](architecture.md)
- Modèle de données (conventions colonnes détaillées) : [data-model.md](data-model.md)
- Contrats (conventions interfaces détaillées) : [contracts.md](contracts.md)
- Fonctionnel (règles métier nommées) : [fonctionnel.md](fonctionnel.md)
- ADR : (à confirmer — dossier non encore créé)

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

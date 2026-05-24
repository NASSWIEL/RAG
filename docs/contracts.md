<!--
TEMPLATE — Contrats d'interface
================================
Public cible : (1) une IA qui doit appeler / modifier une interface sans relire toutes les
définitions OpenAPI / proto / handler, (2) un humain qui prend le projet.

Ce document décrit TOUS les points d'interaction du système : APIs exposées, APIs consommées,
événements (in / out), commandes CLI, jobs batch, fichiers d'entrée/sortie. C'est le « plan
de vol » des entrées/sorties — toute requête qui entre, toute donnée qui sort, doit y figurer.

Garde-fous :
- Toute signature listée doit pointer vers le code authoritatif (handler, contrôleur,
  schéma OpenAPI, fichier proto). Si pas de pointeur précis, marquer "(à confirmer)".
- Codes d'erreur listés DOIVENT être couverts par au moins un test ; sinon le signaler.
- Le format des payloads est résumé ici, le détail vit dans les schémas (lien §0).

Bloc « Mode d'emploi » en fin de fichier.
-->

# Contrats d'interface — RAG

| Champ | Valeur |
|---|---|
| **Dernière mise à jour** | 2026-05-24 |
| **Mise à jour par** | agent IA |
| **PR de référence** | (à confirmer) |
| **Type d'interfaces** | Appels natifs Python (lib pure) |
| **Schémas authoritatifs** | Docstrings inline — `gemini_client.py`, `pdf_loader.py`, `rag_engine.py`, `text_processor.py` |

> **Résumé en 1 phrase** : Le système expose une API Python native (`RAGEngine`) qui charge un PDF depuis une URL, construit un index vectoriel via LlamaIndex + HuggingFace, et répond à des questions en langage naturel via le LLM Gemini.

---

## 0. Inventaire global

| Type d'interface | Nombre | Schéma authoritatif |
|---|---|---|
| **Endpoints HTTP exposés** | 0 | — lib Python pure, aucun endpoint HTTP |
| **Endpoints HTTP consommés** | 1 | `load_pdf_from_url` → URL PDF quelconque (GET, timeout 30 s) |
| **Topics / queues produits** | 0 | — |
| **Topics / queues consommés** | 0 | — |
| **Commandes CLI** | 0 | — |
| **Jobs / batchs** | 0 | — |
| **Méthodes / fonctions publiques Python** | 6 | Voir §2 |
| **Fichiers E/S (formats fixes)** | 1 | PDF en entrée — voir §9 |

---

## 1. Conventions transverses des interfaces

### 1.1 Authentification et autorisation

| Type d'interface | Mécanisme | Token / claims | Refresh / rotation |
|---|---|---|---|
| **API publique** | — (lib Python pure, pas d'exposition réseau) | — | — |
| **API interne (Gemini)** | Variable d'environnement `GOOGLE_API_KEY` | Clé API Google | (à confirmer) |
| **Événements** | — | — | — |
| **CLI / Batch** | — | — | — |

### 1.2 En-têtes / métadonnées standards

| En-tête / champ | Rôle | Direction | Obligatoire ? |
|---|---|---|---|
| `X-Request-Id` | Corrélation | I/O | Non applicable — lib Python pure |
| `traceparent` | Tracing W3C | I/O | Non applicable |
| `Idempotency-Key` | Idempotence | I (mutations) | Non applicable |
| `Accept-Language` | i18n | I | Non applicable |

### 1.3 Codes d'erreur normalisés

> Catégoriser. Tout endpoint doit s'aligner sur ce barème — les déviations sont listées dans le bloc de l'endpoint et tracées comme dette.

| Catégorie | HTTP | Code applicatif | Sens | Idempotent ? |
|---|---|---|---|---|
| Succès | 200 / 201 / 204 | `OK` | — | — |
| Validation | 400 | `INVALID_INPUT` | Schéma / contrainte côté client | Oui |
| Auth manquante | 401 | `UNAUTHENTICATED` | Pas de credential | Oui |
| Auth refusée | 403 | `FORBIDDEN` | Credential OK mais droit absent | Oui |
| Ressource absente | 404 | `NOT_FOUND` | — | Oui |
| Conflit | 409 | `CONFLICT` | État incompatible | — |
| Trop de requêtes | 429 | `RATE_LIMITED` | Avec `Retry-After` | Oui |
| Erreur serveur | 500 | `INTERNAL_ERROR` | Bug ou état incohérent | — |
| Service tiers KO | 502 / 503 / 504 | `UPSTREAM_*` | Avec corrélation | — |

### 1.4 Format de réponse d'erreur

```json
{
  "error": {
    "code": "INVALID_INPUT",
    "message": "human-readable",
    "details": [
      { "field": "email", "rule": "format" }
    ],
    "trace_id": "..."
  }
}
```

> Non applicable — lib Python pure ; les erreurs sont levées comme exceptions Python standard.

### 1.5 Versioning

- **API HTTP** : Non applicable.
- **Événements** : Non applicable.
- **Politique de breaking change** : (à confirmer) — aucune politique formelle définie à ce stade.

---

## 2. API publique Python (appels natifs)

> Ce système est une lib Python pure — pas d'endpoint HTTP. Les contrats ci-dessous décrivent les fonctions et méthodes publiques appelables directement par l'import.

### 2.1 `RAGEngine` — moteur principal RAG

**Code source** : [rag_engine.py](../rag_engine.py)

#### `RAGEngine.__init__(pdf_url, storage_dir="./storage")` — initialisation et indexation

| Méta | Valeur |
|---|---|
| **Code source** | [rag_engine.py:16](../rag_engine.py) |
| **Auth requise** | Variable d'env `GOOGLE_API_KEY` (transmise à Gemini) |
| **Idempotent** | Oui — si l'index est déjà en cache (`storage_dir`), il est rechargé sans re-téléchargement |
| **Effets de bord** | Télécharge le PDF, écrit l'index vectoriel dans `storage_dir/index_<md5(pdf_url)>/` |

| Paramètre | Type | Obligatoire | Description |
|---|---|---|---|
| `pdf_url` | `str` | Oui | URL publique d'un fichier PDF à indexer |
| `storage_dir` | `str` | Non (défaut `"./storage"`) | Répertoire local de persistance de l'index vectoriel |

**Retour** : instance `RAGEngine` prête à l'emploi.

**Pièges / particularités**

- ⚠️ Si `GOOGLE_API_KEY` est absente de l'environnement, un `KeyError` est levé dès l'init.
- ⚠️ Le cache est indexé par MD5 de `pdf_url` — changer l'URL force un re-indexage complet.

---

#### `RAGEngine.query(question) -> str` — question en langage naturel

| Méta | Valeur |
|---|---|
| **Code source** | [rag_engine.py:70](../rag_engine.py) |
| **Idempotent** | Oui (lecture seule) |
| **SLO p95** | (à confirmer) |

| Paramètre | Type | Obligatoire | Description |
|---|---|---|---|
| `question` | `str` | Oui | Question en langage naturel à poser sur le document indexé |

**Retour** : `str` — réponse générée par le pipeline RAG (retrieval top-k=3, mode `compact`).

---

### 2.2 `gemini_client` — initialisation du LLM

**Code source** : [gemini_client.py](../gemini_client.py)

#### `initialize_gemini_llm() -> Gemini` — configure et retourne le LLM Gemini

| Méta | Valeur |
|---|---|
| **Code source** | [gemini_client.py:9](../gemini_client.py) |
| **Auth requise** | `GOOGLE_API_KEY` (env) |
| **Effets de bord** | Écrit `Settings.llm` dans le singleton global LlamaIndex |

**Paramètres** : aucun.

**Retour** : instance `Gemini` configurée (`model="models/gemini-2.5-flash"`, `temperature=0.1`).

- ⚠️ Modifie le singleton global `llama_index.core.Settings.llm` — tout appel ultérieur à LlamaIndex sans instanciation explicite utilisera ce LLM.

---

### 2.3 `pdf_loader` — chargement de PDF

**Code source** : [pdf_loader.py](../pdf_loader.py)

#### `load_pdf_from_url(url) -> str` — télécharge un PDF et retourne le chemin local

| Paramètre | Type | Obligatoire | Description |
|---|---|---|---|
| `url` | `str` | Oui | URL HTTP(S) du PDF à télécharger |

**Retour** : `str` — chemin local du fichier temporaire (`tempfile.gettempdir()/temp_rag_document.pdf`).

**Effets de bord** : requête GET avec timeout 30 s ; écrit dans le répertoire temporaire système.

- ⚠️ Le fichier temporaire est toujours nommé `temp_rag_document.pdf` — appels parallèles s'écrasent mutuellement.

---

#### `load_documents_from_pdf(pdf_path) -> list` — parse un PDF local

| Paramètre | Type | Obligatoire | Description |
|---|---|---|---|
| `pdf_path` | `str` | Oui | Chemin local vers le fichier PDF |

**Retour** : `list` de documents LlamaIndex extraits par `PDFReader`.

---

### 2.4 `text_processor` — traitement du texte et parsing

**Code source** : [text_processor.py](../text_processor.py)

#### `setup_advanced_text_processing() -> HuggingFaceEmbedding` — configure les embeddings

**Paramètres** : aucun.

**Retour** : instance `HuggingFaceEmbedding` (`model_name="BAAI/bge-small-en-v1.5"`).

**Effets de bord** : écrit `Settings.embed_model`, `Settings.chunk_size=512`, `Settings.chunk_overlap=50` dans le singleton global LlamaIndex.

---

#### `create_node_parser() -> SentenceSplitter` — crée le parseur de nœuds

**Paramètres** : aucun.

**Retour** : `SentenceSplitter(chunk_size=512, chunk_overlap=50)`.

---

## 3. APIs consommées

> Une ligne par dépendance. Le détail des contrats vit chez le fournisseur — ici on note ce dont on dépend et comment on s'en protège.

| Service | Endpoint | Type | Criticité | Timeout | Retry | Circuit breaker |
|---|---|---|---|---|---|---|
| Google Gemini API | SDK `llama_index.llms.gemini` | Sync | Bloquante | (à confirmer) | (à confirmer) | Aucun |
| URL PDF source | `GET <pdf_url>` | Sync HTTP | Bloquante (init) | 30 s | Aucun | Aucun |

**Modes dégradés** : En cas d'échec du téléchargement PDF ou d'appel Gemini, une exception Python est propagée — aucun mécanisme de fallback ou de retry n'est implémenté à ce stade (à confirmer).

---

## 4. Événements / messages

> Non applicable — ce système est une lib Python pure et ne produit ni ne consomme aucun topic ou queue de messages.

---

## 5. Commandes CLI

> Non applicable — aucune interface CLI exposée.

---

## 6. Jobs / batchs

> Non applicable — aucun job ou batch planifié.

---

## 7. Mode d'appel inter-composants (interne)

Les modules s'appellent directement dans le même processus Python. Le flux canonique est :

```python
# Appel typique depuis le code consommateur
from rag_engine import RAGEngine

engine = RAGEngine(pdf_url="https://example.com/doc.pdf", storage_dir="./storage")
answer = engine.query("Quelle est la conclusion du document ?")
```

**Flux interne lors de `RAGEngine.__init__`** :

1. `text_processor.setup_advanced_text_processing()` → configure `Settings.embed_model`
2. `gemini_client.initialize_gemini_llm()` → configure `Settings.llm`
3. `text_processor.create_node_parser()` → configure `Settings.node_parser`
4. Si pas de cache : `pdf_loader.load_pdf_from_url(pdf_url)` → `pdf_loader.load_documents_from_pdf(path)` → `VectorStoreIndex.from_documents()`
5. Si cache présent : `StorageContext.from_defaults()` → `load_index_from_storage()`

**Points-clés** :

- Aucune propagation de contexte de corrélation — lib synchrone monothread.
- Aucune instrumentation (métriques, spans) à ce stade (à confirmer).
- Les erreurs propagent comme exceptions Python non typées.

---

## 8. Matrice d'appels

> Lignes = appelants. Colonnes = appelés. Cellule = mode d'appel. Vide = aucun appel.
> Légende : ✓ = inconditionnel · (c) = conditionnel (préciser dans le bloc concerné).

| Appelant ↓ / Appelé → | `gemini_client` | `pdf_loader` | `text_processor` | Gemini API | URL PDF |
|---|---|---|---|---|---|
| **`RAGEngine`** | ✓ | ✓ | ✓ | — | — |
| **`gemini_client`** | — | — | — | ✓ | — |
| **`pdf_loader`** | — | — | — | — | ✓ |
| **`text_processor`** | — | — | — | — | — |

> Une ligne / colonne vide = composant terminal ou racine. C'est une information.

---

## 9. Formats des E/S physiques

> Pour tout fichier ou flux à structure fixe lu/écrit par le système.

### 9.1 `<pdf_url>` — fichier PDF en entrée

- **Type** : PDF (binaire)
- **Encodage** : binaire ; contenu textuel extrait par `PDFReader` (LlamaIndex)
- **Délimiteur** : N/A
- **Taille typique** : (à confirmer)

| Champ logique | Type | Description |
|---|---|---|
| Contenu textuel | `str` (extrait) | Pages PDF converties en texte par `PDFReader`, segmentées en chunks de 512 tokens avec overlap 50 |

---

## 10. Transactions, commits, rollback

| Composant | Scope transactionnel | Commit | Rollback |
|---|---|---|---|
| `RAGEngine.__init__` (écriture index) | Aucun — écriture fichiers locaux | Implicite (fin d'écriture disque) | Aucun — répertoire partiel possible si interruption |
| `RAGEngine.query` | Aucun (lecture seule) | N/A | N/A |

> Cas critiques : si `RAGEngine.__init__` est interrompu pendant `_save_index`, le répertoire de cache peut être incomplet. Au prochain lancement, `os.path.exists(index_path)` peut renvoyer `True` sur un index corrompu (à confirmer).

---

## 11. Contraintes non-fonctionnelles par interface

| Interface | Latence p95 cible | Disponibilité | Idempotence | Re-entrance |
|---|---|---|---|---|
| `RAGEngine.__init__` | (à confirmer) | (à confirmer) | Oui — cache MD5 sur `pdf_url` | Non — ne pas appeler en parallèle sur le même `storage_dir` |
| `RAGEngine.query` | (à confirmer) | Dépend de Gemini API | Oui (lecture seule) | Oui |
| `load_pdf_from_url` | ≤ 30 s (timeout) | Dépend de la source | Non — écrase toujours le même fichier temp | Non |

---

## 12. Tests de contrat

| Interface | Outil | Localisation | Côté |
|---|---|---|---|
| API Python publique | pytest | [tests/](../tests/) | Provider (à confirmer) |

---

## 13. Recommandations actives

1. **Ajouter un mécanisme de retry sur `load_pdf_from_url`** — actuellement une seule tentative avec timeout 30 s ; les PDF volumineux ou les réseaux instables entraînent un `ConnectionError` non récupéré (à confirmer).
2. **Valider l'intégrité du cache avant chargement** — `_load_index` ne vérifie pas que le répertoire est complet ; un index corrompu produit une erreur opaque de LlamaIndex (à confirmer).
3. **Ajouter `GOOGLE_API_KEY` au message d'erreur** — le `KeyError` brut n'indique pas à l'utilisateur quelle variable est manquante.

---

## Références

- Architecture : [architecture.md](architecture.md)
- Modèle de données (effets de bord en base) : [data-model.md](data-model.md)
- Glossaire : [glossaire.md](glossaire.md)
- Règles fonctionnelles applicables aux endpoints : [fonctionnel.md](fonctionnel.md)

---

<!--
MODE D'EMPLOI DU TEMPLATE
=========================

POUR L'IA QUI MET À JOUR CE FICHIER

Déclencheurs (mettre à jour les sections concernées si la PR touche…) :

| Modification dans la PR | Sections à relire |
|---|---|
| Nouvel endpoint HTTP / gRPC / GraphQL | §0, §2 (nouveau bloc) |
| Modification de signature endpoint | §2 (bloc concerné), §1.3 si nouveau code d'erreur |
| Nouvel appel à un service externe | §0, §3 |
| Nouveau topic / event produit | §0, §4.1 |
| Nouveau topic / event consommé | §0, §4.2 |
| Nouvelle commande CLI | §0, §5 |
| Nouveau job batch | §0, §6 |
| Nouveau format de fichier E/S | §0, §9 |
| Changement de mécanisme d'auth | §1.1 |
| Nouvel en-tête transverse | §1.2 |
| Nouveau code d'erreur normalisé | §1.3, §1.4 si format évolue |
| Nouvel appel inter-composants interne | §7, §8 (matrice) |
| SLO / timeout / retry modifié | §3 (si externe), §11 (si exposé) |
| Nouveau test de contrat | §12 |

Auto-checks :
- [ ] Chaque endpoint §2 pointe vers un fichier source réel.
- [ ] Les codes de §1.3 sont effectivement émis quelque part dans le code.
- [ ] La matrice §8 ne mentionne aucun composant supprimé.
- [ ] Tout topic §4 a un schéma référencé.
- [ ] Aucune (à confirmer) > 60 jours sans ticket.

POUR LE RELECTEUR HUMAIN

- Vérifier la cohérence avec OpenAPI / proto : si écart, signaler dans la PR.
- Les SLO §11 doivent provenir d'un dashboard ou d'un ADR, pas être inventés.
- Si la matrice §8 devient illisible (trop de colonnes), envisager un découpage par domaine.

POUR ADAPTER À UN AUTRE PROJET

1. Si le système n'expose RIEN au monde extérieur (lib pure) : §1, §2, §3 deviennent
   minces ; §7 (appels internes) devient le cœur du document.
2. Si le système est PUREMENT batch : déplacer §6 en début de document, §2 et §3 disparaissent.
3. Si le système est event-only : §4 devient le cœur, structurer par flux.
4. Garder le pattern « inventaire global §0 + détails ensuite » même si certaines catégories
   sont vides : « 0 endpoints exposés » est une information utile.
5. Pour les systèmes critiques en sécurité, dédoubler §2 par niveau de criticité (publique
   non authentifiée / authentifiée / interne / admin).
-->

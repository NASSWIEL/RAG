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
| **Dernière mise à jour** | 2026-05-21 |
| **Mise à jour par** | agent IA |
| **PR de référence** | (à confirmer) |
| **Type d'interfaces** | Appels natifs Python (lib pure — CLI interactif) |
| **Schémas authoritatifs** | Docstrings inline dans [`rag_engine.py`](../rag_engine.py), [`gemini_client.py`](../gemini_client.py), [`pdf_loader.py`](../pdf_loader.py), [`text_processor.py`](../text_processor.py) |

> **Résumé en 1 phrase** : Le système expose une interface Python native (`RAGEngine`) qui charge un PDF depuis une URL, construit un index vectoriel, et répond à des questions en langage naturel via Gemini ; aucun endpoint HTTP ni événement n'est exposé.

---

## 0. Inventaire global

| Type d'interface | Nombre | Schéma authoritatif |
|---|---|---|
| **Endpoints HTTP exposés** | 0 | — |
| **Endpoints HTTP consommés** | 1 (PDF via URL + Gemini API) | [pdf_loader.py](../pdf_loader.py), [gemini_client.py](../gemini_client.py) |
| **Topics / queues produits** | 0 | — |
| **Topics / queues consommés** | 0 | — |
| **Commandes CLI** | 1 (`python main.py`) | [main.py](../main.py) |
| **Jobs / batchs** | 0 | — |
| **Méthodes publiques (lib native)** | 8 (`RAGEngine.__init__`, `RAGEngine.query`, `initialize_gemini_llm`, `get_llm`, `generate_text`, `summarize`, `rerank_passages`, `reset_llm`) | [§7](#7-mode-dappel-inter-composants-interne) |
| **Fichiers E/S (formats fixes)** | 1 (PDF en entrée, stockage index sur disque) | [§9](#9-formats-des-es-physiques) |

---

## 1. Conventions transverses des interfaces

### 1.1 Authentification et autorisation

| Type d'interface | Mécanisme | Token / claims | Refresh / rotation |
|---|---|---|---|
| **API Gemini (consommée)** | API key via variable d'environnement `GOOGLE_API_KEY` | Clé secrète, pas de claims JWT | Rotation manuelle |
| **Téléchargement PDF** | Aucune — URL publique (arxiv.org par défaut) | — | — |
| **CLI** | Aucune auth — lancement local | — | — |

### 1.2 En-têtes / métadonnées standards

| En-tête / champ | Rôle | Direction | Obligatoire ? |
|---|---|---|---|
| `X-Request-Id` | Corrélation | I/O | Non applicable (pas d'HTTP exposé) |
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

> Non applicable : le système est une lib Python native sans HTTP. Les erreurs sont propagées via exceptions Python standard (ex. `requests.exceptions.RequestException`, `KeyError` si `GOOGLE_API_KEY` absent).

### 1.5 Versioning

- **API HTTP** : non applicable.
- **Événements** : non applicable.
- **Interface native** : pas de versioning formel défini. Le modèle Gemini utilisé est fixé à `models/gemini-2.5-flash` dans [`gemini_client.py`](../gemini_client.py) — tout changement de modèle est un breaking change potentiel.
- **Politique de breaking change** : (à confirmer)

---

## 2. APIs exposées (HTTP / gRPC / GraphQL)

> Ce système n'expose aucun endpoint HTTP, gRPC ni GraphQL. Voir §7 pour les méthodes publiques Python.

---

## 3. APIs consommées

| Service | Endpoint | Type | Criticité | Timeout | Retry | Circuit breaker |
|---|---|---|---|---|---|---|
| Gemini (Google AI) | `models/gemini-2.5-flash` via SDK `llama_index.llms.gemini` | Sync | Bloquante — sans LLM, aucune réponse possible | (à confirmer) | Aucun défini | Aucun défini |
| Source PDF (URL arbitraire) | URL configurée dans `main.py` — `https://arxiv.org/pdf/2005.11401.pdf` par défaut | Sync HTTP GET | Bloquante au premier lancement ; mise en cache locale ensuite | 30 s (défini dans [`pdf_loader.py`](../pdf_loader.py)) | Aucun | Aucun |

**Modes dégradés** : si le téléchargement PDF échoue, une exception `requests` est levée et le programme s'arrête. Si Gemini est indisponible, l'appel échoue sans fallback. Aucun mode dégradé n'est implémenté.

---

## 4. Événements / messages

> Ce système ne produit ni ne consomme aucun topic ou queue. Section non applicable.

---

## 5. Commandes CLI

| Commande | Rôle | Code source | Authentif | Effets de bord |
|---|---|---|---|---|
| `python main.py` | Lance la boucle interactive de questions-réponses RAG | [main.py](../main.py) | Variable d'env `GOOGLE_API_KEY` requise | Téléchargement PDF si pas en cache ; écriture de l'index dans `./storage/index_<md5>/` |

---

## 6. Jobs / batchs

> Ce système ne contient aucun job ou batch planifié. Section non applicable.

---

## 7. Mode d'appel inter-composants (interne)

Le cœur du système est la classe `RAGEngine`. Les deux méthodes publiques constituent l'interface principale.

### `RAGEngine.__init__(pdf_url, storage_dir="./storage")`

| Paramètre | Type | Obligatoire | Description |
|---|---|---|---|
| `pdf_url` | `str` | Oui | URL publique du PDF à indexer |
| `storage_dir` | `str` | Non (défaut `./storage`) | Répertoire de persistance de l'index vectoriel |

**Effets de bord** :
- Téléchargement du PDF si non en cache (via `load_pdf_from_url`)
- Construction de l'index vectoriel avec `VectorStoreIndex` + embeddings `BAAI/bge-small-en-v1.5`
- Persistance de l'index dans `<storage_dir>/index_<md5(pdf_url)>/`
- Initialisation du LLM Gemini (`models/gemini-2.5-flash`, `temperature=0.1`)

⚠️ La variable d'environnement `GOOGLE_API_KEY` doit être définie avant l'instantiation, sinon `KeyError` levée.

**Code source** : [`rag_engine.py`](../rag_engine.py)

---

### `RAGEngine.query(question) -> str`

| Paramètre | Type | Obligatoire | Description |
|---|---|---|---|
| `question` | `str` | Oui | Question en langage naturel |

**Retour** : `str` — réponse générée par Gemini à partir des 3 chunks les plus similaires (`similarity_top_k=3`, `response_mode="compact"`).

**Effets de bord** : appel réseau vers l'API Gemini. Aucune écriture disque.

**Code source** : [`rag_engine.py:70`](../rag_engine.py)

---

### Fonctions utilitaires publiques

| Fonction | Module | Signature | Rôle |
|---|---|---|---|
| `initialize_gemini_llm` | [`gemini_client.py`](../gemini_client.py) | `(model: str = "models/gemini-2.5-flash", temperature: float = 0.1, max_tokens: int = 1024) -> Gemini` | Instancie et enregistre le LLM singleton dans `Settings` ; retourne l'instance créée |
| `get_llm` | [`gemini_client.py`](../gemini_client.py) | `() -> Gemini` | Retourne le singleton LLM courant ; appelle `initialize_gemini_llm()` avec les valeurs par défaut si non encore initialisé |
| `generate_text` | [`gemini_client.py`](../gemini_client.py) | `(prompt: str, temperature: float \| None = None) -> str` | Génère une réponse plain-text sans contexte RAG ; `temperature` surcharge temporaire sans remplacer le singleton. Lève `ValueError` si `prompt` vide |
| `summarize` | [`gemini_client.py`](../gemini_client.py) | `(text: str, max_words: int = 150) -> str` | Résume `text` en approximativement `max_words` mots via `generate_text`. Lève `ValueError` si `text` vide |
| `rerank_passages` | [`gemini_client.py`](../gemini_client.py) | `(query: str, passages: list[str]) -> list[str]` | Rerank zero-shot des passages par pertinence à `query` via Gemini ; retourne l'ordre original en cas d'échec de parsing |
| `reset_llm` | [`gemini_client.py`](../gemini_client.py) | `() -> None` | Réinitialise le singleton LLM à `None`, forçant une ré-initialisation au prochain appel |
| `load_pdf_from_url` | [`pdf_loader.py`](../pdf_loader.py) | `(url: str) -> str` | Télécharge le PDF et retourne le chemin local |
| `load_documents_from_pdf` | [`pdf_loader.py`](../pdf_loader.py) | `(pdf_path: str) -> list` | Parse le PDF local en liste de documents LlamaIndex |
| `setup_advanced_text_processing` | [`text_processor.py`](../text_processor.py) | `() -> HuggingFaceEmbedding` | Configure embeddings `BAAI/bge-small-en-v1.5`, `chunk_size=512`, `chunk_overlap=50` |
| `create_node_parser` | [`text_processor.py`](../text_processor.py) | `() -> SentenceSplitter` | Crée un `SentenceSplitter(chunk_size=512, chunk_overlap=50)` |

```python
# Appel canonique
from rag_engine import RAGEngine

rag = RAGEngine("https://arxiv.org/pdf/2005.11401.pdf")
answer = rag.query("What is RAG?")
print(answer)
```

---

## 8. Matrice d'appels

> Légende : ✓ = inconditionnel · (c) = conditionnel

| Appelant ↓ / Appelé → | `pdf_loader` | `gemini_client` | `text_processor` | `VectorStoreIndex` (llama_index) | API Gemini (réseau) |
|---|---|---|---|---|---|
| **`main.py`** | — | — | — | — | — |
| **`RAGEngine.__init__`** | ✓ (c) | ✓ | ✓ | ✓ | — |
| **`RAGEngine.query`** | — | — | — | ✓ | ✓ |
| **`generate_text`** | — | — | — | — | ✓ |
| **`summarize`** | — | ✓ (`generate_text`) | — | — | ✓ (indirect) |
| **`rerank_passages`** | — | ✓ (`generate_text`) | — | — | ✓ (indirect) |

> `pdf_loader` est appelé conditionnellement : uniquement si l'index n'est pas déjà en cache disque.

---

## 9. Formats des E/S physiques

### 9.1 PDF en entrée — source de documents

- **Type** : PDF binaire
- **Encodage** : binaire (décodé par `PDFReader` de llama_index)
- **Taille typique** : (à confirmer) — le PDF de référence est `arxiv.org/pdf/2005.11401.pdf`
- **Parser** : `llama_index.readers.file.PDFReader`

### 9.2 Index vectoriel persisté — cache disque

- **Type** : répertoire `./storage/index_<md5(pdf_url)>/` contenant les fichiers de stockage LlamaIndex
- **Format** : JSON/binaire géré par `StorageContext` de LlamaIndex
- **Écriture** : `index.storage_context.persist(persist_dir=...)` dans [`rag_engine.py`](../rag_engine.py)
- **Lecture** : `StorageContext.from_defaults(persist_dir=...)` + `load_index_from_storage`

⚠️ Si le `pdf_url` change, un nouveau hash MD5 est calculé et un nouvel index est créé. L'ancien index n'est pas supprimé automatiquement.

---

## 10. Transactions, commits, rollback

| Composant | Scope transactionnel | Commit | Rollback |
|---|---|---|---|
| `RAGEngine.__init__` (écriture index) | Écriture disque après indexation complète | `storage_context.persist()` une fois | Aucun — si le process s'interrompt pendant la persistance, l'index partiel peut être corrompu |

> ⚠️ Pas de base de données — le seul état persisté est l'index vectoriel sur disque. Aucune saga ni transaction multi-tables.

---

## 11. Contraintes non-fonctionnelles par interface

| Interface | Latence p95 cible | Disponibilité | Idempotence | Re-entrance |
|---|---|---|---|---|
| `RAGEngine.__init__` (premier lancement) | (à confirmer) — dépend du téléchargement PDF + génération embeddings | Dépend de l'URL source et de l'API Gemini | Oui — l'index est mis en cache par hash d'URL | Non re-entrant |
| `RAGEngine.__init__` (cache chaud) | (à confirmer) — lecture disque uniquement | Locale | Oui | Non re-entrant |
| `RAGEngine.query` | (à confirmer) — dépend de la latence Gemini API | Dépend de l'API Gemini | Non (réponses LLM non déterministes) | Non re-entrant |
| Téléchargement PDF (`requests.get`) | Timeout 30 s | Dépend de l'URL source | Oui | — |

---

## 12. Tests de contrat

| Interface | Outil | Localisation | Côté |
|---|---|---|---|
| `initialize_gemini_llm` | pytest | [`tests/test_gemini_client.py`](../tests/test_gemini_client.py) | Provider (à confirmer) |

> Les autres méthodes publiques (`RAGEngine.query`, `load_pdf_from_url`, etc.) ne semblent pas avoir de tests de contrat dédiés — (à confirmer).

---

## 13. Recommandations actives

1. **Ajouter un retry + timeout sur l'appel Gemini** — actuellement aucune gestion de panne, une erreur réseau transitoire interrompt la session.
2. **Gérer la corruption d'index partiel** — si le processus est tué pendant `storage_context.persist()`, l'index est potentiellement inutilisable au prochain démarrage sans nettoyage manuel.
3. **Externaliser `pdf_url` en paramètre CLI** — actuellement codé en dur dans `main.py`, ce qui rend le programme difficile à réutiliser sans modification du code.

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

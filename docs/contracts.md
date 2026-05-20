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

# Contrats d'interface — RAG (Retrieval-Augmented Generation)

| Champ | Valeur |
|---|---|
| **Dernière mise à jour** | 2026-05-20 |
| **Mise à jour par** | agent IA |
| **PR de référence** | (à confirmer) |
| **Type d'interfaces** | Appels natifs Python (bibliothèque) |
| **Schémas authoritatifs** | Signatures de fonctions dans le code source |

> **Résumé en 1 phrase** : Le système expose une interface Python (`RAGEngine`) pour interroger un PDF via RAG, consomme l'API Gemini (Google) pour la génération, et persiste les embeddings sur disque local.

---

## 0. Inventaire global

| Type d'interface | Nombre | Schéma authoritatif |
|---|---|---|
| **Endpoints HTTP exposés** | 0 | — |
| **Endpoints HTTP consommés** | 1 | [gemini_client.py](../gemini_client.py) (Gemini API via llama-index) |
| **Topics / queues produits** | 0 | — |
| **Topics / queues consommés** | 0 | — |
| **Commandes CLI** | 0 | — (point d'entrée `main.py` non paramétré) |
| **Jobs / batchs** | 0 | — |
| **Interfaces Python publiques** | 10 | [§2](#2-interfaces-python-publiques) |
| **Fichiers E/S (formats fixes)** | 1 | [§9](#9-formats-des-es-physiques) |

---

## 1. Conventions transverses des interfaces

### 1.1 Authentification et autorisation

| Type d'interface | Mécanisme | Token / claims | Refresh / rotation |
|---|---|---|---|
| **API Gemini (consommée)** | API key (`GOOGLE_API_KEY`) via variable de config | Clé lue depuis `config.py` | Manuelle — aucune rotation automatique |
| **Interfaces Python internes** | Aucune — bibliothèque sans auth | — | — |

### 1.2 En-têtes / métadonnées standards

> Non applicable — le système n'expose pas d'interface HTTP. Les appels à l'API Gemini délèguent la gestion des en-têtes à la bibliothèque `llama-index-llms-gemini`.

### 1.3 Codes d'erreur normalisés

> Non applicable — pas d'interface HTTP. Les erreurs sont des exceptions Python non typées. Pas de codes d'erreur applicatifs normalisés.

### 1.4 Format de réponse d'erreur

> Non applicable — pas d'interface HTTP. Les erreurs sont propagées comme exceptions Python standard (ex. `requests.exceptions.RequestException` pour les téléchargements PDF, exceptions llama-index pour les appels Gemini).

### 1.5 Versioning

- **Interfaces Python** : pas de versioning formel — les changements de signature sont des breaking changes directs (à confirmer)
- **Politique de breaking change** : (à confirmer)

---

## 2. Interfaces Python publiques

> Le système est une bibliothèque Python. Les contrats ci-dessous sont les API d'appel public. Pas d'endpoints HTTP exposés.

### 2.1 `RAGEngine` — moteur principal de RAG

**Code source** : [rag_engine.py](../rag_engine.py)

#### `RAGEngine.__init__(pdf_url, storage_dir="./storage")` — initialisation et construction de l'index

| Paramètre | Type | Obligatoire | Description |
|---|---|---|---|
| `pdf_url` | `str` | Oui | URL publique du PDF à indexer |
| `storage_dir` | `str` | Non (défaut `"./storage"`) | Répertoire de persistance des embeddings sur disque |

**Effets de bord**

- Télécharge le PDF depuis `pdf_url` si l'index n'existe pas encore en cache
- Construit un `VectorStoreIndex` (llama-index) avec `similarity_top_k=3`, `response_mode="compact"`
- Initialise le modèle Gemini via `initialize_gemini_llm()` et l'embedding HuggingFace `BAAI/bge-small-en-v1.5`
- Persiste l'index dans `{storage_dir}/index_{md5(pdf_url)}/` si première exécution

**Pièges**

- ⚠️ Le téléchargement PDF écrit dans `tempfile.gettempdir()` avec un nom fixe (`temp_rag_document.pdf`) — appels concurrents avec des URLs différentes s'écrasent mutuellement

#### `RAGEngine.query(question) -> str` — interrogation du document

| Paramètre | Type | Obligatoire | Description |
|---|---|---|---|
| `question` | `str` | Oui | Question en langage naturel |

**Retour** : `str` — réponse générée par Gemini à partir des chunks les plus proches sémantiquement

**Effets de bord**

- Appel réseau sortant vers l'API Gemini (`models/gemini-2.5-flash`)
- Affiche des traces `print` sur stdout

#### `RAGEngine._get_index_path() -> str` — calcul du chemin de cache

> Méthode interne. Retourne `{storage_dir}/index_{md5(pdf_url)}`. Le hash est calculé avec `hashlib.md5(..., usedforsecurity=False)`.

#### `RAGEngine._save_index(index_path)` — persistance de l'index

> Méthode interne. Crée le répertoire et appelle `storage_context.persist(persist_dir=index_path)`.

#### `RAGEngine._load_index(index_path)` — chargement de l'index depuis le cache

> Méthode interne. Reconstruit un `StorageContext` depuis le répertoire et retourne un index llama-index.

---

### 2.2 `pdf_loader` — chargement de documents PDF

**Code source** : [pdf_loader.py](../pdf_loader.py)

#### `load_pdf_from_url(url) -> str` — téléchargement d'un PDF

| Paramètre | Type | Obligatoire | Description |
|---|---|---|---|
| `url` | `str` | Oui | URL HTTP(S) publique du PDF |

**Retour** : `str` — chemin local du fichier temporaire téléchargé

**Effets de bord**

- Requête HTTP GET avec `timeout=30` secondes
- Écrit dans `{tempfile.gettempdir()}/temp_rag_document.pdf` (chemin fixe)

**Pièges**

- ⚠️ Pas de vérification du code HTTP de réponse — une URL retournant 404 écrit quand même un fichier (contenu HTML) sans lever d'exception

#### `load_documents_from_pdf(pdf_path) -> list` — extraction du contenu

| Paramètre | Type | Obligatoire | Description |
|---|---|---|---|
| `pdf_path` | `str` | Oui | Chemin local vers un fichier PDF |

**Retour** : `list[Document]` — liste de documents llama-index (un par page)

---

### 2.3 `gemini_client` — client LLM Gemini

**Code source** : [gemini_client.py](../gemini_client.py)

**Constantes de module**

| Constante | Valeur |
|---|---|
| `_DEFAULT_MODEL` | `"models/gemini-2.5-flash"` |
| `_DEFAULT_TEMPERATURE` | `0.1` |
| `_DEFAULT_MAX_TOKENS` | `1024` |

**Singleton de module** : `_llm_instance: Gemini | None = None` — instance réutilisée par tous les appelants du module.

#### `initialize_gemini_llm(model, temperature, max_tokens) -> Gemini` — création et enregistrement du LLM

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `model` | `str` | `_DEFAULT_MODEL` | Identifiant du modèle Gemini |
| `temperature` | `float` | `_DEFAULT_TEMPERATURE` | Température d'échantillonnage |
| `max_tokens` | `int` | `_DEFAULT_MAX_TOKENS` | Nombre maximum de tokens générés |

**Retour** : instance `Gemini` (llama-index) configurée

**Effets de bord**

- Assigne `Settings.llm` globalement (llama-index global settings)
- Stocke l'instance dans `_llm_instance`
- Lit `GOOGLE_API_KEY` depuis `config.py`

#### `get_llm() -> Gemini` — accès au singleton

**Retour** : `Gemini` — instance active ; appelle `initialize_gemini_llm()` avec les valeurs par défaut si `_llm_instance` est `None`.

#### `generate_text(prompt, temperature) -> str` — génération de texte libre

| Paramètre | Type | Obligatoire | Description |
|---|---|---|---|
| `prompt` | `str` | Oui | Prompt envoyé à Gemini |
| `temperature` | `float \| None` | Non (défaut `None`) | Surcharge temporaire de température |

**Retour** : `str` — texte généré par Gemini

**Erreurs** : `ValueError` si `prompt` est vide ou blanc

**Pièges**

- ⚠️ Si `temperature` est fourni, une instance `Gemini` temporaire est créée sans remplacer le singleton — l'appel est plus coûteux en initialisation

#### `summarize(text, max_words) -> str` — résumé de texte

| Paramètre | Type | Obligatoire | Description |
|---|---|---|---|
| `text` | `str` | Oui | Texte source à résumer |
| `max_words` | `int` | Non (défaut `150`) | Longueur cible du résumé en mots |

**Retour** : `str` — résumé généré via `generate_text`

**Erreurs** : `ValueError` si `text` est vide ou blanc

#### `rerank_passages(query, passages) -> list[str]` — reclassement de passages

| Paramètre | Type | Obligatoire | Description |
|---|---|---|---|
| `query` | `str` | Oui | Question de l'utilisateur |
| `passages` | `list[str]` | Oui | Passages à reclasser |

**Retour** : `list[str]` — passages reordonnés du plus au moins pertinent ; retourne l'ordre d'entrée en cas d'échec de parsing

**Effets de bord** : appel réseau vers Gemini à `temperature=0.0`

#### `reset_llm() -> None` — réinitialisation du singleton

Remet `_llm_instance` à `None`. Principalement utile pour les tests.

---

## 3. APIs consommées

| Service | Endpoint | Type | Criticité | Timeout | Retry | Circuit breaker |
|---|---|---|---|---|---|---|
| **Gemini API (Google)** | `models/gemini-2.5-flash` (via llama-index) | Sync | Bloquante | Délégué à llama-index (à confirmer) | Aucun configuré | Aucun |
| **URL PDF source** | `GET {pdf_url}` (via `requests`) | Sync | Bloquante | 30 s | Aucun | Aucun |

**Modes dégradés** : Aucun mécanisme de fallback — toute indisponibilité de Gemini ou de l'URL PDF lève une exception non gérée.

---

## 4. Événements / messages

> Non applicable — le système ne produit ni ne consomme de topics ou queues de messages.

---

## 5. Commandes CLI

> Pas de commandes CLI formelles. Le point d'entrée `main.py` lance une boucle interactive non paramétrée.

| Commande | Rôle | Code source | Authentif | Effets de bord |
|---|---|---|---|---|
| `python main.py` | Lance le moteur RAG sur l'URL PDF codée en dur puis ouvre une boucle REPL | [main.py](../main.py) | `GOOGLE_API_KEY` dans `config.py` | Téléchargement PDF, création/lecture du cache d'index, appels Gemini |

---

## 6. Jobs / batchs

> Non applicable — le système ne comporte pas de jobs batch ou cron.

---

## 7. Mode d'appel inter-composants (interne)

Appel canonique pour interroger le système depuis un script Python :

```python
from rag_engine import RAGEngine

rag = RAGEngine(
    pdf_url="https://example.com/document.pdf",
    storage_dir="./storage"   # optionnel
)
answer: str = rag.query("Quelle est la conclusion principale ?")
```

**Points-clés** :

- Pas de propagation de contexte ni d'instrumentation — les seules traces sont des `print` sur stdout
- Gestion des erreurs : aucune — les exceptions remontent à l'appelant sans wrapping
- L'initialisation est coûteuse (téléchargement + embedding) — à faire une seule fois par processus

---

## 8. Matrice d'appels

> Légende : ✓ = inconditionnel · (c) = conditionnel · — = aucun appel

| Appelant ↓ / Appelé → | `RAGEngine` | `pdf_loader` | `gemini_client` | `text_processor` | Gemini API | URL PDF |
|---|---|---|---|---|---|---|
| **`main.py`** | ✓ | — | — | — | — | — |
| **`RAGEngine`** | — | ✓ (c) | ✓ | ✓ | ✓ | — |
| **`pdf_loader`** | — | — | — | — | — | ✓ |
| **`gemini_client`** | — | — | — | — | ✓ | — |

> `pdf_loader` n'est appelé par `RAGEngine` que si l'index n'est pas en cache `(c)`.

---

## 9. Formats des E/S physiques

### 9.1 Répertoire de cache d'index — sortie / archive

- **Chemin** : `{storage_dir}/index_{md5(pdf_url)}/` (ex. `./storage/index_a1b2c3d4.../`)
- **Type** : Répertoire llama-index (format JSON interne de `StorageContext`)
- **Encodage** : UTF-8
- **Taille typique** : (à confirmer)

| Fichier | Description |
|---|---|
| `docstore.json` | Documents bruts indexés |
| `index_store.json` | Métadonnées de l'index vectoriel |
| `vector_store.json` | Vecteurs d'embeddings |
| `graph_store.json` | (à confirmer) |

### 9.2 Fichier PDF temporaire — entrée

- **Chemin** : `{tempfile.gettempdir()}/temp_rag_document.pdf`
- **Type** : PDF binaire
- **Taille typique** : (à confirmer)

---

## 10. Transactions, commits, rollback

| Composant | Scope transactionnel | Commit | Rollback |
|---|---|---|---|
| `RAGEngine._save_index` | Écriture de répertoire (pas de transaction FS) | Après `persist()` complet | Aucun — répertoire partiellement écrit possible en cas d'interruption |

> Pas de base de données, pas de saga. Le seul état persisté est le cache d'index sur disque.

---

## 11. Contraintes non-fonctionnelles par interface

| Interface | Latence p95 cible | Disponibilité | Idempotence | Re-entrance |
|---|---|---|---|---|
| `RAGEngine.query()` | (à confirmer) | Dépend de Gemini API | Oui — même question → même réponse (déterministe avec `temperature=0.1`) | Non — partage de `query_engine` entre appels (à confirmer) |
| `load_pdf_from_url()` | ≤ 30 s (timeout HTTP) | Dépend de l'URL source | Non — écrase le fichier temp à chaque appel | Non |
| `RAGEngine.__init__()` | (à confirmer) | — | Oui si cache présent, coûteux sinon | Non (à confirmer) |

---

## 12. Tests de contrat

| Interface | Outil | Localisation | Côté |
|---|---|---|---|
| `initialize_gemini_llm()` | pytest | [tests/test_gemini_client.py](../tests/test_gemini_client.py) | Provider |

> Couverture partielle — `RAGEngine`, `load_pdf_from_url`, `load_documents_from_pdf` n'ont pas de tests de contrat identifiés (à confirmer).

---

## 13. Recommandations actives

1. **Ajouter une vérification du code HTTP dans `load_pdf_from_url`** — une URL en erreur 404/500 écrit silencieusement un fichier invalide qui fera échouer le `PDFReader` en aval.
2. **Paramétrer l'URL PDF dans `main.py`** — actuellement codée en dur (`https://arxiv.org/pdf/2005.11401.pdf`), ce qui rend le script non réutilisable sans modification du code.

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

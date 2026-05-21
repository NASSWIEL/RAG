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
| **Type d'interfaces** | Bibliothèque Python — appels natifs + CLI interactive |
| **Schémas authoritatifs** | Signatures de fonctions dans le code source (docstrings Google-style) |

> **Résumé en 1 phrase** : Le système expose une API Python (`RAGEngine`) pour indexer des PDF et répondre à des questions en langage naturel ; il consomme l'API Gemini de Google et le hub HuggingFace pour les embeddings.

---

## 0. Inventaire global

| Type d'interface | Nombre | Schéma authoritatif |
|---|---|---|
| **Endpoints HTTP exposés** | 0 | N/A — bibliothèque Python pure |
| **Endpoints HTTP consommés** | 2 | API Gemini (Google) + HuggingFace Hub (téléchargement modèle) |
| **Topics / queues produits** | 0 | N/A |
| **Topics / queues consommés** | 0 | N/A |
| **Commandes CLI** | 1 | [main.py](../main.py) — boucle interactive |
| **Jobs / batchs** | 0 | N/A |
| **Fichiers E/S (formats fixes)** | 1 | [§9](#9-formats-des-es-physiques) — PDF en entrée |

---

## 1. Conventions transverses des interfaces

### 1.1 Authentification et autorisation

| Type d'interface | Mécanisme | Token / claims | Refresh / rotation |
|---|---|---|---|
| **API publique exposée** | N/A — bibliothèque locale | — | — |
| **API Gemini (consommée)** | API key via variable d'environnement `GOOGLE_API_KEY` | Clé secrète chargée par `python-dotenv` | Manuelle (à confirmer) |
| **HuggingFace Hub (consommée)** | Téléchargement public (pas d'auth requise pour `BAAI/bge-small-en-v1.5`) | — | — |
| **CLI** | Aucune — processus local | — | — |

### 1.2 En-têtes / métadonnées standards

N/A — le système ne possède pas d'interface HTTP exposée. Les appels HTTP sortants (téléchargement PDF via `requests.get`) utilisent les en-têtes par défaut de la bibliothèque `requests` (à confirmer : pas de headers personnalisés détectés dans le code).

### 1.3 Codes d'erreur normalisés

N/A — pas d'interface HTTP exposée. Les erreurs sont propagées via des exceptions Python standard (ex. `KeyError` si `GOOGLE_API_KEY` absent, `requests.exceptions.RequestException` si téléchargement PDF échoue). Pas de codes d'erreur normalisés définis dans le code à ce stade (à confirmer).

### 1.4 Format de réponse d'erreur

N/A — bibliothèque Python. Les erreurs remontent par levée d'exception non gérée vers l'appelant.

### 1.5 Versioning

- **API Python** : pas de versioning explicite déclaré — le module est distribué directement depuis le dépôt (à confirmer).
- **Politique de breaking change** : (à confirmer)

---

## 2. API Python publique

> Le système est une bibliothèque Python — l'interface publique est constituée des méthodes de `RAGEngine` et des fonctions auxiliaires exportées par chaque module.

### 2.1 `RAGEngine` — orchestrateur principal

**Code source** : [rag_engine.py](../rag_engine.py)

#### `RAGEngine.__init__(pdf_url, storage_dir="./storage")` — initialisation du moteur RAG

| Paramètre | Type | Obligatoire | Description |
|---|---|---|---|
| `pdf_url` | `str` | Oui | URL publique du PDF à indexer |
| `storage_dir` | `str` | Non (défaut `"./storage"`) | Répertoire où persister l'index vectoriel |

**Comportement** :

1. Configure les embeddings HuggingFace (`BAAI/bge-small-en-v1.5`) via `setup_advanced_text_processing()`.
2. Initialise le LLM Gemini (`models/gemini-2.5-flash`) via `initialize_gemini_llm()`.
3. Crée un `SentenceSplitter` (chunk 512 / overlap 50) via `create_node_parser()`.
4. Si un index en cache existe (`./storage/index_<md5(pdf_url)>/`), le charge ; sinon, télécharge le PDF, le découpe, génère les embeddings et persiste l'index.

**Retour** : instance de `RAGEngine` prête à répondre à des questions.

**Pièges / particularités** :

- ⚠️ La clé d'environnement `GOOGLE_API_KEY` doit être présente (via `.env` ou variable système) avant l'instanciation — une `KeyError` sera levée sinon.
- ⚠️ Le cache est identifié par le MD5 de `pdf_url` ; changer l'URL force une ré-indexation complète même si le fichier PDF est identique.

---

#### `RAGEngine.query(question) -> str` — interrogation du moteur

| Paramètre | Type | Obligatoire | Description |
|---|---|---|---|
| `question` | `str` | Oui | Question en langage naturel |

**Retour** : `str` — réponse générée par Gemini à partir des passages les plus pertinents (top-3 par similarité cosinus).

**Effets de bord** : appel réseau vers l'API Gemini (`models/gemini-2.5-flash`).

---

### 2.2 `gemini_client` — client LLM

**Code source** : [gemini_client.py](../gemini_client.py)

#### `initialize_gemini_llm() -> Gemini`

Initialise une instance `Gemini` avec `model="models/gemini-2.5-flash"`, `temperature=0.1`, et l'enregistre dans `Settings.llm` de LlamaIndex.

**Pré-requis** : variable d'environnement `GOOGLE_API_KEY`.

---

### 2.3 `pdf_loader` — chargement des documents

**Code source** : [pdf_loader.py](../pdf_loader.py)

#### `load_pdf_from_url(url) -> str`

| Paramètre | Type | Description |
|---|---|---|
| `url` | `str` | URL HTTP(S) du PDF |

**Retour** : chemin local du fichier temporaire téléchargé (`str`). Timeout HTTP fixé à 30 secondes.

#### `load_documents_from_pdf(pdf_path) -> list`

| Paramètre | Type | Description |
|---|---|---|
| `pdf_path` | `str` | Chemin local du fichier PDF |

**Retour** : `list` de documents LlamaIndex extraits par `PDFReader`.

---

### 2.4 `text_processor` — traitement du texte

**Code source** : [text_processor.py](../text_processor.py)

#### `setup_advanced_text_processing() -> HuggingFaceEmbedding`

Configure `Settings.embed_model`, `Settings.chunk_size = 512`, `Settings.chunk_overlap = 50`. Retourne le modèle d'embedding `BAAI/bge-small-en-v1.5`.

#### `create_node_parser() -> SentenceSplitter`

Retourne un `SentenceSplitter(chunk_size=512, chunk_overlap=50)` configuré pour le découpage en nœuds.

---

## 3. APIs consommées

| Service | Endpoint | Type | Criticité | Timeout | Retry | Circuit breaker |
|---|---|---|---|---|---|---|
| Google Gemini | `models/gemini-2.5-flash` (SDK LlamaIndex) | Sync | Bloquante — sans LLM, pas de réponse | (à confirmer) | (à confirmer) | Aucun détecté |
| HuggingFace Hub | Téléchargement modèle `BAAI/bge-small-en-v1.5` | Sync (init uniquement) | Bloquante au premier démarrage ; mise en cache locale ensuite | (à confirmer) | (à confirmer) | Aucun détecté |
| HTTP (PDF source) | URL fournie par l'appelant | Sync (`requests.get`) | Bloquante au premier indexage ; l'index en cache supprime cette dépendance | 30 s | Aucun | Aucun |

**Modes dégradés** : aucun fallback implémenté — toute erreur des services tiers est propagée via exception non gérée (à confirmer).

---

## 4. Événements / messages

N/A — le système ne produit ni ne consomme aucun topic/queue. Architecture sans broker de messages.

---

## 5. Commandes CLI

| Commande | Rôle | Code source | Authentif | Effets de bord |
|---|---|---|---|---|
| `python main.py` | Boucle interactive de questions-réponses sur un PDF | [main.py](../main.py) | Variable d'env `GOOGLE_API_KEY` | Téléchargement PDF (réseau), génération d'embeddings, écriture du cache dans `./storage/`, appels API Gemini |

⚠️ L'URL du PDF est codée en dur dans `main.py` (`https://arxiv.org/pdf/2005.11401.pdf`) — la modifier nécessite une édition du fichier source.

---

## 6. Jobs / batchs

N/A — aucun job planifié ni batch défini dans le projet.

---

## 7. Mode d'appel inter-composants (interne)

`RAGEngine` est l'orchestrateur central. Tous les appels se font en Python dans le même processus.

```python
# Flux d'initialisation (RAGEngine.__init__)
embed_model = setup_advanced_text_processing()   # text_processor → Settings.embed_model
llm         = initialize_gemini_llm()            # gemini_client  → Settings.llm
parser      = create_node_parser()               # text_processor → Settings.node_parser

# Si pas de cache :
pdf_path    = load_pdf_from_url(pdf_url)         # pdf_loader     → HTTP GET + fichier tmp
documents   = load_documents_from_pdf(pdf_path)  # pdf_loader     → PDFReader
index       = VectorStoreIndex.from_documents(documents)  # LlamaIndex core

# Flux de requête (RAGEngine.query)
response    = self.query_engine.query(question)  # LlamaIndex → Gemini API
return str(response)
```

**Points-clés** :

- Pas de propagation de contexte inter-services (système single-process).
- Pas d'instrumentation (métriques, spans) à ce stade (à confirmer).
- Les erreurs remontent par exception non gérée jusqu'à l'appelant.

---

## 8. Matrice d'appels

> Légende : ✓ = inconditionnel · (c) = conditionnel · — = aucun appel

| Appelant ↓ / Appelé → | `RAGEngine` | `pdf_loader` | `text_processor` | `gemini_client` | Gemini API | HuggingFace |
|---|---|---|---|---|---|---|
| **`main.py`** | ✓ | — | — | — | — | — |
| **`RAGEngine`** | — | (c) init, si pas de cache | ✓ init | ✓ init | (c) via query_engine | — |
| **`text_processor`** | — | — | — | — | — | ✓ init (dl modèle) |
| **`gemini_client`** | — | — | — | — | ✓ init + chaque query | — |

> `pdf_loader` et `text_processor` sont des composants terminaux (pas d'appels sortants vers d'autres modules du projet).

---

## 9. Formats des E/S physiques

### 9.1 `*.pdf` — document source (entrée)

- **Type** : PDF binaire
- **Encodage** : binaire (lu par `PDFReader` de LlamaIndex)
- **Source** : URL HTTP(S) quelconque fournie lors de l'instanciation de `RAGEngine`
- **Taille typique** : (à confirmer)

| Champ | Description |
|---|---|
| Contenu textuel | Extrait page par page par `PDFReader`, converti en objets `Document` LlamaIndex |

### 9.2 Répertoire `./storage/index_<md5>/` — cache de l'index vectoriel (sortie/entrée)

- **Type** : artefacts LlamaIndex persistés via `StorageContext.persist()`
- **Localisation** : `storage_dir/index_<md5(pdf_url)>/` (défaut : `./storage/`)
- **Contenu** : fichiers internes LlamaIndex (JSON + index binaire) — format géré par la bibliothèque (à confirmer selon version LlamaIndex installée)

---

## 10. Transactions, commits, rollback

| Composant | Scope transactionnel | Commit | Rollback |
|---|---|---|---|
| `RAGEngine` (indexation) | Aucun — écriture fichiers sur disque | Implicite à la fin du `persist()` | Aucun — en cas d'erreur, le répertoire de cache peut rester partiel (à confirmer) |
| Appels Gemini API | Aucun — sans état | — | — |

> ⚠️ Si le processus est interrompu pendant `_save_index()`, le répertoire de cache peut être dans un état partiel — une ré-exécution devrait le détecter et reconstruire (à confirmer).

---

## 11. Contraintes non-fonctionnelles par interface

| Interface | Latence p95 cible | Disponibilité | Idempotence | Re-entrance |
|---|---|---|---|---|
| `RAGEngine.query()` | (à confirmer) | Dépend de l'API Gemini | Oui — lecture seule sur l'index | Non testée |
| `RAGEngine.__init__()` (avec cache) | (à confirmer) | 100% local | Oui | Non |
| `RAGEngine.__init__()` (sans cache) | (à confirmer) | Dépend réseau + Gemini + HuggingFace | Partiellement — cache peut être incomplet si interruption | Non |
| Téléchargement PDF (`load_pdf_from_url`) | Timeout 30 s | Dépend de la source | Oui (fichier tmp écrasé) | Oui |

---

## 12. Tests de contrat

Aucun test de contrat formel détecté dans le projet à ce stade (à confirmer — voir `tests/`).

---

## 13. Recommandations actives

1. **Externaliser l'URL du PDF hors du code source** — `main.py` contient une URL codée en dur ; la passer en argument CLI ou variable d'environnement améliorerait la réutilisabilité (à confirmer, pas de ticket).
2. **Ajouter une gestion d'erreur explicite sur `GOOGLE_API_KEY` absent** — actuellement `KeyError` brute levée par `os.environ["GOOGLE_API_KEY"]` (à confirmer, pas de ticket).
3. **Vérifier l'état partiel du cache en cas d'interruption** — `_save_index()` n'a pas de mécanisme atomique ; un cache corrompu bloquerait les runs suivants (à confirmer).

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

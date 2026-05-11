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

# Contrats d'interface — RAG (PDF Question-Answering)

| Champ | Valeur |
|---|---|
| **Dernière mise à jour** | 2026-05-11 |
| **Mise à jour par** | agent IA |
| **PR de référence** | fcd7a06 |
| **Type d'interfaces** | Appels natifs Python / CLI interactif |
| **Schémas authoritatifs** | Docstrings inline — [`rag_engine.py`](../rag_engine.py), [`pdf_loader.py`](../pdf_loader.py), [`text_processor.py`](../text_processor.py), [`gemini_client.py`](../gemini_client.py) |

> **Résumé en 1 phrase** : Le système expose une interface Python en appels directs (`RAGEngine`) et un CLI interactif (`main.py`) ; il consomme l'API Google Gemini (LLM) et des URLs de PDF publiques, et persiste les embeddings dans `./storage/`.

---

## 0. Inventaire global

| Type d'interface | Nombre | Schéma authoritatif |
|---|---|---|
| **Endpoints HTTP exposés** | 0 | — |
| **Endpoints HTTP consommés** | 1 (API Google Gemini) | [Google AI docs](https://ai.google.dev/api) |
| **Topics / queues produits** | 0 | — |
| **Topics / queues consommés** | 0 | — |
| **Commandes CLI** | 1 (`python main.py`) | [README.md](../README.md) |
| **Jobs / batchs** | 0 | — |
| **Fichiers E/S (formats fixes)** | 1 (PDF en entrée) | [§9](#9-formats-des-es-physiques) |
| **Appels natifs Python (publics)** | 4 | Docstrings — voir §2 |

---

## 1. Conventions transverses des interfaces

### 1.1 Authentification et autorisation

| Type d'interface | Mécanisme | Token / claims | Refresh / rotation |
|---|---|---|---|
| **API Google Gemini (consommée)** | API key (`GOOGLE_API_KEY` dans `config.py`) | — | Manuelle (à confirmer) |
| **CLI** | Aucun — exécution locale | — | — |
| **Appels natifs Python** | Aucun — pas d'exposition réseau | — | — |

### 1.2 En-têtes / métadonnées standards

> Ce système n'expose pas d'API HTTP. Pas d'en-têtes HTTP transverses à définir. Les appels sortants vers Google Gemini utilisent les en-têtes gérés automatiquement par le SDK `google-generativeai`.

### 1.3 Codes d'erreur normalisés

> Ce système n'expose pas d'API HTTP. Les erreurs sont levées sous forme d'exceptions Python non interceptées. Pas de codes HTTP applicatifs définis. Les erreurs d'API Gemini remontent via les exceptions du SDK `google-generativeai` (à confirmer).

### 1.4 Format de réponse d'erreur

> Non applicable — pas d'API HTTP exposée. Les erreurs sont propagées sous forme d'exceptions Python standard.

### 1.5 Versioning

> Pas de versioning formalisé — bibliothèque Python interne. Le modèle Gemini utilisé est `models/gemini-2.5-flash` (configurable dans [`gemini_client.py`](../gemini_client.py)).

---

## 2. Interfaces publiques Python (appels natifs)

> Ce système est une bibliothèque Python interne — pas d'API HTTP exposée. Les contrats publics sont les fonctions et méthodes ci-dessous.

### 2.1 `RAGEngine` — Moteur RAG principal

**Code source** : [`rag_engine.py`](../rag_engine.py)

#### `RAGEngine(pdf_url, storage_dir="./storage")` — Initialisation et indexation

| Paramètre | Type | Obligatoire | Description |
|---|---|---|---|
| `pdf_url` | `str` | Oui | URL publique du PDF à indexer |
| `storage_dir` | `str` | Non (défaut `"./storage"`) | Répertoire de persistance des embeddings |

**Comportement** : télécharge le PDF via `pdf_loader.load_pdf_from_url`, extrait les documents, construit un `VectorStoreIndex` (LlamaIndex) et le persiste sous `storage_dir/index_<md5(pdf_url)>/`. Si le répertoire existe déjà, charge l'index depuis le cache sans re-télécharger.

**Effets de bord** :
- Écriture disque dans `storage_dir/` (création du sous-dossier indexé).
- Appel réseau vers `pdf_url` (HTTP GET, timeout 30 s).
- Appel API Google Gemini pour initialisation du LLM.
- Chargement du modèle HuggingFace `BAAI/bge-small-en-v1.5` depuis le cache local ou HuggingFace Hub.

**Pièges / particularités** :
- ⚠️ Le cache est identifié par le hash MD5 de l'URL — un changement d'URL force un re-indexage, même si le PDF est identique.
- ⚠️ `Settings` LlamaIndex est muté globalement (`Settings.llm`, `Settings.embed_model`, `Settings.node_parser`, `Settings.chunk_size`, `Settings.chunk_overlap`) — effets de bord si plusieurs instances coexistent.

---

#### `RAGEngine.query(question)` — Interrogation du document indexé

| Paramètre | Type | Obligatoire | Description |
|---|---|---|---|
| `question` | `str` | Oui | Question en langage naturel posée sur le PDF indexé |

**Retour** : `str` — réponse générée par Gemini à partir des chunks les plus proches (top-k = 3, mode `compact`).

**Effets de bord** :
- Appel API Google Gemini (réseau, latence variable).

**Pièges / particularités** :
- ⚠️ La qualité de la réponse dépend de la pertinence des 3 chunks récupérés — les questions hors-scope du PDF produisent des réponses approximatives sans avertissement explicite.

---

### 2.2 `pdf_loader` — Chargement de PDF

**Code source** : [`pdf_loader.py`](../pdf_loader.py)

#### `load_pdf_from_url(url)` — Téléchargement d'un PDF depuis une URL

| Paramètre | Type | Obligatoire | Description |
|---|---|---|---|
| `url` | `str` | Oui | URL HTTP/HTTPS du fichier PDF |

**Retour** : `str` — chemin local du fichier temporaire (`tempfile.gettempdir()/temp_rag_document.pdf`).

**Effets de bord** : écriture dans le répertoire temporaire système. Le fichier est écrasé à chaque appel.

**Pièges / particularités** :
- ⚠️ Le nom de fichier temporaire est fixe (`temp_rag_document.pdf`) — appels concurrents se marcheraient dessus.

---

#### `load_documents_from_pdf(pdf_path)` — Extraction de documents depuis un PDF local

| Paramètre | Type | Obligatoire | Description |
|---|---|---|---|
| `pdf_path` | `str` | Oui | Chemin local vers un fichier PDF |

**Retour** : `list` — liste de documents LlamaIndex extraits page par page via `PDFReader`.

---

### 2.3 `text_processor` — Traitement du texte

**Code source** : [`text_processor.py`](../text_processor.py)

#### `setup_advanced_text_processing()` — Configuration des embeddings

**Paramètres** : aucun.

**Retour** : `HuggingFaceEmbedding` — instance configurée avec le modèle `BAAI/bge-small-en-v1.5`.

**Effets de bord** : mutation globale de `Settings.embed_model`, `Settings.chunk_size` (512), `Settings.chunk_overlap` (50).

---

#### `create_node_parser()` — Création du parser de nœuds

**Paramètres** : aucun.

**Retour** : `SentenceSplitter` configuré avec `chunk_size=512` et `chunk_overlap=50`.

---

### 2.4 `gemini_client` — Client LLM Gemini

**Code source** : [`gemini_client.py`](../gemini_client.py)

#### `initialize_gemini_llm()` — Initialisation du LLM Gemini

**Paramètres** : aucun (lit `GOOGLE_API_KEY` depuis `config.py`).

**Retour** : `Gemini` — instance LlamaIndex configurée avec `model="models/gemini-2.5-flash"` et `temperature=0.1`.

**Effets de bord** : mutation globale de `Settings.llm`. Nécessite `GOOGLE_API_KEY` valide dans `config.py`.

---

## 3. APIs consommées

| Service | Endpoint | Type | Criticité | Timeout | Retry | Circuit breaker |
|---|---|---|---|---|---|---|
| Google Gemini API | SDK `llama_index.llms.gemini` (API REST sous-jacente) | Sync | Bloquante (pas de fallback LLM) | Géré par le SDK (à confirmer) | Non configuré | Non configuré |
| HuggingFace Hub | Téléchargement modèle `BAAI/bge-small-en-v1.5` | Sync (initialisation uniquement) | Dégradable (cache local après 1er téléchargement) | Géré par `transformers` (à confirmer) | Non configuré | Non configuré |
| URL PDF fournie | `GET <pdf_url>` via `requests` | Sync | Bloquante (pas de fallback) | 30 s | Non configuré | Non configuré |

**Modes dégradés** : aucun mécanisme de dégradation explicite — si Google Gemini est KO ou si le PDF est inaccessible, le système lève une exception non interceptée.

---

## 4. Événements / messages

> Ce système ne produit ni ne consomme de topics/queues. Section non applicable.

---

## 5. Commandes CLI

| Commande | Rôle | Code source | Authentif | Effets de bord |
|---|---|---|---|---|
| `python main.py` | Lance la boucle interactive de questions-réponses sur le PDF configuré | [`main.py`](../main.py) | Aucune (locale) | Téléchargement PDF, écriture `./storage/`, appels API Gemini |

---

## 6. Jobs / batchs

> Ce système ne comporte pas de jobs ou batchs planifiés. Section non applicable.

---

## 7. Mode d'appel inter-composants (interne)

Les composants communiquent par appels de fonctions Python directs dans le même processus. Le flux d'orchestration standard est :

```python
# main.py → RAGEngine (orchestrateur)
engine = RAGEngine(pdf_url="https://...")

# RAGEngine.__init__ appelle en séquence :
#   text_processor.setup_advanced_text_processing()  → Settings.embed_model
#   gemini_client.initialize_gemini_llm()            → Settings.llm
#   text_processor.create_node_parser()              → Settings.node_parser
#   pdf_loader.load_pdf_from_url(pdf_url)            → chemin local
#   pdf_loader.load_documents_from_pdf(pdf_path)     → list[Document]

answer = engine.query("Ma question")
```

**Points-clés** :
- Pas de propagation de contexte de corrélation entre composants.
- Pas d'instrumentation (métriques, spans) — les logs sont de simples `print()`.
- Gestion des erreurs : exceptions Python non interceptées remontent jusqu'au point d'entrée.

---

## 8. Matrice d'appels

> Légende : ✓ = inconditionnel · (c) = conditionnel · — = aucun appel

| Appelant ↓ / Appelé → | `main.py` | `RAGEngine` | `pdf_loader` | `text_processor` | `gemini_client` | Gemini API | HuggingFace Hub | URL PDF |
|---|---|---|---|---|---|---|---|---|
| **`main.py`** | — | ✓ | — | — | — | — | — | — |
| **`RAGEngine`** | — | — | ✓ | ✓ | ✓ | — | — | — |
| **`pdf_loader`** | — | — | — | — | — | — | — | ✓ |
| **`gemini_client`** | — | — | — | — | — | ✓ | — | — |
| **`text_processor`** | — | — | — | — | — | — | ✓ (c) | — |

---

## 9. Formats des E/S physiques

### 9.1 PDF — entrée document source

- **Type** : PDF binaire
- **Accès** : URL HTTP/HTTPS fournie par l'appelant
- **Stockage temporaire** : `<tempdir>/temp_rag_document.pdf` (écrasé à chaque appel)
- **Parser** : `llama_index.readers.file.PDFReader` (via `pypdf`)
- **Taille typique** : (à confirmer)

### 9.2 Index vectoriel — sortie/cache embarqué

- **Type** : Répertoire LlamaIndex (`docstore.json`, `index_store.json`, `vector_store.json`, etc.)
- **Localisation** : `<storage_dir>/index_<md5(pdf_url)>/`
- **Défaut** : `./storage/index_<hash>/`
- **Encodage** : JSON (UTF-8)

---

## 10. Transactions, commits, rollback

| Composant | Scope transactionnel | Commit | Rollback |
|---|---|---|---|
| `RAGEngine.__init__` | Aucun (pas de BDD) | Écriture disque `_save_index()` après indexation complète | Aucun — index partiel possible si crash pendant l'écriture |
| `pdf_loader` | Aucun | Écriture fichier temporaire | Aucun |

> Pas de transactions ACID. L'écriture de l'index est un `persist()` LlamaIndex — en cas d'interruption, le répertoire peut être laissé dans un état incomplet et sera rechargé comme valide au prochain démarrage (à confirmer).

---

## 11. Contraintes non-fonctionnelles par interface

| Interface | Latence p95 cible | Disponibilité | Idempotence | Re-entrance |
|---|---|---|---|---|
| `RAGEngine.__init__` (1er appel — sans cache) | (à confirmer) | Dépend de l'accès réseau et HuggingFace Hub | Oui — si le cache existe, ré-exécution sans effet | Non (mutation `Settings` global) |
| `RAGEngine.__init__` (appels suivants — avec cache) | (à confirmer) | Locale (cache disque) | Oui | Non (mutation `Settings` global) |
| `RAGEngine.query` | (à confirmer) | Dépend de Google Gemini API | Non formalisé | Oui (lecture seule) |
| `python main.py` (CLI) | — | Locale | Oui | Non |

---

## 12. Tests de contrat

> Aucun test de contrat formalisé identifié dans le dépôt. (à confirmer)

---

## 13. Recommandations actives

1. **Isoler la mutation de `Settings` LlamaIndex** — Les fonctions `setup_advanced_text_processing()`, `initialize_gemini_llm()` et `create_node_parser()` mutent un état global (`Settings`). En cas d'usage multi-thread ou multi-instance, ce couplage implicite peut causer des comportements inattendus. Envisager de passer les paramètres explicitement plutôt que via `Settings`.
2. **Ajouter une gestion d'erreur réseau** — `load_pdf_from_url` et l'appel Gemini ne disposent d'aucun retry ni fallback. Un timeout réseau ou une indisponibilité de l'API entraîne un crash non géré.
3. **Sécuriser `GOOGLE_API_KEY`** — La clé est lue depuis `config.py` en clair ; préférer une variable d'environnement ou un gestionnaire de secrets.

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

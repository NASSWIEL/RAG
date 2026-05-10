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
| **Dernière mise à jour** | 2026-05-10 |
| **Mise à jour par** | agent IA |
| **PR de référence** | (à confirmer) |
| **Type d'interfaces** | CLI · appels natifs Python (aucun endpoint HTTP exposé) |
| **Schémas authoritatifs** | Signatures dans [`rag_engine.py`](../rag_engine.py), [`pdf_loader.py`](../pdf_loader.py), [`gemini_client.py`](../gemini_client.py), [`text_processor.py`](../text_processor.py) |

> **Résumé en 1 phrase** : Le système expose une API Python interne (`RAGEngine`) et des fonctions utilitaires, consomme l'API Google Gemini et des PDFs distants via HTTP, et persiste les index vectoriels sur le disque local (`./storage/`).

---

## 0. Inventaire global

| Type d'interface | Nombre | Schéma authoritatif |
|---|---|---|
| **Endpoints HTTP exposés** | 0 | — |
| **Endpoints HTTP consommés** | 2 | Google Gemini API · URL PDF arbitraire (voir §3) |
| **Topics / queues produits** | 0 | — |
| **Topics / queues consommés** | 0 | — |
| **Commandes CLI** | 1 | [`main.py`](../main.py) (voir §5) |
| **Jobs / batchs** | 0 | — |
| **API Python interne (publique)** | 4 | [`rag_engine.py`](../rag_engine.py), [`pdf_loader.py`](../pdf_loader.py), [`gemini_client.py`](../gemini_client.py), [`text_processor.py`](../text_processor.py) (voir §7) |
| **Fichiers E/S (formats fixes)** | 1 | Index vectoriel persisté dans `./storage/` (voir §9) |

---

## 1. Conventions transverses des interfaces

### 1.1 Authentification et autorisation

| Type d'interface | Mécanisme | Token / claims | Refresh / rotation |
|---|---|---|---|
| **Google Gemini (consommé)** | API Key (`GOOGLE_API_KEY` dans [`config.py`](../config.py)) | Clé transmise dans chaque requête SDK | Manuelle — aucune rotation automatique |
| **Téléchargement PDF (consommé)** | Aucun — requête HTTP GET non authentifiée | — | — |
| **API Python interne** | Aucune — appels en-process | — | — |
| **CLI** | Aucune — exécuté en local | — | — |

### 1.2 En-têtes / métadonnées standards

> Le système n'expose aucun endpoint HTTP. Aucun en-tête standard n'est défini côté serveur. Les en-têtes des requêtes sortantes (Gemini SDK, `requests.get` vers l'URL PDF) sont gérés par les bibliothèques tierces.

### 1.3 Codes d'erreur normalisés

> Le système n'expose aucun endpoint HTTP. Les erreurs remontent sous forme d'exceptions Python non typées (à confirmer). Aucun code d'erreur applicatif normalisé n'est défini à ce stade.

### 1.4 Format de réponse d'erreur

> Non applicable — aucun endpoint HTTP exposé. Les erreurs se propagent via les mécanismes d'exception Python standard.

### 1.5 Versioning

- **API Python interne** : aucun versioning explicite — la compatibilité est assurée au niveau du dépôt (à confirmer).
- **Politique de breaking change** : (à confirmer)

---

## 2. APIs exposées (HTTP / gRPC / GraphQL)

> Non applicable — le système n'expose aucun endpoint HTTP, gRPC ou GraphQL. Voir §7 pour les contrats de l'API Python interne.

---

## 3. APIs consommées

| Service | Endpoint / mode | Type | Criticité | Timeout | Retry | Circuit breaker |
|---|---|---|---|---|---|---|
| **Google Gemini** (`models/gemini-2.5-flash`) | SDK Python `llama-index-llms-gemini` | Sync | Bloquante — toute requête nécessite le LLM | (à confirmer — géré par le SDK) | (à confirmer) | Aucun |
| **URL PDF distante** | `GET <pdf_url>` via `requests.get` | Sync | Bloquante à l'initialisation | 30 s (défini dans [`pdf_loader.py`](../pdf_loader.py)) | Aucun | Aucun |

**Modes dégradés** :
- **Gemini KO** : l'appel `query()` lève une exception non interceptée — aucun fallback défini (à confirmer).
- **URL PDF inaccessible** : `requests.get` lève une exception après 30 s de timeout — aucun retry ni fallback défini.

---

## 4. Événements / messages

> Non applicable — le système ne produit ni ne consomme de topics/queues. Toute la communication est synchrone (appels en-process ou HTTP vers services tiers).

---

## 5. Commandes CLI

| Commande | Rôle | Code source | Authentif | Effets de bord |
|---|---|---|---|---|
| `python main.py` | Lance la boucle interactive de questions-réponses RAG | [`main.py`](../main.py) | `GOOGLE_API_KEY` dans [`config.py`](../config.py) | Téléchargement du PDF distant, écriture du cache d'index dans `./storage/`, appels réseau vers l'API Gemini |

**Options / variables** :

- `pdf_url` dans `main.py` (ligne 8) — URL du PDF à indexer. Valeur par défaut : `https://arxiv.org/pdf/2005.11401.pdf`.
- `storage_dir` dans `RAGEngine.__init__` — répertoire de cache des embeddings. Valeur par défaut : `./storage`.
- Pour quitter la boucle interactive : saisir `q`.

---

## 6. Jobs / batchs

> Non applicable — le système ne comporte aucun job batch ni tâche planifiée.

---

## 7. Mode d'appel inter-composants (interne) — API Python publique

> Le système est une bibliothèque Python. Toutes les interfaces publiques sont des fonctions et classes Python appelées en-process.

### 7.1 `RAGEngine` — [`rag_engine.py`](../rag_engine.py)

Orchestrateur principal du pipeline RAG.

#### `RAGEngine.__init__(pdf_url, storage_dir="./storage")`

| Paramètre | Type | Obligatoire | Description |
|---|---|---|---|
| `pdf_url` | `str` | Oui | URL publique du PDF à indexer |
| `storage_dir` | `str` | Non | Répertoire de persistance des embeddings. Défaut : `./storage` |

**Effets de bord** :
- Appelle `setup_advanced_text_processing()`, `initialize_gemini_llm()`, `create_node_parser()`.
- Si un cache existe pour `pdf_url` (identifié par `md5(pdf_url)`), charge l'index depuis le disque.
- Sinon : télécharge le PDF via `load_pdf_from_url()`, extrait les documents via `load_documents_from_pdf()`, construit un `VectorStoreIndex`, et persiste l'index dans `storage_dir/index_<md5hash>/`.

**Pièges** :
- ⚠️ L'invalidation du cache est basée sur le hash MD5 de l'URL uniquement — un changement de contenu du PDF à la même URL ne déclenche pas un re-indexage.
- ⚠️ `storage_dir` doit être accessible en écriture au premier lancement.

#### `RAGEngine.query(question) -> str`

| Paramètre | Type | Obligatoire | Description |
|---|---|---|---|
| `question` | `str` | Oui | Question en langage naturel à soumettre au moteur RAG |

**Retour** : `str` — réponse générée par Gemini à partir des chunks les plus proches sémantiquement (`similarity_top_k=3`, `response_mode="compact"`).

**Effets de bord** : appel réseau sortant vers l'API Google Gemini.

---

### 7.2 `pdf_loader` — [`pdf_loader.py`](../pdf_loader.py)

#### `load_pdf_from_url(url) -> str`

| Paramètre | Type | Obligatoire | Description |
|---|---|---|---|
| `url` | `str` | Oui | URL publique du PDF |

**Retour** : `str` — chemin local du fichier temporaire téléchargé (`<tempdir>/temp_rag_document.pdf`).

**Effets de bord** : requête HTTP GET avec `timeout=30 s`, écriture d'un fichier temporaire.

**Pièges** :
- ⚠️ Le fichier temporaire est toujours écrit au même emplacement (`temp_rag_document.pdf`) — les appels concurrents écraseraient le fichier mutuellement.

#### `load_documents_from_pdf(pdf_path) -> list`

| Paramètre | Type | Obligatoire | Description |
|---|---|---|---|
| `pdf_path` | `str` | Oui | Chemin local vers le fichier PDF |

**Retour** : `list` de documents LlamaIndex extraits page par page via `llama_index.readers.file.PDFReader`.

---

### 7.3 `gemini_client` — [`gemini_client.py`](../gemini_client.py)

#### `initialize_gemini_llm() -> Gemini`

Aucun paramètre. Lit `GOOGLE_API_KEY` depuis [`config.py`](../config.py).

**Retour** : instance `llama_index.llms.gemini.Gemini` configurée avec le modèle `models/gemini-2.5-flash`, `temperature=0.1`.

**Effets de bord** : enregistre le LLM dans `llama_index.core.Settings.llm` (singleton global).

**Pièges** :
- ⚠️ Mutation du singleton global `Settings.llm` — incompatible avec une utilisation multi-moteur dans le même process.

---

### 7.4 `text_processor` — [`text_processor.py`](../text_processor.py)

#### `setup_advanced_text_processing() -> HuggingFaceEmbedding`

Aucun paramètre.

**Retour** : instance `HuggingFaceEmbedding` utilisant le modèle `BAAI/bge-small-en-v1.5`.

**Effets de bord** : enregistre le modèle dans `Settings.embed_model`, définit `Settings.chunk_size=512` et `Settings.chunk_overlap=50` (singleton global).

#### `create_node_parser() -> SentenceSplitter`

Aucun paramètre.

**Retour** : instance `SentenceSplitter` avec `chunk_size=512`, `chunk_overlap=50`.

**Effets de bord** : aucun (l'enregistrement dans `Settings.node_parser` est fait dans `RAGEngine.__init__`).

**Points-clés** :
- Aucune propagation de contexte de corrélation entre les composants.
- Aucune instrumentation (métriques, spans) définie à ce stade (à confirmer).
- Les erreurs se propagent via les exceptions Python standard — aucune exception typée définie.

---

## 8. Matrice d'appels

> Légende : ✓ = inconditionnel · (c) = conditionnel · — = aucun appel

| Appelant ↓ / Appelé → | `RAGEngine` | `pdf_loader` | `gemini_client` | `text_processor` | Gemini API | URL PDF |
|---|---|---|---|---|---|---|
| **`main.py`** | ✓ | — | — | — | — | — |
| **`RAGEngine.__init__`** | — | ✓ | ✓ | ✓ | — | — |
| **`RAGEngine.query`** | — | — | — | — | ✓ (via SDK) | — |
| **`pdf_loader`** | — | — | — | — | — | ✓ |
| **`gemini_client`** | — | — | — | — | — | — |
| **`text_processor`** | — | — | — | — | — | — |

---

## 9. Formats des E/S physiques

### 9.1 Index vectoriel persisté — sortie/cache

- **Rôle** : cache des embeddings et de l'index LlamaIndex pour un PDF donné.
- **Localisation** : `<storage_dir>/index_<md5(pdf_url)>/` (ex. `./storage/index_a1b2c3d4.../`).
- **Type** : répertoire contenant les fichiers de persistance LlamaIndex (`docstore.json`, `index_store.json`, `vector_store.json`, `graph_store.json`) (à confirmer — format géré par `llama_index.core.StorageContext`).
- **Encodage** : JSON UTF-8 (à confirmer).
- **Taille typique** : dépend du PDF source (à confirmer).
- **Rejouable** : oui — supprimer le répertoire force un re-indexage complet.

### 9.2 Fichier PDF temporaire — entrée

- **Rôle** : copie locale temporaire du PDF téléchargé.
- **Localisation** : `<tempdir>/temp_rag_document.pdf` (chemin système via `tempfile.gettempdir()`).
- **Type** : fichier PDF binaire.
- **Cycle de vie** : non supprimé automatiquement — réutilisé à chaque run (à confirmer).

---

## 10. Transactions, commits, rollback

| Composant | Scope transactionnel | Commit | Rollback |
|---|---|---|---|
| `RAGEngine.__init__` (persistance index) | Par initialisation — écriture atomique du répertoire de cache | `StorageContext.persist()` après indexage complet | Aucun — un index partiellement écrit peut corrompre le cache (à confirmer) |
| `load_pdf_from_url` | Aucun — écriture fichier temporaire | Implicite à la fermeture du fichier | Aucun |

> Aucune base de données relationnelle n'est impliquée. Aucune saga ni transaction distribuée.

---

## 11. Contraintes non-fonctionnelles par interface

| Interface | Latence p95 cible | Disponibilité | Idempotence | Re-entrance |
|---|---|---|---|---|
| `RAGEngine.__init__` (cache chaud) | (à confirmer) | Dépend du disque local | Oui — relecture du cache existant | Non (mutation de `Settings` global) |
| `RAGEngine.__init__` (cache froid) | Dépend taille PDF + réseau | Dépend de l'URL PDF et de l'API Gemini | Oui si le cache est proprement écrit | Non |
| `RAGEngine.query` | Dépend de la latence Gemini API | Dépend de l'API Gemini | Non garanti (LLM non déterministe) | (à confirmer) |
| `load_pdf_from_url` | ≤ 30 s (timeout fixé) | Dépend de l'URL distante | Oui (écrase le même fichier temp) | Non (écriture fichier partagé) |

---

## 12. Tests de contrat

> Aucun test de contrat formalisé n'est défini à ce stade (à confirmer). Les tests unitaires pytest sont configurés dans `tests/` (voir `pyproject.toml`).

---

## 13. Recommandations actives

1. **Gérer les exceptions de manière typée** — `RAGEngine.query` et `load_pdf_from_url` propagent des exceptions non typées ; définir des exceptions métier améliorerait la robustesse des appelants.
2. **Éviter les mutations du singleton `Settings`** — `initialize_gemini_llm` et `setup_advanced_text_processing` mutent `llama_index.core.Settings` globalement, rendant l'utilisation multi-instance impossible dans le même process.
3. **Ajouter un retry sur `load_pdf_from_url`** — aucun retry n'est défini pour les échecs réseau transitoires.
4. **Nettoyage du fichier temporaire** — `temp_rag_document.pdf` n'est pas supprimé après indexage ; risque de fuite disque ou de race condition en usage concurrent.

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

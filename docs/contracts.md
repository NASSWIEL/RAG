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
| **Dernière mise à jour** | 2026-05-11 |
| **Mise à jour par** | agent IA (doc-sync) |
| **PR de référence** | fcd7a06 |
| **Type d'interfaces** | Appels Python natifs (lib / CLI interactif) |
| **Schémas authoritatifs** | Docstrings inline — voir [rag_engine.py](../rag_engine.py), [gemini_client.py](../gemini_client.py), [pdf_loader.py](../pdf_loader.py), [text_processor.py](../text_processor.py) |

> **Résumé en 1 phrase** : Le système expose une API Python interne (`RAGEngine`) qui indexe un PDF distant et répond à des questions en langage naturel via Gemini 2.5 Flash ; il ne publie aucun endpoint HTTP ni événement.

---

## 0. Inventaire global

| Type d'interface | Nombre | Schéma authoritatif |
|---|---|---|
| **Endpoints HTTP exposés** | 0 | — |
| **Endpoints HTTP consommés** | 1 (Google Gemini API) | [§3](#3-apis-consommées) |
| **Topics / queues produits** | 0 | — |
| **Topics / queues consommés** | 0 | — |
| **Commandes CLI** | 0 (point d'entrée interactif `main.py`, pas de CLI structurée) | — |
| **Jobs / batchs** | 0 | — |
| **Fichiers E/S (formats fixes)** | 1 (PDF en entrée, index persisté sur disque) | [§9](#9-formats-des-es-physiques) |
| **API Python interne** | 1 classe publique (`RAGEngine`) | [§7](#7-mode-dappel-inter-composants-interne) |

---

## 1. Conventions transverses des interfaces

### 1.1 Authentification et autorisation

| Type d'interface | Mécanisme | Token / claims | Refresh / rotation |
|---|---|---|---|
| **API publique** | Aucune — pas d'endpoint exposé | — | — |
| **API interne (Python)** | Aucune — appel in-process | — | — |
| **Appel Gemini (sortant)** | API key (`GOOGLE_API_KEY`) via variable d'environnement | Clé simple | Rotation manuelle |
| **Événements** | Aucun | — | — |

### 1.2 En-têtes / métadonnées standards

> Le système n'expose pas d'API HTTP ; cette section ne concerne que les appels sortants vers Gemini.

| En-tête / champ | Rôle | Direction | Obligatoire ? |
|---|---|---|---|
| `x-goog-api-key` | Authentification Gemini (injecté par le SDK) | Sortant | Oui |

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

> Pas d'API HTTP exposée. Les erreurs sont propagées sous forme d'exceptions Python standard (ex. `requests.exceptions.RequestException` pour le téléchargement du PDF, exceptions LlamaIndex pour l'indexation).

### 1.5 Versioning

- **API Python** : pas de versioning formel ; la signature de `RAGEngine.__init__` et `RAGEngine.query` constitue le contrat public (à confirmer).
- **Modèle LLM** : `models/gemini-2.5-flash` — hardcodé dans [gemini_client.py](../gemini_client.py).
- **Politique de breaking change** : (à confirmer)

---

## 2. APIs exposées (HTTP / gRPC / GraphQL)

> Ce système n'expose aucun endpoint HTTP, gRPC ou GraphQL. Voir §7 pour l'API Python interne.

---

## 3. APIs consommées

> Une ligne par dépendance.

| Service | Endpoint | Type | Criticité | Timeout | Retry | Circuit breaker |
|---|---|---|---|---|---|---|
| Google Gemini API | `POST https://generativelanguage.googleapis.com/...` (via SDK `llama_index.llms.gemini`) | Sync | Bloquante (génération de réponse) | 30 s (à confirmer) | Non configuré | Non configuré |
| HTTP (téléchargement PDF) | URL arbitraire fournie à `RAGEngine.__init__` | Sync | Bloquante (première indexation) | 30 s ([pdf_loader.py](../pdf_loader.py)#L21) | Non configuré | Non configuré |

**Modes dégradés** : Aucune logique de fallback implémentée — toute erreur réseau ou Gemini lève une exception non interceptée et termine le processus.

---

## 4. Événements / messages

> Ce système ne produit ni ne consomme de topics ou queues. Section non applicable.

---

## 5. Commandes CLI

> Pas de CLI structurée. Le point d'entrée interactif est `python main.py` — il lance une boucle REPL qui appelle `RAGEngine.query()` en boucle jusqu'à la saisie de `q`.

| Commande | Rôle | Code source | Authentif | Effets de bord |
|---|---|---|---|---|
| `python main.py` | Lance le REPL RAG interactif avec le PDF arxiv 2005.11401 | [main.py](../main.py) | `GOOGLE_API_KEY` requis en env | Téléchargement PDF, création de `./storage/index_<md5>/` au premier lancement |

---

## 6. Jobs / batchs

> Aucun job ou batch planifié. Section non applicable.

---

## 7. Mode d'appel inter-composants (interne)

> Le système est monolithique in-process. L'API publique est la classe `RAGEngine`. Les autres modules (`gemini_client`, `pdf_loader`, `text_processor`) sont des utilitaires internes appelés uniquement par `RAGEngine.__init__`.

### 7.1 `RAGEngine` — API Python publique

| Méta | Valeur |
|---|---|
| **Code source** | [rag_engine.py](../rag_engine.py) |
| **Auth requise** | `GOOGLE_API_KEY` en variable d'environnement (via `config.py`) |

#### `RAGEngine(pdf_url, storage_dir="./storage")` — constructeur

| Paramètre | Type | Obligatoire | Description |
|---|---|---|---|
| `pdf_url` | `str` | Oui | URL du PDF à indexer |
| `storage_dir` | `str` | Non (défaut `"./storage"`) | Répertoire de persistance de l'index |

**Effets de bord** : télécharge le PDF si absent du cache, crée `./storage/index_<md5(pdf_url)>/`, initialise LlamaIndex Settings (modèle d'embedding `BAAI/bge-small-en-v1.5`, chunk 512, overlap 50), initialise Gemini LLM (`models/gemini-2.5-flash`, temperature 0.1).

**Pièges / particularités**

- ⚠️ Le cache d'index est identifié par un hash MD5 de `pdf_url` — changer l'URL (même contenu) invalide le cache.
- ⚠️ `Settings` LlamaIndex est global (singleton) — instancier plusieurs `RAGEngine` en parallèle peut produire des effets de bord inattendus.

#### `RAGEngine.query(question: str) -> str` — interrogation

| Paramètre | Type | Obligatoire | Description |
|---|---|---|---|
| `question` | `str` | Oui | Question en langage naturel |

**Retour** : `str` — réponse générée par Gemini à partir des 3 chunks les plus similaires (`similarity_top_k=3`, `response_mode="compact"`).

### 7.2 Fonctions utilitaires internes

| Fonction | Module | Rôle |
|---|---|---|
| `initialize_gemini_llm() -> Gemini` | [gemini_client.py](../gemini_client.py) | Crée et enregistre le LLM Gemini dans Settings |
| `setup_advanced_text_processing() -> HuggingFaceEmbedding` | [text_processor.py](../text_processor.py) | Configure l'embedding `BAAI/bge-small-en-v1.5` + chunk settings |
| `create_node_parser() -> SentenceSplitter` | [text_processor.py](../text_processor.py) | Crée un `SentenceSplitter(chunk_size=512, chunk_overlap=50)` |
| `load_pdf_from_url(url: str) -> str` | [pdf_loader.py](../pdf_loader.py) | Télécharge le PDF (timeout 30 s) vers un fichier temp ; retourne le chemin |
| `load_documents_from_pdf(pdf_path: str) -> list` | [pdf_loader.py](../pdf_loader.py) | Parse le PDF via `PDFReader` ; retourne une liste de documents LlamaIndex |

**Points-clés** :

- Pas de propagation de contexte de corrélation entre les modules.
- Pas d'instrumentation (métriques, spans) — uniquement des `print()` de progression.
- Gestion des erreurs : aucune — toute exception se propage jusqu'au `main()` et termine le processus.

---

## 8. Matrice d'appels

> Légende : ✓ = inconditionnel · (c) = conditionnel · — = aucun appel.

| Appelant ↓ / Appelé → | `RAGEngine` | `gemini_client` | `pdf_loader` | `text_processor` | Gemini API | HTTP (PDF) |
|---|---|---|---|---|---|---|
| **`main.py`** | ✓ | — | — | — | — | — |
| **`RAGEngine`** | — | ✓ | (c) init seulement si pas de cache | ✓ | — | — |
| **`gemini_client`** | — | — | — | — | ✓ | — |
| **`pdf_loader`** | — | — | — | — | — | ✓ |

---

## 9. Formats des E/S physiques

### 9.1 PDF d'entrée — source de connaissance

- **Type** : PDF binaire téléchargé via HTTP
- **Chemin temporaire** : `<tempdir>/temp_rag_document.pdf` (écrasé à chaque lancement sans cache)
- **Taille typique** : (à confirmer)
- **Encodage** : géré par `PDFReader` (LlamaIndex)

### 9.2 Index vectoriel persisté — cache disque

- **Chemin** : `./storage/index_<md5(pdf_url)>/` (JSON LlamaIndex)
- **Type** : répertoire JSON (format interne LlamaIndex `StorageContext`)
- **Encodage** : UTF-8 / JSON
- **Création** : automatique au premier lancement ; rechargé aux lancements suivants

| Fichier dans le répertoire | Description |
|---|---|
| `docstore.json` | Documents LlamaIndex sérialisés |
| `index_store.json` | Métadonnées de l'index |
| `vector_store.json` | Vecteurs d'embedding |
| `graph_store.json` | Graphe de relations (vide si non utilisé) |

---

## 10. Transactions, commits, rollback

| Composant | Scope transactionnel | Commit | Rollback |
|---|---|---|---|
| `RAGEngine._save_index` | Écriture disque de l'index (non atomique) | Après `VectorStoreIndex.from_documents` | Aucun — répertoire partiel possible si crash pendant la sauvegarde |

> ⚠️ Si le processus est interrompu pendant `_save_index`, le répertoire `./storage/index_<md5>/` peut être incomplet et provoquer une erreur au chargement suivant. Supprimer manuellement le répertoire corrompu pour forcer une réindexation.

---

## 11. Contraintes non-fonctionnelles par interface

| Interface | Latence p95 cible | Disponibilité | Idempotence | Re-entrance |
|---|---|---|---|---|
| `RAGEngine.query()` | (à confirmer) | Dépend de la disponibilité Gemini API | Oui — même question produit une réponse équivalente | Non (Settings LlamaIndex global non thread-safe) |
| Téléchargement PDF | (à confirmer) | Dépend de la source externe | Oui — même URL produit le même fichier | Non |
| Sauvegarde index disque | (à confirmer) | Dépend du système de fichiers | Oui si index complet | Non |

---

## 12. Tests de contrat

| Interface | Outil | Localisation | Côté |
|---|---|---|---|
| `RAGEngine` (Gemini mock) | pytest | [tests/test_gemini_client.py](../tests/test_gemini_client.py) | Consumer |

> ⚠️ Les tests couvrent uniquement `gemini_client`. Il n'existe pas de tests de contrat pour `RAGEngine.query()`, `pdf_loader`, ni `text_processor` — dette à adresser.

---

## 13. Recommandations actives

1. **Ajouter une gestion d'erreur dans `RAGEngine`** — toute exception réseau ou Gemini termine actuellement le processus sans message clair. Implémenter un try/except avec messages explicites et stratégie de retry configurable.
2. **Rendre `Settings` LlamaIndex non-global** — la dépendance au singleton rend `RAGEngine` non thread-safe et les tests difficiles à isoler. Passer les settings en paramètre ou via un contexte local.
3. **Ajouter des tests de contrat pour `RAGEngine.query()`** — seul `gemini_client` est testé ; les chemins d'indexation et de requête ne sont pas couverts.
4. **Documenter `config.py`** — ce module est importé par `gemini_client` mais n'est pas listé dans les fichiers du projet ; confirmer sa présence et son contenu (à confirmer).

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

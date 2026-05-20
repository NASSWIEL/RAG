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
| **Dernière mise à jour** | 2026-05-20 |
| **Mise à jour par** | agent IA |
| **PR de référence** | (à confirmer) |
| **Type d'interfaces** | CLI / appels natifs Python (aucune interface réseau exposée) |
| **Schémas authoritatifs** | N/A — le code Python est la source de vérité (pas d'OpenAPI/proto) |

> **Résumé en 1 phrase** : Application Python CLI qui télécharge un PDF depuis une URL, l'indexe avec des embeddings BAAI/bge-small-en-v1.5, et répond à des questions en langage naturel via Gemini 2.5 Flash (pipeline RAG LlamaIndex).

---

## 0. Inventaire global

| Type d'interface | Nombre | Schéma authoritatif |
|---|---|---|
| **Endpoints HTTP exposés** | 0 | N/A |
| **Endpoints HTTP consommés** | 2 | Voir §3 |
| **Topics / queues produits** | 0 | N/A |
| **Topics / queues consommés** | 0 | N/A |
| **Commandes CLI** | 1 | [§5](#5-commandes-cli) |
| **Jobs / batchs** | 0 | N/A |
| **Fichiers E/S (formats fixes)** | 2 | [§9](#9-formats-des-es-physiques) |

---

## 1. Conventions transverses des interfaces

### 1.1 Authentification et autorisation

| Type d'interface | Mécanisme | Token / claims | Refresh / rotation |
|---|---|---|---|
| **API publique** | N/A — aucune interface publique exposée | — | — |
| **API interne** | N/A | — | — |
| **Événements** | N/A | — | — |
| **CLI / Batch** | Variable d'environnement `GOOGLE_API_KEY` injectée au démarrage | Clé API Google Gemini | N/A (clé statique) |

### 1.2 En-têtes / métadonnées standards

Aucun en-tête HTTP géré côté application — le système est purement CLI sans serveur HTTP exposé. Les appels vers Google Gemini API et les URLs PDF utilisent les en-têtes par défaut de la librairie `requests` et du SDK Google.

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

> N/A — aucune interface HTTP exposée. Les erreurs remontent via exceptions Python non capturées vers le terminal.

### 1.5 Versioning

- **API HTTP** : N/A — aucune interface HTTP exposée.
- **Événements** : N/A — aucun topic/queue.
- **Politique de breaking change** : N/A — application CLI; toute rupture de signature Python est visible directement dans le code source.

---

## 2. APIs exposées (HTTP / gRPC / GraphQL)

Aucun endpoint exposé — le système est une application CLI sans serveur HTTP.

---

## 3. APIs consommées

> Une ligne par dépendance. Le détail des contrats vit chez le fournisseur — ici on note ce dont on dépend et comment on s'en protège.

| Service | Endpoint | Type | Criticité | Timeout | Retry | Circuit breaker |
|---|---|---|---|---|---|---|
| Google Gemini API | SDK `llama_index.llms.gemini` — modèle `models/gemini-2.5-flash` | Sync | Bloquante (sans LLM, pas de réponse possible) | Délégué au SDK Google | Aucun (à confirmer) | Aucun |
| URL PDF arbitraire | `GET <url>` via `requests` | Sync | Bloquante (PDF requis pour l'indexation) | 30 s ([`pdf_loader.py:21`](../pdf_loader.py)) | Aucun | Aucun |
| HuggingFace Hub | Téléchargement du modèle `BAAI/bge-small-en-v1.5` au premier lancement | Sync (init) | Bloquante au premier démarrage ; mis en cache ensuite | Délégué à la librairie HuggingFace | Aucun | Aucun |

**Modes dégradés** : Aucun mécanisme de fallback implémenté — toute erreur des services tiers (Gemini, URL PDF, HuggingFace Hub) remonte en exception Python non gérée et arrête le programme.

---

## 4. Événements / messages

Aucun topic ni queue — le système n'utilise pas de broker de messages.

---

## 5. Commandes CLI

| Commande | Rôle | Code source | Authentif | Effets de bord |
|---|---|---|---|---|
| `python main.py` | Lance une session interactive de questions/réponses sur un PDF hardcodé (`https://arxiv.org/pdf/2005.11401.pdf`) | [`../main.py`](../main.py) | Variable `GOOGLE_API_KEY` dans l'environnement | Télécharge le PDF si absent ; crée le répertoire `./storage/index_<md5>` pour mettre en cache les embeddings |

---

## 6. Jobs / batchs

Aucun job batch — le système est entièrement déclenché à la demande via la CLI.

---

## 7. Mode d'appel inter-composants (interne)

Tous les composants s'appellent directement en Python dans le même processus. Aucun bus interne, aucune sérialisation inter-processus.

```python
# Flux canonique d'initialisation (main.py → RAGEngine.__init__)
rag = RAGEngine(pdf_url="https://...")
# RAGEngine appelle successivement :
#   setup_advanced_text_processing()  → text_processor.py
#   initialize_gemini_llm()           → gemini_client.py
#   create_node_parser()              → text_processor.py
#   load_pdf_from_url(pdf_url)        → pdf_loader.py  (si pas de cache)
#   load_documents_from_pdf(pdf_path) → pdf_loader.py  (si pas de cache)

# Flux de requête (main.py → RAGEngine.query)
answer = rag.query(question)
# RAGEngine.query appelle self.query_engine.query(question)
# LlamaIndex gère en interne l'appel à Gemini via Settings.llm
```

**Points-clés** :

- Aucune propagation de contexte de corrélation — les `print()` sont les seuls logs.
- Aucune instrumentation (métriques, traces) — à ajouter si déploiement en production envisagé.
- Gestion des erreurs : aucune `try/except` applicative — toute exception remonte jusqu'à `main()` et arrête le programme.

### 7.1 Surface publique de `gemini_client`

> Source : [`../gemini_client.py`](../gemini_client.py). Singleton module-level `_state["llm_instance"]` partagé entre tous les appelants.

| Fonction | Signature | Rôle | Effets de bord |
|---|---|---|---|
| `initialize_gemini_llm` | `(model: str, temperature: float, max_tokens: int) -> Gemini` | Instancie et enregistre le singleton Gemini ; positionne `Settings.llm` de LlamaIndex | Mutate `_state` et `Settings.llm` |
| `get_llm` | `() -> Gemini` | Retourne le singleton existant ou l'initialise avec les valeurs par défaut | Peut muter `_state` si premier appel |
| `generate_text` | `(prompt: str, temperature: float \| None = None) -> str` | Génère une réponse texte brute sans contexte RAG ; lève `ValueError` si `prompt` vide | Aucun effet de bord sur le singleton (l'override de `temperature` crée une instance temporaire) |
| `summarize` | `(text: str, max_words: int = 150) -> str` | Résume `text` en environ `max_words` mots via `generate_text` ; lève `ValueError` si `text` vide | Aucun |
| `rerank_passages` | `(query: str, passages: list[str]) -> list[str]` | Rerank zéro-shot des passages par pertinence via Gemini ; retourne l'ordre original en cas d'échec de parsing | Aucun |
| `reset_llm` | `() -> None` | Vide le singleton — le prochain appel à `get_llm` réinitialise | Mutate `_state["llm_instance"] = None` |

**Valeurs par défaut** : `model = "models/gemini-2.5-flash"`, `temperature = 0.1`, `max_tokens = 1024`.

> (à confirmer) — `summarize` et `rerank_passages` ne sont pas encore appelées depuis `RAGEngine` ; elles constituent une API utilitaire disponible pour des usages futurs ou des scripts tiers.

---

## 8. Matrice d'appels

> Lignes = appelants. Colonnes = appelés. Cellule = mode d'appel. Vide = aucun appel.
> Légende : ✓ = inconditionnel · (c) = conditionnel (préciser dans le bloc concerné).

| Appelant ↓ / Appelé → | `RAGEngine` | `gemini_client` | `pdf_loader` | `text_processor` |
|---|---|---|---|---|
| **`main`** | ✓ | — | — | — |
| **`RAGEngine`** | — | ✓ | (c) | ✓ |

> (c) `RAGEngine → pdf_loader` : uniquement si aucun cache d'index n'existe pour l'URL donnée (`./storage/index_<md5>/`).

---

## 9. Formats des E/S physiques

> Pour tout fichier ou flux à structure fixe lu/écrit par le système.

### 9.1 `<url>.pdf` — fichier PDF d'entrée

- **Type** : PDF binaire
- **Encodage** : Binaire (géré par `PDFReader` de LlamaIndex)
- **Chemin local** : `<tempdir>/temp_rag_document.pdf` (fichier temporaire, écrasé à chaque lancement ; voir [`../pdf_loader.py`](../pdf_loader.py))
- **Taille typique** : (à confirmer)

### 9.2 `./storage/index_<md5>/` — cache des embeddings LlamaIndex

- **Type** : Répertoire LlamaIndex (`StorageContext.persist`) — fichiers JSON internes
- **Encodage** : UTF-8 / JSON
- **Chemin** : `./storage/index_<md5 de l'URL PDF>/` (créé par [`../rag_engine.py`](../rag_engine.py))
- **Taille typique** : (à confirmer)

> ⚠️ Le cache est keyed par le hash MD5 de l'URL du PDF (non sécurisé, `usedforsecurity=False`). Si l'URL change, un nouveau répertoire est créé sans supprimer l'ancien.

---

## 10. Transactions, commits, rollback

| Composant | Scope transactionnel | Commit | Rollback |
|---|---|---|---|
| `RAGEngine._save_index` | Par session d'indexation | `StorageContext.persist()` — écriture atomique de répertoire (à confirmer) | Aucun — si le processus est interrompu pendant la sauvegarde, le cache peut être corrompu |

> Cas critiques : interruption pendant `_save_index` peut laisser un répertoire de cache partiel qui sera rechargé comme s'il était valide au prochain lancement. (à confirmer)

---

## 11. Contraintes non-fonctionnelles par interface

| Interface | Latence p95 cible | Disponibilité | Idempotence | Re-entrance |
|---|---|---|---|---|
| `RAGEngine.query()` | (à confirmer) | Dépend de Google Gemini API | Oui — même question → même réponse (température 0.1, faible variabilité) | Non (instance partagée, usage séquentiel) |
| `load_pdf_from_url()` | ≤ 30 s (timeout configuré) | Dépend de l'URL distante | Non — pas de dédup | Oui |
| `_save_index()` | (à confirmer) | Locale (disque) | Non — écrase le répertoire existant | Non |

---

## 12. Tests de contrat

Aucun test de contrat formel implémenté. (à confirmer) — voir `tests/` si existants.

---

## 13. Recommandations actives

1. **Ajouter gestion d'erreur sur `load_pdf_from_url`** — un timeout ou une URL invalide arrête le programme sans message clair ; un `try/except requests.RequestException` améliorerait l'UX.
2. **Externaliser l'URL PDF de `main.py`** — actuellement hardcodée (`https://arxiv.org/pdf/2005.11401.pdf`) ; la passer en argument CLI (`sys.argv` ou `argparse`) rendrait le système générique.
3. **Vérifier l'intégrité du cache au chargement** — `_load_index` ne valide pas que le répertoire de cache est complet avant de l'utiliser.

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

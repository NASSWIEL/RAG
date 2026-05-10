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
| **Type d'interfaces** | CLI · appels natifs Python |
| **Schémas authoritatifs** | Code source — voir liens par section ci-dessous |

> **Résumé en 1 phrase** : Le système expose une interface CLI interactive (`main.py`) et une API Python publique (`RAGEngine`) ; il consomme l'API HTTP d'arxiv (téléchargement PDF) et l'API Google Gemini, et persiste les embeddings dans `./storage/`.

---

## 0. Inventaire global

| Type d'interface | Nombre | Schéma authoritatif |
|---|---|---|
| **Endpoints HTTP exposés** | 0 | — |
| **Endpoints HTTP consommés** | 2 | Voir §3 |
| **Topics / queues produits** | 0 | — |
| **Topics / queues consommés** | 0 | — |
| **Commandes CLI** | 1 | [main.py](../main.py) |
| **Jobs / batchs** | 0 | — |
| **Fichiers E/S (formats fixes)** | 1 | [§9](#9-formats-des-es-physiques) |
| **API Python publique** | 1 classe | [rag_engine.py](../rag_engine.py) |

---

## 1. Conventions transverses des interfaces

### 1.1 Authentification et autorisation

| Type d'interface | Mécanisme | Token / claims | Refresh / rotation |
|---|---|---|---|
| **API publique** | Aucune — pas d'endpoint HTTP exposé | — | — |
| **API interne (Gemini)** | API key (`GOOGLE_API_KEY` dans `config.py`) | Clé statique | Manuel — à renouveler hors du code |
| **Événements** | Aucun bus d'événements | — | — |
| **CLI** | Aucun — exécution locale | — | — |

### 1.2 En-têtes / métadonnées standards

> Le système n'expose pas d'API HTTP — cette section s'applique uniquement aux appels sortants vers les services tiers.

| En-tête / champ | Rôle | Direction | Obligatoire ? |
|---|---|---|---|
| `Authorization` | Clé API Google Gemini (gérée par le SDK) | Sortant | Oui |
| — | Pas d'en-têtes personnalisés supplémentaires | — | — |

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

> Non applicable — le système n'expose pas d'API HTTP. Les erreurs sont propagées sous forme d'exceptions Python standard.

### 1.5 Versioning

- **API HTTP** : Non applicable — aucun endpoint HTTP exposé.
- **API Python** : Pas de versioning formalisé — version alignée sur le tag du dépôt (à confirmer).
- **Événements** : Non applicable.
- **Politique de breaking change** : (à confirmer)

---

## 2. APIs exposées (HTTP / gRPC / GraphQL)

> Le système n'expose aucun endpoint HTTP, gRPC ou GraphQL. L'interface publique est exclusivement une API Python (`RAGEngine`) et une CLI. Voir §5 pour la CLI et §7 pour les appels internes.

### 2.1 `RAGEngine` — API Python publique

#### `RAGEngine(pdf_url, storage_dir="./storage")` — Constructeur : initialise l'index vectoriel

| Méta | Valeur |
|---|---|
| **Code source** | [rag_engine.py](../rag_engine.py) |
| **Schéma** | — |
| **Auth requise** | Clé Google API dans `config.py` |
| **Idempotent** | Oui (détecte le cache existant par hash MD5 de l'URL) |
| **Rate-limited** | Non (côté système) |
| **SLO p95** | (à confirmer) — première exécution : téléchargement + embedding ; exécutions suivantes : chargement du cache uniquement |

**Paramètres**

| Paramètre | Type | Obligatoire | Défaut | Description |
|---|---|---|---|---|
| `pdf_url` | `str` | Oui | — | URL publique du PDF à indexer |
| `storage_dir` | `str` | Non | `"./storage"` | Répertoire de persistance des embeddings |

**Effets de bord**

- Télécharge le PDF depuis `pdf_url` (HTTP GET, timeout 30 s) vers un fichier temporaire.
- Génère les embeddings via `BAAI/bge-small-en-v1.5` (HuggingFace) si non présents en cache.
- Persiste l'index vectoriel dans `{storage_dir}/index_{md5(pdf_url)}/`.
- Configure globalement `llama_index.core.Settings` (embed model, LLM, chunk size).

**Pièges / particularités**

- ⚠️ `Settings` est un singleton global de LlamaIndex — instancier plusieurs `RAGEngine` dans le même processus écrase les paramètres précédents.
- ⚠️ La clé Gemini est lue depuis `config.py` à l'import — un fichier manquant lève `ImportError` immédiatement.

#### `RAGEngine.query(question: str) -> str` — Recherche sémantique et génération de réponse

| Méta | Valeur |
|---|---|
| **Code source** | [rag_engine.py](../rag_engine.py) |
| **Idempotent** | Non (la réponse dépend du LLM Gemini, non déterministe) |
| **Rate-limited** | Limité par les quotas de l'API Google Gemini |
| **SLO p95** | (à confirmer) |

**Paramètres**

| Paramètre | Type | Obligatoire | Description |
|---|---|---|
| `question` | `str` | Oui | Question en langage naturel |

**Retour**

`str` — Réponse générée par Gemini à partir des 3 chunks les plus similaires (`similarity_top_k=3`, `response_mode="compact"`).

**Effets de bord**

- Appel sortant à l'API Google Gemini (`models/gemini-2.5-flash`, `temperature=0.1`).

**Pièges / particularités**

- ⚠️ Toute erreur de l'API Gemini (quota, réseau) est propagée sans gestion dans le code source actuel.

---

## 3. APIs consommées

> Une ligne par dépendance. Le détail des contrats vit chez le fournisseur.

| Service | Endpoint | Type | Criticité | Timeout | Retry | Circuit breaker |
|---|---|---|---|---|---|---|
| Source PDF (ex. arxiv.org) | `GET {pdf_url}` | Sync HTTP | Bloquante (première exécution) | 30 s ([pdf_loader.py](../pdf_loader.py)) | Aucun | Aucun |
| Google Gemini API | `models/gemini-2.5-flash` via SDK | Sync (gRPC/HTTP2) | Bloquante à chaque query | (à confirmer — géré par le SDK) | (à confirmer) | Aucun |

**Modes dégradés** :
- **Source PDF KO** : `requests.get` lève une exception non interceptée — le processus s'arrête. Pas de fallback.
- **Google Gemini KO** : exception propagée à l'appelant sans gestion de retry ni fallback.

---

## 4. Événements / messages

> Le système ne produit ni ne consomme de topics ou queues. Aucun broker de messages n'est utilisé. Cette section est sans objet pour la version actuelle.

---

## 5. Commandes CLI

| Commande | Rôle | Code source | Authentif | Effets de bord |
|---|---|---|---|---|
| `python main.py` | Démarre une session interactive de questions-réponses sur un PDF | [main.py](../main.py) | Clé `GOOGLE_API_KEY` dans `config.py` | Téléchargement PDF, écriture dans `./storage/`, appels API Gemini |

**Comportement** :
- Lit `pdf_url` codé en dur dans `main.py` (`https://arxiv.org/pdf/2005.11401.pdf` par défaut).
- Lance une boucle REPL — saisir `q` pour quitter.
- Chaque question déclenche `RAGEngine.query()` et affiche la réponse en stdout.

---

## 6. Jobs / batchs

> Le système ne contient aucun job planifié ni batch. Cette section est sans objet pour la version actuelle.

---

## 7. Mode d'appel inter-composants (interne)

Tous les composants s'exécutent dans le même processus Python. La chaîne d'appel est la suivante :

```python
# main.py
rag = RAGEngine(pdf_url)          # initialise : pdf_loader → text_processor → gemini_client
answer = rag.query(user_query)    # interroge : VectorStoreIndex → Gemini LLM
```

**Points-clés** :

- Pas de propagation de contexte de corrélation entre composants.
- Pas d'instrumentation (métriques, spans) — uniquement des `print()` en stdout.
- Gestion d'erreur : aucune — toute exception remonte jusqu'à `main()` et arrête le processus.

---

## 8. Matrice d'appels

> Lignes = appelants. Colonnes = appelés. Cellule = mode d'appel. Vide = aucun appel.
> Légende : ✓ = inconditionnel · (c) = conditionnel (lors de la première indexation uniquement).

| Appelant ↓ / Appelé → | `RAGEngine` | `pdf_loader` | `text_processor` | `gemini_client` | Google Gemini API | Source PDF (HTTP) |
|---|---|---|---|---|---|---|
| **`main.py`** | ✓ | — | — | — | — | — |
| **`RAGEngine`** | — | (c) | ✓ | ✓ | ✓ | — |
| **`pdf_loader`** | — | — | — | — | — | (c) |

> `pdf_loader` appelle la source PDF uniquement lors de la première indexation (absence de cache).

---

## 9. Formats des E/S physiques

### 9.1 `./storage/index_{md5(pdf_url)}/` — Index vectoriel persisté (sortie / cache)

- **Type** : Répertoire contenant des fichiers JSON générés par LlamaIndex (`StorageContext.persist`)
- **Encodage** : UTF-8
- **Délimiteur** : N/A (JSON)
- **Taille typique** : Variable selon la taille du PDF et le nombre de chunks

| Fichier | Type | Description |
|---|---|---|
| `docstore.json` | JSON | Stockage des nœuds de texte |
| `index_store.json` | JSON | Métadonnées de l'index vectoriel |
| `vector_store.json` | JSON | Vecteurs d'embedding |
| `graph_store.json` | JSON | Graphe de relations (vide si non utilisé) |

> ⚠️ La clé de cache est le hash MD5 de l'URL du PDF — changer l'URL force une réindexation complète, même si le contenu du PDF est identique.

---

## 10. Transactions, commits, rollback

| Composant | Scope transactionnel | Commit | Rollback |
|---|---|---|---|
| `RAGEngine._save_index` | Par indexation complète | Fichiers écrits via `StorageContext.persist` | Aucun — un crash en cours d'écriture peut laisser un cache partiel |

> ⚠️ Cas critique : si le processus est interrompu pendant `_save_index`, le répertoire de cache peut être partiellement écrit. Au prochain démarrage, LlamaIndex tentera de charger ce cache partiel, ce qui peut lever une exception. Solution manuelle : supprimer le répertoire `./storage/index_{hash}/`.

---

## 11. Contraintes non-fonctionnelles par interface

| Interface | Latence p95 cible | Disponibilité | Idempotence | Re-entrance |
|---|---|---|---|---|
| `RAGEngine.__init__` (avec cache) | (à confirmer) | Locale — dépend du FS | Oui — hash MD5 de l'URL | Non (Settings global) |
| `RAGEngine.__init__` (sans cache) | (à confirmer) — dépend taille PDF + GPU | Dépend arxiv.org + Gemini API | Oui | Non |
| `RAGEngine.query` | (à confirmer) — dépend Gemini API | Dépend de la disponibilité de l'API Google | Non (LLM non déterministe) | Oui |
| CLI `python main.py` | N/A (interactif) | Locale | N/A | Non |

---

## 12. Tests de contrat

| Interface | Outil | Localisation | Côté |
|---|---|---|---|
| `RAGEngine.query` | pytest | [tests/](../tests/) | Provider (à confirmer) |

> ⚠️ La couverture de tests de contrat est à confirmer — aucun test de contrat formel (Pact, dredd) n'a été identifié dans le code source.

---

## 13. Recommandations actives

1. **Ajouter une gestion d'erreur dans `RAGEngine.query` et `load_pdf_from_url`** — Actuellement toute exception (quota Gemini, timeout réseau, PDF inaccessible) arrête le processus. Un retry avec backoff et un message d'erreur explicite amélioreraient la robustesse. (à confirmer)
2. **Externaliser `pdf_url` hors du code** — La valeur est codée en dur dans `main.py`. Un argument CLI ou une variable d'environnement permettrait de changer la source sans modifier le code. (à confirmer)
3. **Protéger `config.py` contre le commit de la clé API** — Vérifier que `config.py` est dans `.gitignore` ou utiliser une variable d'environnement. (à confirmer)
4. **Gérer le cache partiel après crash** — Voir §10 : un crash pendant `_save_index` peut corrompre le cache. Ajouter une écriture atomique (temp dir + rename) ou une vérification d'intégrité au chargement. (à confirmer)

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

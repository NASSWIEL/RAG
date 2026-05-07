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
| **Dernière mise à jour** | 2026-05-07 |
| **Mise à jour par** | agent IA |
| **PR de référence** | (à confirmer) |
| **Type d'interfaces** | Appels natifs Python (lib pure) |
| **Schémas authoritatifs** | Signatures Python dans [`pdf_loader.py`](../pdf_loader.py) et [`rag_engine.py`](../rag_engine.py) |

> **Résumé en 1 phrase** : Le système expose une API Python pour télécharger un PDF depuis une URL, en extraire le texte sous forme de Documents llama-index, puis interroger un index vectoriel via `RAGEngine.query`.

---

## 0. Inventaire global

| Type d'interface | Nombre | Schéma authoritatif |
|---|---|---|
| **Endpoints HTTP exposés** | 0 | — |
| **Endpoints HTTP consommés** | 1 (téléchargement PDF via `requests`) | [`pdf_loader.py`](../pdf_loader.py) |
| **Topics / queues produits** | 0 | — |
| **Topics / queues consommés** | 0 | — |
| **Commandes CLI** | 0 | — |
| **Jobs / batchs** | 0 | — |
| **Fichiers E/S (formats fixes)** | 1 (fichier PDF en entrée) | [§9](#9-formats-des-es-physiques) |
| **API Python publique** | 3 fonctions / méthodes | [§7](#7-mode-dappel-inter-composants-interne) |

---

## 1. Conventions transverses des interfaces

### 1.1 Authentification et autorisation

| Type d'interface | Mécanisme | Token / claims | Refresh / rotation |
|---|---|---|---|
| **API Python (lib)** | Aucune — lib appelée en-process | — | — |
| **HTTP consommé (PDF URL)** | Aucune (URL publique attendue) | — | — |
| **LLM Gemini** | Clé API via variable d'environnement (à confirmer) | — | — |

### 1.2 En-têtes / métadonnées standards

> Pas d'interface HTTP exposée. Les en-têtes ci-dessous concernent les appels HTTP sortants (téléchargement PDF).

| En-tête / champ | Rôle | Direction | Obligatoire ? |
|---|---|---|---|
| `content-length` | Calcul de progression du téléchargement | Entrant (réponse serveur) | Non (optionnel) |

### 1.3 Exceptions normalisées

> Lib Python — pas de codes HTTP exposés. Les exceptions ci-dessous constituent le contrat d'erreur public.

| Exception | Levée par | Condition | Idempotente ? |
|---|---|---|---|
| `requests.HTTPError` | `load_pdf_from_url` | Réponse non-2xx du serveur PDF | Oui |
| `ValueError` | `load_documents_from_pdf` | Aucun texte extractible dans le PDF | Oui |
| `ImportError` | `load_documents_from_pdf` | Dépendance `pypdf` absente | Oui |
| `MemoryError` | `RAGEngine.__init__` | Mémoire insuffisante lors de l'embedding | Non |
| `OSError` | `load_pdf_from_url` | Echec d'écriture du fichier temporaire | — |

### 1.4 Format de réponse d'erreur

> Lib Python — les erreurs remontent via des exceptions standard Python (voir §1.3). Pas d'enveloppe JSON.

### 1.5 Versioning

- **API Python** : versioning via `pyproject.toml` (à confirmer)
- **Politique de breaking change** : (à confirmer)

---

## 2. APIs exposées (HTTP / gRPC / GraphQL)

> Aucun endpoint HTTP/gRPC/GraphQL exposé — le système est une lib Python pure. Voir §7 pour les contrats d'appel natifs.

---

## 3. APIs consommées

| Service | Endpoint | Type | Criticité | Timeout | Retry | Circuit breaker |
|---|---|---|---|---|---|---|
| Serveur PDF distant | `GET <pdf_url>` (URL configurable) | Sync HTTP streaming | Bloquante | connect 10 s / read 60 s | Aucun (à confirmer) | Aucun (à confirmer) |
| Gemini LLM | (à confirmer) | Sync | Bloquante pour `query` | (à confirmer) | (à confirmer) | (à confirmer) |

**Modes dégradés** : En cas d'échec du téléchargement PDF, `requests.HTTPError` est propagée à l'appelant. Le fichier temporaire est supprimé avant propagation. Aucune stratégie de retry ni de fallback implémentée (à confirmer).

---

## 4. Événements / messages

> Aucun broker de messages utilisé — le système est une lib Python synchrone. Cette section est sans objet.

---

## 5. Commandes CLI

> Aucune commande CLI exposée.

---

## 6. Jobs / batchs

> Aucun job ou batch planifié.

---

## 7. Mode d'appel inter-composants (interne)

> C'est le cœur du contrat pour cette lib. Trois points d'entrée publics.

### 7.1 `load_pdf_from_url(url: str) -> str`

| Méta | Valeur |
|---|---|
| **Code source** | [`pdf_loader.py:14`](../pdf_loader.py#L14) |
| **Responsabilité** | Télécharge un PDF depuis une URL par streaming et retourne le chemin d'un fichier temporaire |
| **Idempotente** | Oui (crée un nouveau fichier temporaire à chaque appel) |

**Paramètres**

| Paramètre | Type | Obligatoire | Description |
|---|---|---|---|
| `url` | `str` | Oui | URL HTTP(S) publique du PDF à télécharger |

**Retour** : `str` — chemin absolu du fichier temporaire `.pdf`. L'appelant est responsable de sa suppression.

**Exceptions**

| Exception | Condition |
|---|---|
| `requests.HTTPError` | Réponse HTTP non-2xx |
| `OSError` | Echec d'écriture du fichier temporaire |

**Pièges / particularités**

- ⚠️ Le fichier temporaire **doit être supprimé par l'appelant** (`os.unlink(path)`). En cas d'exception levée par cette fonction, le fichier est déjà nettoyé en interne.
- Timeout fixe : connect 10 s / read 60 s (constante `_REQUEST_TIMEOUT` dans [`pdf_loader.py`](../pdf_loader.py)).
- Chunk de téléchargement : 8 MB (constante `_DOWNLOAD_CHUNK`).

---

### 7.2 `load_documents_from_pdf(pdf_path: str) -> list[Document]`

| Méta | Valeur |
|---|---|
| **Code source** | [`pdf_loader.py:49`](../pdf_loader.py#L49) |
| **Responsabilité** | Extrait le texte d'un PDF sous forme de `Document` llama-index, traité par batches de pages |
| **Idempotente** | Oui |

**Paramètres**

| Paramètre | Type | Obligatoire | Description |
|---|---|---|---|
| `pdf_path` | `str` | Oui | Chemin local vers le fichier PDF |

**Retour** : `list[Document]` — liste de `llama_index.core.Document`, un par page contenant du texte. Métadonnées par document : `page_label`, `file_path`, `total_pages`.

**Exceptions**

| Exception | Condition |
|---|---|
| `ValueError` | Aucun texte extractible dans le PDF |
| `ImportError` | Dépendance `pypdf` absente (message : `pypdf is required: uv add pypdf`) |

**Pièges / particularités**

- ⚠️ Les pages sans texte (ex. pages d'images) sont silencieusement ignorées. Un PDF 100 % image lèvera `ValueError`.
- `gc.collect()` est appelé entre chaque batch de 50 pages (`_PAGE_BATCH`) pour limiter la mémoire de pointe.

---

### 7.3 `RAGEngine(pdf_url: str, storage_dir: str = "./storage")`

| Méta | Valeur |
|---|---|
| **Code source** | [`rag_engine.py:12`](../rag_engine.py#L12) |
| **Responsabilité** | Point d'entrée principal : construit ou charge l'index vectoriel, puis répond aux questions |

**Constructeur**

| Paramètre | Type | Obligatoire | Description |
|---|---|---|---|
| `pdf_url` | `str` | Oui | URL publique du PDF source |
| `storage_dir` | `str` | Non (défaut `"./storage"`) | Répertoire de persistance de l'index vectoriel |

**Comportement** : si un index pour ce PDF existe déjà dans `storage_dir` (clé = MD5 de l'URL), il est chargé ; sinon le PDF est téléchargé, indexé, et persisté.

**Exceptions propagées depuis `__init__`**

| Exception | Condition |
|---|---|
| `requests.HTTPError` | Echec de téléchargement PDF |
| `ValueError` | PDF sans texte extractible |
| `ImportError` | `pypdf` absent |
| `MemoryError` | Mémoire insuffisante lors de l'embedding |

#### Méthode publique : `query(question: str) -> str`

| Méta | Valeur |
|---|---|
| **Code source** | [`rag_engine.py:85`](../rag_engine.py#L85) |

**Paramètres**

| Paramètre | Type | Obligatoire | Description |
|---|---|---|---|
| `question` | `str` | Oui | Question en langage naturel |

**Retour** : `str` — réponse générée par le LLM Gemini à partir des 3 chunks les plus similaires (`similarity_top_k=3`, `response_mode="compact"`).

**Pièges / particularités**

- ⚠️ Les méthodes `_get_index_path`, `_save_index` et `_load_index` sont privées — ne pas les appeler directement.

```python
# Patron d'utilisation canonique
from rag_engine import RAGEngine

engine = RAGEngine(pdf_url="https://example.com/doc.pdf", storage_dir="./storage")
answer = engine.query("Quel est le sujet principal du document ?")
print(answer)
```

---

## 8. Matrice d'appels

> Légende : ✓ = inconditionnel · (c) = conditionnel · — = aucun appel

| Appelant ↓ / Appelé → | `load_pdf_from_url` | `load_documents_from_pdf` | `RAGEngine.query` | Gemini LLM |
|---|---|---|---|---|
| **Code applicatif (appelant externe)** | (c) | (c) | ✓ | — |
| **`RAGEngine.__init__`** | (c) | (c) | — | — |
| **`RAGEngine.query`** | — | — | — | ✓ |

> `RAGEngine.__init__` appelle `load_pdf_from_url` + `load_documents_from_pdf` uniquement si aucun index en cache.

---

## 9. Formats des E/S physiques

### 9.1 Fichier PDF — entrée

- **Type** : PDF (binary)
- **Source** : URL HTTP(S) distante, téléchargé dans un fichier temporaire système
- **Taille typique** : (à confirmer)
- **Contrainte** : doit contenir du texte extractible (pas uniquement des images scannées)

| Champ metadata llama-index | Type | Description |
|---|---|---|
| `page_label` | `str` | Numéro de page (base 1) |
| `file_path` | `str` | Chemin local du PDF temporaire |
| `total_pages` | `str` | Nombre total de pages du document |

### 9.2 Index vectoriel — sortie (cache)

- **Type** : Répertoire persisté par llama-index (`StorageContext.persist`)
- **Localisation** : `<storage_dir>/index_<md5_url>/`
- **Contenu** : fichiers internes llama-index (à confirmer — docstore, vector store, graph store)
- **Clé de cache** : MD5 (non sécurisé, à usage d'identification uniquement) de l'URL PDF

---

## 10. Transactions, commits, rollback

| Composant | Scope transactionnel | Commit | Rollback |
|---|---|---|---|
| `load_pdf_from_url` | Aucun (I/O fichier) | — | Suppression du fichier temporaire sur exception |
| `RAGEngine.__init__` (indexation) | Aucun | Persistance via `StorageContext.persist` après indexation complète | Aucun — index partiel possible si crash pendant `persist` |
| `RAGEngine.query` | Aucun | — | — |

> ⚠️ Si le processus est interrompu pendant `_save_index`, le répertoire d'index peut être partiel. Supprimer le répertoire `<storage_dir>/index_<md5_url>/` pour forcer une ré-indexation.

---

## 11. Contraintes non-fonctionnelles par interface

| Interface | Latence p95 cible | Disponibilité | Idempotence | Re-entrance |
|---|---|---|---|---|
| `load_pdf_from_url` | Dépend du réseau et de la taille du PDF | Dépend du serveur distant | Oui | Non (fichier temporaire différent à chaque appel) |
| `load_documents_from_pdf` | (à confirmer) | N/A (local) | Oui | Oui |
| `RAGEngine.__init__` (cache chaud) | (à confirmer) | N/A (local) | Oui | Non (état interne modifié) |
| `RAGEngine.__init__` (cache froid) | Dépend du réseau + taille PDF + embedding | N/A | Oui | Non |
| `RAGEngine.query` | (à confirmer — dépend de Gemini) | Dépend de Gemini | Non (LLM non déterministe) | Oui (lecture seule) |

---

## 12. Tests de contrat

| Interface | Outil | Localisation | Côté |
|---|---|---|---|
| `load_pdf_from_url` | (à confirmer) | (à confirmer) | (à confirmer) |
| `load_documents_from_pdf` | (à confirmer) | (à confirmer) | (à confirmer) |
| `RAGEngine.query` | (à confirmer) | (à confirmer) | (à confirmer) |

---

## 13. Recommandations actives

1. **Ajouter une stratégie de retry sur `load_pdf_from_url`** — Un seul échec réseau transitoire fait planter l'initialisation complète de `RAGEngine`. Envisager `tenacity` ou similaire.
2. **Protéger `_save_index` contre les index partiels** — En cas d'interruption pendant la persistance, l'index peut être corrompu silencieusement. Écrire dans un répertoire temporaire et renommer atomiquement.
3. **Documenter la variable d'environnement Gemini** — Le mécanisme d'authentification Gemini n'est pas documenté dans le code visible (à confirmer via `gemini_client.py`).

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

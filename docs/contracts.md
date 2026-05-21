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

# Contrats d'interface — RAG Q&A

| Champ | Valeur |
|---|---|
| **Dernière mise à jour** | 2026-05-21 |
| **Mise à jour par** | agent IA doc-patcher |
| **PR de référence** | (à confirmer) |
| **Type d'interfaces** | CLI / appels natifs Python |
| **Schémas authoritatifs** | Code source dans les modules racine (`gemini_client.py`, `pdf_loader.py`, `text_processor.py`, `rag_engine.py`, `main.py`) |

> **Résumé en 1 phrase** : Le système expose une interface CLI interactive pour interroger un PDF arxiv via RAG, consomme l'API Gemini et un modèle HuggingFace, et persiste l'index vectoriel localement dans `./storage`.

---

## 0. Inventaire global

| Type d'interface | Nombre | Schéma authoritatif |
|---|---|---|
| **Endpoints HTTP exposés** | 0 | N/A — application CLI locale |
| **Endpoints HTTP consommés** | 2 | Gemini API (via `llama_index.llms.gemini`), arxiv PDF HTTP GET |
| **Topics / queues produits** | 0 | N/A |
| **Topics / queues consommés** | 0 | N/A |
| **Commandes CLI** | 1 | [§5](#5-commandes-cli) |
| **Jobs / batchs** | 0 | N/A |
| **Fichiers E/S (formats fixes)** | 0 | Voir §9 (index vectoriel persisté, pas de format fixe structuré) |

---

## 1. Conventions transverses des interfaces

Application CLI mono-processus locale. Il n'y a pas d'API HTTP exposée, pas d'événements, pas d'en-têtes réseau standards gérés par l'application elle-même.

### 1.1 Authentification et autorisation

| Type d'interface | Mécanisme | Remarque |
|---|---|---|
| **CLI** | Aucune authentification applicative | L'accès à la machine suffit |
| **Appel Gemini API** | Variable d'environnement `GEMINI_API_KEY` | Lue au démarrage dans `gemini_client.py` — si absente, `KeyError` levée |
| **Téléchargement PDF** | Aucune (URL publique arxiv) | Timeout=30s |

### 1.2 En-têtes / métadonnées standards

N/A — pas d'API HTTP exposée. Les appels sortants vers l'API Gemini sont gérés par la librairie `llama_index.llms.gemini` (à confirmer).

### 1.3 Codes d'erreur

Pas de codes HTTP normalisés. Les erreurs se propagent comme exceptions Python standard :

| Situation | Exception attendue |
|---|---|
| `GEMINI_API_KEY` absente | `KeyError` dans `initialize_gemini_llm()` / `generate_text()` |
| PDF inaccessible / timeout | `Exception` dans `load_pdf_from_url()` (timeout=30s) |
| PDF illisible | Exception PDFReader dans `load_documents_from_pdf()` |
| Index corrompu sur disque | Exception LlamaIndex dans `_load_index()` |
| `prompt` vide ou whitespace dans `generate_text()` | `ValueError("prompt must not be empty")` |
| `text` vide ou whitespace dans `summarize()` | `ValueError("text must not be empty")` |

### 1.4 Format de réponse

Sortie console texte brut (stdout). `RAGEngine.query()` retourne un `str` Python.

### 1.5 Versioning

Pas de versioning d'API. Le modèle Gemini utilisé est fixé à `models/gemini-2.5-flash` dans `gemini_client.py`.

---

## 2. APIs exposées (HTTP / gRPC / GraphQL)

Aucune API HTTP/gRPC/GraphQL exposée — application CLI locale uniquement.

---

## 3. APIs consommées

| Service | Appel | Type | Criticité | Timeout | Retry | Circuit breaker |
|---|---|---|---|---|---|---|
| **Gemini API** | Inférence LLM via `llama_index.llms.gemini` (modèle `models/gemini-2.5-flash`) | Sync | Bloquante — sans LLM, pas de réponse | (à confirmer) géré par la lib | (à confirmer) géré par la lib | Aucun |
| **arxiv HTTP** | `GET https://arxiv.org/pdf/2005.11401.pdf` | Sync | Bloquante au premier lancement ; ignorée si cache présent | 30s (fixé dans `load_pdf_from_url`) | Aucun | Aucun |
| **HuggingFace Hub** | Téléchargement modèle `BAAI/bge-small-en-v1.5` au premier lancement | Sync | Bloquante au premier lancement ; mise en cache locale ensuite | (à confirmer) géré par la lib | (à confirmer) | Aucun |

**Modes dégradés** : Aucun fallback implémenté. Si l'API Gemini est KO ou si le PDF arxiv est inaccessible, l'application lève une exception non gérée. Le cache local de l'index vectoriel (`./storage`) évite le re-téléchargement du PDF et le re-calcul des embeddings pour les lancements suivants.

---

## 4. Événements / messages

Aucun broker de messages, aucun topic produit ou consommé. Application CLI sans bus d'événements.

---

## 5. Commandes CLI

| Commande | Rôle | Code source | Authentif | Effets de bord |
|---|---|---|---|---|
| `python main.py` | Démarre la boucle interactive Q&A RAG sur le PDF arxiv 2005.11401 | [main.py](../main.py) | `GEMINI_API_KEY` dans l'environnement | Télécharge le PDF si absent, télécharge le modèle HuggingFace si absent, persiste l'index vectoriel dans `./storage/index_<md5>` |

**Utilisation** :

```
python main.py
# > Your question: <question en langage naturel>
# > Answer: <réponse générée>
# Taper 'q' pour quitter.
```

⚠️ Le premier lancement peut durer plusieurs minutes (téléchargement PDF + modèle + calcul des embeddings). Les lancements suivants chargent l'index depuis le cache local.

---

## 6. Jobs / batchs

Aucun job batch ni tâche planifiée.

---

## 7. Mode d'appel inter-composants (interne)

Tous les composants s'exécutent dans le même processus Python. Flux d'appel principal :

```python
# main.py
rag = RAGEngine(pdf_url="https://arxiv.org/pdf/2005.11401.pdf")
answer = rag.query(question)  # -> str

# RAGEngine.__init__ orchestre :
#   setup_advanced_text_processing()  -> HuggingFaceEmbedding (BAAI/bge-small-en-v1.5, chunk_size=512, overlap=50)
#   initialize_gemini_llm()           -> Gemini (models/gemini-2.5-flash), enregistré dans Settings.llm
#   create_node_parser()              -> SentenceSplitter(chunk_size=512, chunk_overlap=50)
#   load_pdf_from_url(url)            -> str (chemin fichier temp local)
#   load_documents_from_pdf(path)     -> list[Document]
#   VectorStoreIndex.from_documents() -> index (persiste dans ./storage/index_<md5>)
#
# RAGEngine.query(question) :
#   self.query_engine.query(question) -> Response  (similarity_top_k=3, response_mode="compact")
#   -> str
```

**Interface publique de `gemini_client.py`** (à partir du diff) :

| Fonction | Signature | Retour | Remarque |
|---|---|---|---|
| `initialize_gemini_llm` | `(model=_DEFAULT_MODEL, temperature=0.1, max_tokens=1024) -> Gemini` | instance `Gemini` | Enregistre dans `Settings.llm` **et** dans le singleton `_state["llm"]` |
| `get_llm` | `() -> Gemini` | instance `Gemini` | Retourne le singleton ; appelle `initialize_gemini_llm()` si non initialisé |
| `generate_text` | `(prompt: str, temperature: float \| None = None) -> str` | texte généré | ⚠️ Si `temperature` est fourni, instancie un nouveau `Gemini` temporaire (hors singleton) à chaque appel |
| `summarize` | `(text: str, max_words: int = 150) -> str` | résumé en prose | Délègue à `generate_text()` |
| `rerank_passages` | `(query: str, passages: list[str]) -> list[str]` | passages réordonnés | Appelle le LLM avec `temperature=0.0` ; retourne `passages` inchangé en cas d'échec de parsing |
| `reset_llm` | `() -> None` | — | Remet `_state["llm"]` à `None` ; utile pour les tests |

**Points-clés** :

- `Settings.llm` et `Settings.node_parser` sont mutés globalement dans `initialize_gemini_llm()` et `RAGEngine.__init__` — ⚠️ pas thread-safe si plusieurs `RAGEngine` sont instanciés en parallèle.
- `gemini_client.py` maintient un singleton module-level (`_state["llm"]`). `get_llm()` évite les ré-initialisations, mais l'état est partagé sur tout le processus — ne pas combiner avec des appels qui changent de modèle sans `reset_llm()`.
- `generate_text(temperature=x)` crée une instance `Gemini` temporaire distincte ; la clé API est relue depuis `os.environ` à chaque appel dans ce cas.
- Le cache de l'index est indexé par le hash MD5 de l'URL du PDF : `./storage/index_<md5(pdf_url)>`. Changer l'URL force un re-calcul complet.
- Pas de propagation de contexte de tracing ni de métriques applicatives.

---

## 8. Matrice d'appels

> Lignes = appelants. Colonnes = appelés. Cellule = mode d'appel. Vide = aucun appel direct.
> Légende : ✓ = inconditionnel · (c) = conditionnel

| Appelant ↓ / Appelé → | `main.py` | `rag_engine.py` | `gemini_client.py` | `pdf_loader.py` | `text_processor.py` |
|---|---|---|---|---|---|
| **`main.py`** | — | ✓ | — | — | — |
| **`rag_engine.py`** | — | — | ✓ | ✓ | ✓ |
| **`gemini_client.py`** | — | — | — | — | — |
| **`pdf_loader.py`** | — | — | — | — | — |
| **`text_processor.py`** | — | — | — | — | — |

---

## 9. Formats des E/S physiques

Pas de format de fichier structuré à documenter. Les seuls fichiers produits sont :

- **Entrée** : PDF binaire téléchargé depuis arxiv, sauvegardé dans un fichier temporaire par `load_pdf_from_url()`. Format opaque, parsé par `PDFReader` de LlamaIndex.
- **Sortie / cache** : Répertoire `./storage/index_<md5(pdf_url)>/` — fichiers internes LlamaIndex (JSON + binaires vectoriels). Structure propriétaire à la librairie, ne pas modifier manuellement.

---

## 10. Transactions, commits, rollback

Pas de base de données. Pas de transactions applicatives.

La seule persistance est l'index vectoriel sur disque (`_save_index`). En cas d'échec pendant `_save_index`, le répertoire partiellement écrit peut rester sur disque — au prochain lancement, si le répertoire existe mais est corrompu, une exception LlamaIndex sera levée. Workaround : supprimer manuellement `./storage/index_<md5>/`.

---

## 11. Contraintes non-fonctionnelles par interface

| Interface | Latence p95 cible | Idempotence | Re-entrance |
|---|---|---|---|
| `python main.py` (1er lancement) | (à confirmer) — dépend du réseau et de la machine | Oui — re-lancer reconstruit le même index | Non — mutations globales de `Settings` |
| `RAGEngine.query()` | (à confirmer) — dépend de l'API Gemini | Non garanti — LLM non-déterministe | Non thread-safe |
| Téléchargement PDF (arxiv) | timeout=30s | Oui | Oui |

---

## 12. Tests de contrat

Aucun test de contrat formel (Pact, dredd, etc.) à ce jour. Les tests d'intégration éventuels seraient dans `tests/` (à confirmer).

---

## 13. Recommandations actives

1. **Gérer `KeyError` sur `GEMINI_API_KEY`** — Actuellement l'absence de la variable d'environnement lève une `KeyError` non gérée. Remplacer par un message d'erreur explicite au démarrage.
2. **Ajouter retry sur l'appel arxiv** — Le téléchargement PDF n'a aucune logique de retry ; un timeout réseau fait planter l'application.
3. **Isoler les mutations de `Settings`** — `initialize_gemini_llm()` et `RAGEngine.__init__` mutent l'état global `Settings` de LlamaIndex, rendant l'usage multi-instances impossible. (à confirmer si cas d'usage futur)

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

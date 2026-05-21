<!--
TEMPLATE — Architecture
=======================
Public cible : (1) une IA assistante de développement qui doit raisonner sur le système
sans relire tout le code à chaque tâche, (2) un humain qui prend le projet en cours.

Ce document est VIVANT : il est mis à jour avant chaque PR (par un skill IA, validé par
un humain). Il décrit le système TEL QU'IL EST, pas tel qu'on aimerait qu'il soit. Les
intentions et décisions à venir vont dans les ADR ou dans la backlog, pas ici.

Garde-fous d'écriture :
- Tout chiffre / nom de composant doit être vérifiable depuis le code ou un artefact
  versionné. Si déduit ou hypothétique, marquer explicitement "(à confirmer)".
- Distinguer "observé" / "décidé" / "ouvert".
- Toute affirmation forte → renvoyer vers un fichier source (code, ADR, schéma).
- Préférer les tableaux et listes courtes à la prose. Une PR doit pouvoir patcher 3 lignes,
  pas un paragraphe.

Le bloc "Mode d'emploi" en fin de fichier détaille la marche à suivre pour l'IA et le relecteur.
-->

# Architecture — RAG

| Champ | Valeur |
|---|---|
| **Dernière mise à jour** | 2026-05-21 |
| **Mise à jour par** | agent IA (doc-patcher) |
| **PR de référence** | (à confirmer) |
| **Version applicative** | (à confirmer) |
| **Statut du document** | Brouillon |

> **Résumé en 1 phrase** : Système RAG en ligne de commande qui télécharge un PDF depuis une URL, en construit un index vectoriel persisté sur disque, puis répond en mode interactif à des questions en langage naturel via Gemini 2.5 Flash.

---

## 1. Vue d'ensemble

### 1.1 Nature du système

| Dimension | Valeur |
|---|---|
| **Type** | CLI |
| **Mode** | Synchrone |
| **Multi-tenant** | Non — usage mono-utilisateur local |
| **Stateful** | Cache disque sous `./storage/index_{md5(url)}/` ; sans cache, traitement stateless |
| **Utilisateurs cibles** | Développeurs / utilisateurs locaux |
| **Volumétrie typique** | Usage interactif ponctuel — pas de cible RPS |

### 1.2 Flux principal

L'utilisateur lance `main.py`. `RAGEngine.__init__` démarre :

1. `text_processor.setup_advanced_text_processing()` configure le modèle d'embedding HuggingFace `BAAI/bge-small-en-v1.5` dans `Settings`.
2. `gemini_client.initialize_gemini_llm()` instancie `Gemini(model="models/gemini-2.5-flash")` depuis la variable d'environnement `GEMINI_API_KEY` et l'enregistre dans `Settings`.
3. `text_processor.create_node_parser()` crée un `SentenceSplitter` (chunk_size=512, overlap=50) enregistré dans `Settings.node_parser`.
4. Si `./storage/index_{md5(url)}/` existe → l'index est chargé depuis le disque via `StorageContext.from_defaults`.
5. Sinon → `pdf_loader.load_pdf_from_url()` télécharge le PDF (HTTP GET, timeout 30 s) vers un fichier temporaire, `pdf_loader.load_documents_from_pdf()` le parse avec `PDFReader`, puis `VectorStoreIndex.from_documents()` construit l'index et le persiste.
6. `index.as_query_engine(similarity_top_k=3, response_mode="compact")` est créé.
7. La boucle interactive lit les questions de l'utilisateur, appelle `rag.query()`, et affiche la réponse.

---

## 2. Stack technique

| Couche | Choix | Version | Justification courte |
|---|---|---|---|
| **Langage principal** | Python | 3.12 | Écosystème LLM/ML dominant |
| **Framework RAG** | llama-index-core | (à confirmer) | VectorStoreIndex, Settings, StorageContext |
| **LLM** | llama-index-llms-gemini / Gemini 2.5 Flash | (à confirmer) | Génération de réponses |
| **Embedding** | llama-index-embeddings-huggingface / BAAI/bge-small-en-v1.5 | (à confirmer) | Encodage sémantique des chunks |
| **Parsing PDF** | llama-index-readers-file (PDFReader) | (à confirmer) | Extraction texte depuis PDF |
| **Persistance** | Système de fichiers local (`./storage/`) | — | Cache disque des index vectoriels |
| **Cache** | Cache disque keyed par MD5(URL) | — | Évite de recalculer les embeddings |
| **Messaging** | Aucun actuellement | — | — |
| **Auth** | Aucune — clé API via variable d'env `GEMINI_API_KEY` | — | — |
| **Observabilité** | Aucune actuellement — `print()` uniquement | — | — |
| **CI/CD** | (à confirmer) | — | — |
| **Déploiement** | Local / machine développeur | — | — |

**Dépendances externes critiques** (services tiers dont la panne dégrade le service) :

| Service | Rôle | Mode d'appel | Fallback |
|---|---|---|---|
| Google Gemini API | Génération de réponses LLM | SDK Python (`llama-index-llms-gemini`) | Aucun — la requête échoue |
| Source PDF (URL HTTP) | Fourniture du document à indexer | `requests.get()` | Aucun si pas de cache existant |
| HuggingFace Hub | Téléchargement du modèle d'embedding au premier lancement | SDK interne HuggingFaceEmbedding | Cache local HuggingFace après premier téléchargement |

---

## 3. Pattern d'architecture

### 3.1 Style retenu

Monolithe procédural en modules Python. Pas de découpage en services — toute la logique tourne dans un seul processus Python. Les modules sont découpés par responsabilité fonctionnelle (chargement PDF, traitement texte, client LLM, moteur RAG, entrypoint CLI).

> Décision tracée dans : aucun ADR formalisé actuellement

### 3.2 Diagramme de composants

```mermaid
flowchart LR
    CLI[main.py\nCLI entrypoint] --> Engine[rag_engine.py\nRAGEngine]
    Engine --> PDF[pdf_loader.py\nHTTP download + PDFReader]
    Engine --> Text[text_processor.py\nHuggingFaceEmbedding + SentenceSplitter]
    Engine --> Gemini[gemini_client.py\nGemini 2.5 Flash]
    Engine --> Index[(VectorStoreIndex\nLlamaIndex)]
    Index --> Storage[(./storage/index_{md5}/\ndisque local)]
    PDF --> ExtPDF[(PDF source\nURL HTTP)]
    Gemini --> GeminiAPI[(Google Gemini API)]
```

### 3.3 Propriétés architecturales

| Propriété | Valeur observée | Justification / mécanisme |
|---|---|---|
| **Couplage** | Fort | `RAGEngine` importe directement les 3 modules utilitaires ; pas d'injection de dépendance |
| **Cohésion** | Haute | Chaque module a une responsabilité unique (PDF, embedding, LLM, orchestration) |
| **Concurrence** | Séquentielle | Boucle interactive bloquante, pas de threads ni async |
| **Idempotence** | Garantie sur l'indexation | Le cache MD5(URL) évite de réindexer le même PDF |
| **Cohérence** | Non applicable | Pas de base de données transactionnelle |
| **Scalabilité** | Non applicable | Usage mono-utilisateur local |

---

## 4. Composants

| Composant | Type | Localisation | Rôle | Entrées | Sorties |
|---|---|---|---|---|---|
| `main.py` | Entrypoint CLI | [`main.py`](../main.py) | Boucle interactive Q&A | Saisie utilisateur | Affichage réponse |
| `RAGEngine` | Module / Orchestrateur | [`rag_engine.py`](../rag_engine.py) | Pipeline complet : indexation + requête | URL PDF, question texte | Réponse string |
| `gemini_client` | Module | [`gemini_client.py`](../gemini_client.py) | Singleton LLM Gemini ; initialisation, génération de texte brut, résumé, reranking LLM | Variable d'env `GEMINI_API_KEY` | Instance `Gemini`, `str` |
| `pdf_loader` | Module | [`pdf_loader.py`](../pdf_loader.py) | Télécharge et parse le PDF | URL HTTP | `List[Document]` LlamaIndex |
| `text_processor` | Module | [`text_processor.py`](../text_processor.py) | Configure embedding + node parser | — | `HuggingFaceEmbedding`, `SentenceSplitter` |

### 4.1 Détails par composant

#### RAGEngine

- **Rôle** : Orchestrateur central — initialise tous les composants, gère le cache disque, expose `query()`.
- **Entrées** : URL PDF (string), répertoire de stockage (défaut `./storage`).
- **Sorties** : Réponse string à chaque appel `query()`.
- **Dépendances internes** : `gemini_client`, `pdf_loader`, `text_processor`.
- **Dépendances externes** : LlamaIndex (`VectorStoreIndex`, `StorageContext`, `Settings`), système de fichiers local.
- **Invariants** : Après `__init__`, `self.query_engine` est toujours instancié (depuis cache ou index fraîchement construit).
- **Pièges connus** :
  - ⚠️ L'URL PDF est hardcodée dans `main.py` (`https://arxiv.org/pdf/2005.11401.pdf`) — toute modification nécessite d'éditer le code source.
  - ⚠️ `GEMINI_API_KEY` est lue via `os.environ[...]` (KeyError si absente) — le processus plante au démarrage sans message explicite.

#### gemini_client

- **Rôle** : Singleton LLM — `initialize_gemini_llm()` instancie et enregistre `Gemini` dans `Settings.llm` ; `get_llm()` retourne le singleton (ou l'initialise à la demande) ; `generate_text()` génère du texte brut depuis un prompt ; `summarize()` résume un texte ; `rerank_passages()` utilise le LLM comme zero-shot reranker sur une liste de passages ; `reset_llm()` vide le singleton.
- **Paramètres par défaut** : modèle `models/gemini-2.5-flash`, température `0.1`, max_tokens `1024`.
- **Dépendances externes** : `llama-index-llms-gemini`, variable d'env `GEMINI_API_KEY`.
- **Pièges connus** :
  - ⚠️ `os.environ["GEMINI_API_KEY"]` lève un `KeyError` si la variable est absente — pas de message d'erreur explicite.
  - ⚠️ `rerank_passages()` crée une instance `Gemini` temporaire à température `0.0` ; si le modèle retourne une réponse non parseable, la fonction retourne les passages dans l'ordre original sans erreur.

#### pdf_loader

- **Rôle** : Télécharge un PDF via HTTP et le parse avec `PDFReader` en `List[Document]`.
- **Entrées** : URL HTTP.
- **Sorties** : Fichier temporaire local ; `List[Document]` LlamaIndex.
- **Dépendances externes** : `requests`, `llama-index-readers-file`.
- **Pièges connus** : ⚠️ Timeout HTTP fixé à 30 s — pas de retry en cas d'échec réseau.

#### text_processor

- **Rôle** : Configure `HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")` et `SentenceSplitter(chunk_size=512, chunk_overlap=50)` dans `Settings` LlamaIndex globaux.
- **Entrées** : Aucune (configuration statique).
- **Sorties** : `HuggingFaceEmbedding`, `SentenceSplitter`.
- **Pièges connus** : ⚠️ Le modèle HuggingFace est téléchargé depuis HuggingFace Hub au premier lancement — nécessite un accès internet.

---

## 5. Architecture des données

> Vue résumée. Le détail est dans [data-model.md](data-model.md).

| Domaine | Stockage | Contenu | Volumétrie | Croissance |
|---|---|---|---|---|
| Index vectoriel | Fichiers locaux `./storage/index_{md5(url)}/` | Vecteurs d'embedding + métadonnées LlamaIndex | Dépend de la taille du PDF | Un répertoire par URL distincte |

**Pattern temporel** : Aucun versioning — l'index est écrasé si le répertoire est supprimé manuellement. Le cache est permanent jusqu'à suppression manuelle.

**Politique de rétention** : Aucune politique automatique — rétention indéfinie sur le disque local.

---

## 6. Interfaces

### 6.1 Interfaces exposées

| Interface | Type | Consommateurs | Stabilité |
|---|---|---|---|
| `RAGEngine.query(question: str) -> str` | API Python | `main.py` | Interne / privé |
| CLI interactive (`main.py`) | CLI stdin/stdout | Utilisateur humain | Interne |

### 6.2 Interfaces consommées

| Interface | Fournisseur | Type d'appel | Criticité |
|---|---|---|---|
| Google Gemini API | Google Cloud | Sync HTTP via SDK | Bloquante |
| URL PDF source | Serveur HTTP tiers | `requests.get()` sync | Bloquante (premier lancement uniquement) |
| HuggingFace Hub | HuggingFace | SDK sync | Bloquante (premier lancement uniquement) |

---

## 7. Déploiement

### 7.1 Topologie

```
Machine locale développeur
└── Python process (main.py)
    ├── ./storage/          ← cache index vectoriels
    └── $TMP/               ← fichiers PDF temporaires
```

### 7.2 Environnements

Aucun environnement de déploiement formalisé actuellement — exécution locale uniquement.

### 7.3 Configuration

- **Format** : Variables d'environnement
- **Source de vérité** : Shell local / `.env` (à confirmer)
- **Secrets** : `GEMINI_API_KEY` — lue via `os.environ["GEMINI_API_KEY"]` ; aucun secret manager, jamais committée
- **Feature flags** : Aucun actuellement

---

## 8. Préoccupations transverses

### 8.1 Sécurité

| Aspect | Mécanisme | Localisation |
|---|---|---|
| **Authentification** | Aucune — usage local | — |
| **Autorisation** | Aucune — usage local | — |
| **Secrets** | Variable d'environnement `GEMINI_API_KEY` | [`gemini_client.py`](../gemini_client.py) |
| **Données sensibles** | Aucun chiffrement des index disque | `./storage/` |
| **Audit** | Aucun actuellement | — |

### 8.2 Observabilité

Aucun outil d'observabilité actuellement — traces d'avancement via `print()` uniquement.

### 8.3 Performance

Aucune cible SLO formalisée actuellement. Facteurs dominants : taille du PDF, latence réseau Gemini API, temps d'embedding au premier indexage.

### 8.4 Résilience

- **Timeouts** : HTTP download PDF → 30 s (`requests.get(timeout=30)`) ; appels Gemini API → délégués au SDK
- **Retry** : Aucun actuellement
- **Circuit breaker** : Aucun actuellement
- **Backpressure** : Non applicable
- **Plan de reprise** : Non applicable

---

## 9. Stratégie de test

| Outil | Rôle | Config |
|---|---|---|
| `pytest` | Exécution des tests unitaires | `testpaths = ["tests"]`, fichiers `test_*.py`, fonctions `test_*` — voir `pyproject.toml` |
| `pytest-cov` | Couverture de code | Plugin pytest — invocation à confirmer |

Les tests sont attendus dans `tests/`. Aucune suite de tests existante détectée dans le repo à ce jour — la configuration `pyproject.toml` établit le cadre mais les fichiers de tests sont à créer.

---

## 10. Workflow de développement

- **Branches** : `main`
- **Convention de commits** : gitlint (`gitlint-core>=0.19.1` en dev dep) — règles à confirmer
- **Linting / formatage** : ruff (`ruff>=0.15.14`) — ligne max 100, target Python 3.12, conventions Google docstring
- **Analyse statique** : pyright (`pyright>=1.1.409`) — mode `standard`, Python 3.12, sources `src/` et `tests/`
- **Sécurité statique** : bandit (`bandit>=1.9.4`) — exclut `tests/`, `.venv/`, `build/`
- **CI** : Aucune pipeline CI distante détectée — outils configurés pour exécution locale via `pyproject.toml`
- **Code review** : (à confirmer)
- **Mise à jour de cette doc** : Skill IA doc-patcher exécuté en pre-PR + relecture humaine.

---

## 11. Décisions et points ouverts

### 11.1 Décisions actives

Aucun ADR formalisé actuellement.

### 11.2 Points ouverts

| # | Question | Impact | Owner |
|---|---|---|---|
| Q1 | URL PDF hardcodée dans `main.py` — doit-elle être passée en argument CLI ? | Empêche l'usage sur d'autres documents sans modifier le code | (à confirmer) |
| Q2 | Absence de retry sur le téléchargement PDF et les appels Gemini — comportement en cas d'erreur réseau ? | Le processus plante sans message clair | (à confirmer) |
| Q3 | `GEMINI_API_KEY` via `os.environ[...]` provoque un `KeyError` si absente — ajouter un message d'erreur explicite ? | Mauvaise expérience développeur | (à confirmer) |

### 11.3 Dette technique structurante

| Item | Origine | Coût d'évitement | Plan |
|---|---|---|---|
| URL PDF hardcodée | Prototype initial | Impossible d'indexer un autre document sans éditer le code | Passer en argument CLI ou fichier de config |
| Absence de gestion d'erreur explicite sur `GEMINI_API_KEY` | Prototype initial | `KeyError` non-informatif au démarrage | Vérifier la présence et afficher un message clair |

---

## 12. Références

- Modèle de données : [data-model.md](data-model.md)
- Contrats d'interface : [contracts.md](contracts.md)
- Glossaire : [glossaire.md](glossaire.md)
- Documentation fonctionnelle : [fonctionnel.md](fonctionnel.md)
- Index général : [index.md](index.md)
- ADR : aucun dossier ADR détecté
- Runbooks ops : non applicable
- Dashboards : non applicable

---

<!--
MODE D'EMPLOI DU TEMPLATE
=========================

POUR L'IA QUI MET À JOUR CE DOCUMENT AVANT UNE PR

Déclencheurs (mettre à jour les sections concernées si la PR touche…) :

| Modification dans la PR | Sections à relire |
|---|---|
| Ajout / suppression / renommage d'un service ou module | §3 (diagramme), §4 (composants) |
| Changement de stack (langage, framework, lib majeure) | §2 |
| Nouvelle table / collection ou schéma modifié | §5 (résumé), data-model.md (détail) |
| Nouvelle interface exposée ou consommée | §6, contracts.md (détail) |
| Changement d'environnement, IaC, manifestes K8s | §7 |
| Nouveau mécanisme d'auth, audit, secrets | §8.1 |
| Nouveau collecteur logs / métriques / dashboards | §8.2 |
| SLO, timeout, retry, circuit breaker modifiés | §8.3, §8.4 |
| Nouveau type de test, framework de test | §9 |
| Changement de workflow CI ou règle de branche | §10 |
| Nouvel ADR fusionné | §11.1 |

Règles d'écriture :
1. Mettre à jour le bloc d'en-tête (date, PR de référence, version, mise à jour par).
2. Pour chaque modification, citer la PR ou le commit en référence si non évident.
3. Ne pas réécrire des sections inchangées — diff minimal.
4. Marquer "(à confirmer)" toute affirmation non vérifiable depuis le code à l'instant T.
5. Si une section devient obsolète, la VIDER avec une mention "Aucun {{X}} actuellement"
   plutôt que la supprimer — l'absence est une information.
6. Si la doc et le code divergent, ouvrir un point dans §11.2 et le signaler dans le
   message de PR — ne PAS aligner silencieusement.

Auto-checks à effectuer avant de proposer la mise à jour :
- [ ] Les composants listés en §4 existent réellement dans le repo (vérifier les chemins).
- [ ] Le diagramme Mermaid §3.2 correspond aux appels effectivement présents dans le code.
- [ ] Les versions §2 correspondent au manifeste de dépendances actuel.
- [ ] Les ADR cités §11.1 existent dans le dossier ADR.
- [ ] Les liens des §12 ne sont pas cassés.

POUR LE RELECTEUR HUMAIN

- Le diff doit être lisible : si plus de 30 lignes changent, demander à l'IA de
  scinder par section.
- Vérifier que les chiffres (SLO, volumétrie, version) ne sont pas inventés.
- Toute hypothèse "(à confirmer)" doit être levée ou explicitement assumée.
- Si une section est trop générique (pleine de "{{}}" non remplis), c'est un signe que
  la section n'a jamais été vraiment travaillée — la signaler ou la vider.

POUR ADAPTER À UN AUTRE PROJET

1. Remplacer `{{NOM_PROJET}}` et tous les autres `{{...}}`.
2. Garder les 12 sections et leur ordre — c'est la grille standard.
3. Une section vide est légitime si « non applicable » ; le marquer explicitement.
4. Adapter le diagramme Mermaid §3.2 au style réel (peut être un schéma C4 niveau 2,
   un diagramme de séquence, etc.).
5. Pour un système purement frontal : §5 et §6 peuvent être minces, mais ne pas
   supprimer — pointer vers le backend qui porte ces aspects.
6. Pour un système purement batch : adapter §1.2 ("flux principal" = traitement d'un fichier),
   §6 (interfaces = formats E/S), §8.3 (perf = débit + temps fenêtre).
-->

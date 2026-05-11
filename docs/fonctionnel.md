<!--
TEMPLATE — Documentation fonctionnelle
=======================================
Public cible : (1) une IA qui doit comprendre POURQUOI un comportement existe avant de
le modifier, (2) un humain (PO, dev qui arrive, support) qui veut connaître les règles
métier sans lire le code.

Ce document est la VOIX DU MÉTIER — pas du code. Il décrit ce que le produit fait, pour
qui, dans quelles conditions, avec quelles règles. Le HOW (techno, archi) vit dans
[architecture.md](architecture.md). Le QUOI (signatures) vit dans [contracts.md](contracts.md).

Garde-fous :
- Pas de jargon technique ici. Si un terme technique est nécessaire, soit on l'explique,
  soit on renvoie au glossaire.
- Une règle métier doit être tracée à un cas d'usage et idéalement à un test fonctionnel.
- Les hors-scope (« ce que le produit NE fait PAS ») sont aussi importants que les scopes —
  ils évitent de sur-interpréter.

Bloc « Mode d'emploi » en fin de fichier.
-->

# Documentation fonctionnelle — RAG PDF Q&A

| Champ | Valeur |
|---|---|
| **Dernière mise à jour** | 2026-05-11 |
| **Mise à jour par** | agent IA |
| **PR de référence** | fcd7a06 |
| **Owner produit** | (à confirmer) |
| **Statut** | À retravailler |

> **Pitch en 1 phrase** : Système de questions-réponses interactif en ligne de commande qui permet à un utilisateur de poser des questions en langage naturel sur un document PDF et d'obtenir des réponses générées par un LLM (Gemini) grâce à la recherche sémantique (RAG).

---

## 1. Vision et raison d'être

### 1.1 Problème adressé

Les documents PDF longs (articles de recherche, rapports) contiennent une grande quantité d'informations difficiles à parcourir manuellement. Ce système permet d'interroger directement le contenu d'un PDF en langage naturel, sans avoir à lire l'intégralité du document. Il est conçu pour des utilisateurs qui souhaitent extraire rapidement des réponses précises à partir d'un document académique ou technique.

### 1.2 Bénéfices clés

| Pour | Bénéfice | Mesuré par |
|---|---|---|
| Utilisateur en ligne de commande | Obtenir une réponse précise à une question sans lire le PDF en entier | Pertinence de la réponse (à confirmer) |
| Utilisateur récurrent | Bénéficier d'embeddings mis en cache pour éviter de recalculer l'index à chaque lancement | Temps de démarrage après premier lancement |

### 1.3 Hors-scope

> Ce que le produit ne fait PAS — délibérément. Inclure les fonctionnalités souvent confondues avec ce produit.

| Ce qui est hors-scope | Pourquoi | Alternative |
|---|---|---|
| Interface graphique ou web | L'application est un outil CLI en mode interactif | (à confirmer) |
| Indexation de plusieurs PDFs simultanément | Un seul PDF est passé au démarrage via `pdf_url` | Adapter `RAGEngine.__init__` pour accepter une liste |
| Gestion des utilisateurs / authentification | Outil local mono-utilisateur | (à confirmer) |
| Modification ou annotation du PDF source | Lecture seule | (à confirmer) |

---

## 2. Personas et utilisateurs

| Persona | Rôle | Fréquence d'usage | Objectif principal | Compétences attendues |
|---|---|---|---|---|
| **Chercheur / analyste** | Utilisateur final CLI | Ponctuel / quotidien | Poser des questions sur un article PDF et obtenir des réponses | Savoir lancer un script Python en ligne de commande |
| **Développeur** | Intégrateur / mainteneur | Ponctuel | Tester ou étendre le moteur RAG | Expert Python |

> Pour chaque persona, lister les **cas d'usage prioritaires** dans §3.

---

## 3. Cas d'usage

### 3.1 Carte des cas d'usage

| ID | Cas d'usage | Persona | Fréquence | Criticité |
|---|---|---|---|---|
| UC-01 | Session de questions-réponses interactive sur un PDF | Chercheur / analyste | Quotidien / ponctuel | Critique |
| UC-02 | Rechargement depuis le cache d'embeddings | Chercheur / analyste | À chaque lancement après le premier | Standard |

### 3.2 Détail par cas d'usage

> Un bloc par UC critique. Forme abrégée pour les UC standards.

#### UC-01 — Session de questions-réponses interactive sur un PDF

| Méta | Valeur |
|---|---|
| **Persona** | Chercheur / analyste |
| **Pré-conditions** | L'URL du PDF est accessible ; les dépendances Python sont installées ; la clé API Gemini est configurée dans l'environnement (à confirmer) |
| **Post-conditions** | Succès : une réponse en langage naturel est affichée pour chaque question. Echec : message d'erreur si le PDF est inaccessible ou si le LLM ne répond pas |
| **Déclencheur** | Lancement de `python main.py` en ligne de commande |
| **Fréquence** | Ponctuel / quotidien |
| **Volumétrie** | (à confirmer) |
| **SLA** | (à confirmer) — première question plus lente si l'index doit être construit |
| **Endpoint(s) / écran(s)** | CLI interactif — pas d'endpoint HTTP |
| **Règles métier appliquées** | RG-01, RG-02 (voir §4) |

**Scénario nominal**

1. L'utilisateur lance `python main.py`.
2. Le système charge ou construit l'index vectoriel à partir du PDF (via `RAGEngine.__init__`).
3. Le système affiche l'invite `Your question:`.
4. L'utilisateur saisit une question en langage naturel.
5. Le système effectue une recherche sémantique sur l'index (top-3 passages), puis génère une réponse via Gemini.
6. La réponse est affichée sous la forme `Answer: <texte>`.
7. Les étapes 3 à 6 se répètent jusqu'à ce que l'utilisateur tape `q`.

**Scénarios alternatifs**

| Variante | Condition | Comportement |
|---|---|---|
| Index déjà en cache | Le répertoire `./storage/index_<md5>` existe | L'index est chargé depuis le disque sans re-télécharger ni re-calculer les embeddings (UC-02) |

**Cas d'erreur métier**

| Cas | Détection | Message utilisateur | Effet |
|---|---|---|---|
| PDF inaccessible à l'URL | Lors du téléchargement dans `load_pdf_from_url` | Exception Python non capturée (à confirmer) | Arrêt du programme |
| Réponse vide du LLM | Lors de `query_engine.query` | (à confirmer) | (à confirmer) |

**Pièges connus**

- ⚠️ La première exécution sur un nouveau PDF peut être longue : téléchargement, extraction, calcul des embeddings et écriture du cache dans `./storage/`.
- ⚠️ Taper `q` (minuscule uniquement) est le seul moyen de quitter la boucle interactive ; `Q`, `quit`, `exit` ne sont pas reconnus.

#### UC-02 — Rechargement depuis le cache d'embeddings (forme abrégée)

Lors de tout lancement ultérieur avec le même `pdf_url`, `RAGEngine.__init__` détecte le répertoire `./storage/index_<md5(pdf_url)>` et charge l'index persisté via `StorageContext.from_defaults`, évitant ainsi le téléchargement et le calcul des embeddings. Règles appliquées : RG-02.

---

## 4. Règles de gestion

> Toute règle métier nommée. Une RG = une assertion sur le système, formulée en langage métier, idéalement testable.

### 4.1 Catalogue

| ID | Nom | Domaine | Cas d'usage liés | Statut |
|---|---|---|---|---|
| RG-01 | Quitter par `q` uniquement | Interaction CLI | UC-01 | Active |
| RG-02 | Cache d'index par empreinte MD5 de l'URL | Persistance / performance | UC-01, UC-02 | Active |
| RG-03 | Top-3 passages pour la génération de réponse | Recherche sémantique | UC-01 | Active |

### 4.2 Détail par règle

#### RG-01 — Quitter par `q` uniquement

- **Énoncé** : La session interactive se termine uniquement lorsque l'utilisateur saisit la chaîne `q` (casse insensible via `.lower()`).
- **Pré-conditions** : La boucle interactive est active.
- **Effet** : Tout autre saisie (y compris `Q`, `quit`, `exit`) est traitée comme une question et envoyée au moteur RAG.
- **Origine** : Décision d'implémentation (`main.py`).
- **Implémentation** : [main.py:15](../main.py#L15)
- **Tests fonctionnels** : (à confirmer)
- **Évolutions** :
  | Date | PR | Changement |
  |---|---|---|
  | 2026-05-11 | fcd7a06 | Création |

#### RG-02 — Cache d'index par empreinte MD5 de l'URL

- **Énoncé** : Le moteur RAG calcule l'empreinte MD5 de `pdf_url` pour déterminer le chemin de cache `./storage/index_<md5>`. Si ce répertoire existe, l'index est chargé depuis le cache ; sinon il est construit et persisté.
- **Pré-conditions** : `RAGEngine.__init__` est appelé avec un `pdf_url`.
- **Effet** : Évite de re-télécharger le PDF et de re-calculer les embeddings à chaque lancement. Deux URLs distinctes produisent deux caches distincts.
- **Origine** : Décision d'implémentation (`rag_engine.py`).
- **Implémentation** : [rag_engine.py:63-65](../rag_engine.py#L63)
- **Tests fonctionnels** : (à confirmer)
- **Évolutions** :
  | Date | PR | Changement |
  |---|---|---|
  | 2026-05-11 | fcd7a06 | Création |

#### RG-03 — Top-3 passages pour la génération de réponse

- **Énoncé** : Le moteur de requête est configuré avec `similarity_top_k=3` et `response_mode="compact"` : les 3 passages les plus proches sémantiquement de la question sont transmis au LLM pour générer la réponse.
- **Pré-conditions** : L'index vectoriel est chargé.
- **Effet** : Limite le contexte fourni au LLM à 3 extraits ; une question sans passage pertinent peut produire une réponse peu précise.
- **Origine** : Décision d'implémentation (`rag_engine.py`).
- **Implémentation** : [rag_engine.py:59](../rag_engine.py#L59)
- **Tests fonctionnels** : (à confirmer)
- **Évolutions** :
  | Date | PR | Changement |
  |---|---|---|
  | 2026-05-11 | fcd7a06 | Création |

---

## 5. Workflows et machines à états

> Ce système est un outil CLI sans entités persistantes à cycle de vie (pas de commande, dossier, contrat). Cette section n'est pas applicable dans l'état actuel du produit.

---

## 6. Données métier de référence

> Référentiels et énumérations qui ont une signification métier. Le détail technique vit dans [data-model.md](data-model.md).

| Référentiel | Valeurs | Source de vérité | Mise à jour |
|---|---|---|---|
| **Commande de sortie** | `q` (insensible à la casse) | `main.py` ligne 15 | Manuel |
| **Mode de réponse LLM** | `compact` | `rag_engine.py` ligne 59 | Manuel |

---

## 7. Cas limites et pathologiques

> Comportements aux extrémités du fonctionnel. Ces cas sont la principale source de bugs en production — les documenter explicitement.

| Cas | Comportement attendu | Justification |
|---|---|---|
| **PDF très long (centaines de pages)** | L'indexation peut durer plusieurs minutes ; le cache évite de répéter ce coût | Indexation par `VectorStoreIndex.from_documents` sans limite de pages |
| **PDF inaccessible ou URL invalide** | Erreur Python non gérée explicitement — arrêt du programme (à confirmer) | Aucune gestion d'erreur réseau visible dans `main.py` / `rag_engine.py` |
| **Question hors sujet du PDF** | Le LLM peut répondre à partir des 3 passages les plus proches, même si la pertinence est faible | `similarity_top_k=3` ; pas de seuil de score minimal visible |
| **Question vide** | (à confirmer) — la chaîne vide est envoyée au moteur RAG sans validation préalable | Aucun garde-fou visible dans la boucle `main()` |
| **Cache corrompu** | `load_index_from_storage` lève une exception ; le programme s'arrête (à confirmer) | Aucune logique de fallback visible |
| **Même URL, PDF mis à jour à la source** | L'index en cache est réutilisé sans vérification du contenu — la mise à jour n'est pas détectée | Le cache est identifié par MD5 de l'URL, pas du contenu |

---

## 8. Conformité et contraintes externes

| Contrainte | Origine | Implémentation |
|---|---|---|
| Licence du PDF indexé | Droit d'auteur du document source | L'URL par défaut pointe vers arXiv (accès ouvert) ; tout autre PDF doit être vérifié par l'utilisateur |
| Conditions d'utilisation de l'API Gemini | Google | Clé API requise ; soumise aux CGU Google AI (à confirmer) |
| Licence des modèles d'embedding (sentence-transformers) | Apache 2.0 (à confirmer) | `requirements.txt` — `sentence-transformers`, `llama-index-embeddings-huggingface` |

---

## 9. KPIs et indicateurs métier

| KPI | Définition | Cible | Source | Dashboard |
|---|---|---|---|---|
| Temps de démarrage (premier lancement) | Durée entre `python main.py` et l'affichage de l'invite | (à confirmer) | Logs console | (à confirmer) |
| Temps de démarrage (cache chaud) | Durée entre `python main.py` et l'affichage de l'invite quand l'index est en cache | (à confirmer) | Logs console | (à confirmer) |
| Temps de réponse par question | Durée entre la saisie et l'affichage de `Answer:` | (à confirmer) | (à confirmer) | (à confirmer) |

> Note : ces KPIs orientent les décisions produit. Toute fonctionnalité majeure devrait s'y rattacher.

---

## 10. Roadmap et évolutions notables

> Historique compressé : grands jalons fonctionnels passés et à venir. Pas un changelog détaillé (qui vit dans `CHANGELOG.md` ou les releases).

### 10.1 Jalons passés

| Date | Version | Évolution majeure | PR / RFC |
|---|---|---|---|
| 2026-05-11 | fcd7a06 | Mise en place initiale : `RAGEngine` avec cache d'index, session interactive CLI, intégration Gemini + sentence-transformers | 03de3db |

### 10.2 À venir (haut niveau)

| Échéance | Évolution | Statut |
|---|---|---|
| (à confirmer) | (à confirmer) | (à confirmer) |

---

## 11. Questions ouvertes côté métier

> Décisions fonctionnelles non tranchées. Distinct des questions techniques (qui vivent dans le glossaire ou les ADR).

| # | Question | Impact si non tranché | Owner | Ouverte depuis |
|---|---|---|---|---|
| QF-01 | Quel modèle d'embedding est effectivement utilisé par `setup_advanced_text_processing` ? | Impacts sur la qualité des réponses et les contraintes de licence | (à confirmer) | 2026-05-11 |
| QF-02 | Comment configurer la clé API Gemini (variable d'environnement, fichier `.env` ?) | Bloque tout utilisateur qui installe le projet | (à confirmer) | 2026-05-11 |
| QF-03 | Comportement attendu lorsque l'utilisateur pose une question vide ou hors sujet ? | Qualité d'expérience utilisateur | (à confirmer) | 2026-05-11 |

---

## Références

- Vue d'ensemble système : [architecture.md](architecture.md)
- Endpoints / interfaces qui matérialisent les UC : [contracts.md](contracts.md)
- Données qui supportent les règles : [data-model.md](data-model.md)
- Vocabulaire métier détaillé : [glossaire.md](glossaire.md)
- Décisions produit historiques : (à confirmer)
- Backlog : (à confirmer)

---

<!--
MODE D'EMPLOI DU TEMPLATE
=========================

POUR L'IA QUI MET À JOUR CE FICHIER

⚠️ La doc fonctionnelle est la PLUS DIFFICILE à maintenir par une IA seule — elle décrit
l'INTENTION, qui n'est pas dans le code. L'IA doit ÊTRE PRUDENTE et ne pas inventer du
sens. Si une PR introduit un comportement nouveau dont l'intention n'est pas claire,
laisser un placeholder et SIGNALER dans la PR plutôt que deviner.

Déclencheurs :

| Modification dans la PR | Sections à relire |
|---|---|
| Nouveau parcours utilisateur / endpoint exposé à un humain | §3 (carte + détail) |
| Nouveau persona ou rôle | §2 |
| Nouvelle règle de validation / contrainte métier | §4 |
| Modification d'un état possible / d'une transition | §5 |
| Nouvelle énumération / référentiel métier | §6 |
| Nouveau cas pathologique géré (ou bug fix significatif) | §7 |
| Conformité (RGPD, accessibilité, audit) | §8 |
| Nouveau KPI suivi | §9 |
| Lancement / dépréciation d'une feature majeure | §10.1 |
| Décision produit en attente | §11 |

Règles spéciales :
- Toute RG-XX nouvelle DOIT pointer vers une implémentation et un test (sinon, signalement
  en PR — c'est une RG « orpheline »).
- Si un UC change de comportement nominal, ajouter une ligne dans son sous-bloc « Évolutions »
  (à créer au besoin), ne pas réécrire silencieusement le scénario.
- Les hors-scope §1.3 vieillissent : si une fonctionnalité hors-scope devient scope, la
  retirer de §1.3 ET la documenter dans §3.

Auto-checks :
- [ ] Chaque RG citée dans un UC §3 existe dans §4.
- [ ] Chaque RG §4 cite une implémentation et un test.
- [ ] Chaque transition §5 a une ligne de tableau associée.
- [ ] Le pitch §0 est à jour avec la dernière évolution majeure §10.1.
- [ ] Aucune QF-XX § 11 ouverte depuis > 90 jours sans note de réunion.

POUR LE RELECTEUR HUMAIN (idéalement le PO ou un expert métier)

- Le doc doit pouvoir être lu par un nouvel arrivant non technique en 30 minutes.
- Vérifier que les RG ne sont pas « techniques déguisées » (ex. « le timeout est de 5s »
  est une contrainte non-fonctionnelle, pas une RG).
- Les KPIs §9 doivent provenir d'un dashboard existant, pas être souhaitables.
- Les hors-scope §1.3 méritent un challenge périodique.

POUR ADAPTER À UN AUTRE PROJET

1. C'est le template le plus PROJET-DÉPENDANT. Pour un produit B2C avec écrans : §3 est
   structuré par parcours utilisateur. Pour un service technique B2B : §3 est structuré par
   intégration partenaire.
2. Si le système est purement technique sans utilisateur final (ex. lib, infra) :
   - §2 = systèmes consommateurs.
   - §3 = scénarios d'intégration.
   - §5 et §9 peuvent disparaître.
3. Pour un système legacy en migration : ajouter une section « Comportements à reproduire
   à l'identique » et « Comportements connus non reproduits (et pourquoi) ».
4. Si le projet a une **forte dimension réglementaire** (banque, santé, public) : §8 prend
   beaucoup de place, mériter sa propre subdivision.
-->

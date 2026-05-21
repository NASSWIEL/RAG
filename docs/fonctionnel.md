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

# Documentation fonctionnelle — RAG Q&A

| Champ | Valeur |
|---|---|
| **Dernière mise à jour** | 2026-05-21 |
| **Mise à jour par** | agent IA |
| **PR de référence** | (à confirmer) |
| **Owner produit** | (à confirmer) |
| **Statut** | À retravailler |

> **Pitch en 1 phrase** : Application de questions-réponses interactive qui permet à un utilisateur d'interroger en langage naturel le contenu d'un article scientifique (le papier RAG), sans lire le PDF.

---

## 1. Vision et raison d'être

### 1.1 Problème adressé

Lire et mémoriser un article scientifique long pour en extraire des réponses précises est coûteux en temps. Cette application permet d'interroger directement le contenu du papier « Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks » (Lewis et al., 2020) via des questions en langage naturel, en s'appuyant sur la recherche sémantique et un LLM pour synthétiser les réponses.

### 1.2 Bénéfices clés

| Pour | Bénéfice | Mesuré par |
|---|---|---|
| **Chercheur / étudiant** | Obtenir une réponse ciblée sans lire l'intégralité du PDF | (à confirmer) |
| **Développeur** | Comprendre rapidement l'architecture RAG décrite dans le papier | (à confirmer) |

### 1.3 Hors-scope

> Ce que le produit ne fait PAS — délibérément. Inclure les fonctionnalités souvent confondues avec ce produit.

| Ce qui est hors-scope | Pourquoi | Alternative |
|---|---|---|
| Interroger plusieurs PDF simultanément | Le PDF source est codé en dur dans `main.py` | Modifier `pdf_url` manuellement |
| Interface graphique / web | Application en ligne de commande uniquement | (à confirmer) |
| Mise à jour du PDF source à la volée | L'URL est une constante hardcodée | Modifier le code source |
| Persistance des conversations | Chaque question est indépendante, pas de mémoire de session | (à confirmer) |

---

## 2. Personas et utilisateurs

| Persona | Rôle | Fréquence d'usage | Objectif principal | Compétences attendues |
|---|---|---|---|---|
| **Utilisateur CLI** | Toute personne souhaitant interroger le papier RAG | Ponctuel | Obtenir des réponses à ses questions sur le contenu du PDF | Savoir lancer `python main.py` dans un terminal |

> Pour chaque persona, lister les **cas d'usage prioritaires** dans §3.

---

## 3. Cas d'usage

### 3.1 Carte des cas d'usage

| ID | Cas d'usage | Persona | Fréquence | Criticité |
|---|---|---|---|---|
| UC-01 | Poser une question sur le PDF | Utilisateur CLI | Ponctuel | Critique |
| UC-02 | Lancement initial (génération des embeddings) | Utilisateur CLI | Une seule fois par PDF | Standard |
| UC-03 | Quitter l'application | Utilisateur CLI | Ponctuel | Standard |

### 3.2 Détail par cas d'usage

> Un bloc par UC critique. Forme abrégée pour les UC standards.

#### UC-01 — Poser une question sur le PDF

| Méta | Valeur |
|---|---|
| **Persona** | Utilisateur CLI |
| **Pré-conditions** | L'application est lancée (`python main.py`) ; la variable d'environnement `GEMINI_API_KEY` est définie |
| **Post-conditions** | Une réponse synthétisée est affichée dans le terminal |
| **Déclencheur** | L'utilisateur saisit une question et appuie sur Entrée |
| **Fréquence** | Ponctuel |
| **Volumétrie** | (à confirmer) |
| **SLA** | (à confirmer) |
| **Endpoint(s) / écran(s)** | Interface terminal — prompt `Your question:` |
| **Règles métier appliquées** | RG-01, RG-02, RG-03 (voir §4) |

**Scénario nominal**

1. L'utilisateur tape une question au prompt `Your question:`.
2. Le système effectue une recherche sémantique dans l'index vectoriel et récupère les 3 passages les plus pertinents (RG-01).
3. Le LLM génère une réponse compacte à partir de ces passages (RG-02).
4. La réponse est affichée sous le libellé `Answer:`.

**Scénarios alternatifs**

| Variante | Condition | Comportement |
|---|---|---|
| Chargement depuis le cache | Embeddings déjà générés pour cette URL (RG-03) | Le chargement de l'index est quasi-instantané au démarrage |
| Premier lancement | Aucun cache existant pour cette URL | Téléchargement du PDF, génération et sauvegarde des embeddings dans `./storage/` (voir UC-02) |

**Cas d'erreur métier**

| Cas | Détection | Message utilisateur | Effet |
|---|---|---|---|
| `GEMINI_API_KEY` absente | Au démarrage, lors de l'init du LLM | `KeyError: 'GEMINI_API_KEY'` (erreur Python non interceptée) | L'application ne démarre pas |

**Pièges connus**

- ⚠️ Le PDF est re-téléchargé à chaque lancement même si le cache d'embeddings existe déjà — seule la génération des embeddings est évitée, pas le téléchargement.

---

#### UC-02 — Lancement initial (génération des embeddings)

| Méta | Valeur |
|---|---|
| **Persona** | Utilisateur CLI |
| **Pré-conditions** | Aucun cache dans `./storage/` pour l'URL courante ; `GEMINI_API_KEY` définie |
| **Post-conditions** | Les embeddings sont persistés dans `./storage/index_<hash_md5_url>/` |
| **Déclencheur** | `python main.py` sans cache existant |
| **Règles métier appliquées** | RG-03, RG-04 |

**Scénario nominal**

1. Téléchargement du PDF depuis `https://arxiv.org/pdf/2005.11401.pdf`.
2. Extraction du texte et découpage en chunks.
3. Génération des vecteurs d'embedding.
4. Sauvegarde dans `./storage/index_<hash>`.
5. Démarrage du mode interactif.

---

#### UC-03 — Quitter l'application

Forme abrégée. L'utilisateur tape `q` au prompt ; l'application se termine proprement.

---

## 4. Règles de gestion

> Toute règle métier nommée. Une RG = une assertion sur le système, formulée en langage métier, idéalement testable.

### 4.1 Catalogue

| ID | Nom | Domaine | Cas d'usage liés | Statut |
|---|---|---|---|---|
| RG-01 | Top-3 passages | Recherche | UC-01 | Active |
| RG-02 | Réponse compacte | Génération | UC-01 | Active |
| RG-03 | Cache par empreinte d'URL | Performance | UC-01, UC-02 | Active |
| RG-04 | Clé API obligatoire | Sécurité | UC-01, UC-02 | Active |

### 4.2 Détail par règle

#### RG-01 — Top-3 passages

- **Énoncé** : Pour chaque question, le système récupère les 3 passages du PDF les plus similaires sémantiquement à la question.
- **Pré-conditions** : L'index vectoriel est chargé.
- **Effet** : Seuls 3 chunks sont transmis au LLM pour générer la réponse.
- **Origine** : Paramètre `similarity_top_k=3` — décision de conception.
- **Implémentation** : [rag_engine.py:55](rag_engine.py#L55)
- **Tests fonctionnels** : (à confirmer)

#### RG-02 — Réponse compacte

- **Énoncé** : Le LLM synthétise les passages récupérés en une réponse condensée (mode `compact`).
- **Pré-conditions** : RG-01 exécutée.
- **Effet** : La réponse agrège les informations sans répéter chaque passage verbatim.
- **Origine** : Paramètre `response_mode="compact"` — décision de conception.
- **Implémentation** : [rag_engine.py:55](rag_engine.py#L55)
- **Tests fonctionnels** : (à confirmer)

#### RG-03 — Cache par empreinte d'URL

- **Énoncé** : Les embeddings générés pour un PDF sont conservés sur disque et réutilisés à chaque lancement suivant, identifiés par l'empreinte MD5 de l'URL du PDF.
- **Pré-conditions** : Le répertoire `./storage/` est accessible en écriture.
- **Effet** : Les lancements suivants n'effectuent pas de nouveau calcul d'embeddings (gain de temps significatif).
- **Origine** : Décision de conception pour éviter la re-génération coûteuse.
- **Implémentation** : [rag_engine.py:59-61](rag_engine.py#L59)
- **Tests fonctionnels** : (à confirmer)

#### RG-04 — Clé API obligatoire

- **Énoncé** : L'application exige que la variable d'environnement `GEMINI_API_KEY` soit définie avant le démarrage.
- **Pré-conditions** : Lancement de l'application.
- **Effet** : Sans cette clé, l'initialisation du LLM échoue et l'application ne démarre pas.
- **Origine** : Contrainte du fournisseur LLM (Google Gemini).
- **Implémentation** : [gemini_client.py:12](gemini_client.py#L12)
- **Tests fonctionnels** : (à confirmer)

---

## 5. Workflows et machines à états

> Pas applicable — l'application n'a pas d'entités avec un cycle de vie propre. Le seul flux notable est le démarrage (avec ou sans cache), décrit dans §3.

---

## 6. Données métier de référence

> Référentiels et énumérations qui ont une signification métier. Le détail technique vit dans [data-model.md](data-model.md).

| Référentiel | Valeurs | Source de vérité | Mise à jour |
|---|---|---|---|
| **PDF source** | `https://arxiv.org/pdf/2005.11401.pdf` (Lewis et al., 2020 — papier RAG original) | `main.py` (constante hardcodée) | Manuelle (modification du code) |

---

## 7. Cas limites et pathologiques

> Comportements aux extrémités du fonctionnel. Ces cas sont la principale source de bugs en production — les documenter explicitement.

| Cas | Comportement attendu | Justification |
|---|---|---|
| **Cache corrompu** | (à confirmer) | Si `./storage/index_<hash>/` existe mais est incomplet, LlamaIndex peut lever une exception au chargement |
| **URL PDF inaccessible** | (à confirmer) | Le téléchargement échoue ; aucun fallback prévu dans le code |
| **Question vide** | L'utilisateur appuie sur Entrée sans texte — la question est transmise telle quelle au moteur | Aucune validation côté application |
| **Question `q` en majuscule** | `Q` ne quitte pas l'application — seul `q` minuscule déclenche la sortie | `main.py:15` : `if user_query.lower() in ['q']` — en fait `lower()` est appelé donc `Q` quitte aussi ⚠️ |
| **Reprise après panne mi-indexation** | Le répertoire partiel reste sur disque ; au prochain lancement, LlamaIndex tente de le charger | (à confirmer) |

---

## 8. Conformité et contraintes externes

| Contrainte | Origine | Implémentation |
|---|---|---|
| Licence du PDF source | Article publié sur arXiv — usage académique/recherche | URL publique, pas de données personnelles traitées |
| Conditions d'usage Google Gemini API | Contrat fournisseur | Clé API privée via variable d'environnement `GEMINI_API_KEY` |

---

## 9. KPIs et indicateurs métier

> Pas de KPIs définis à ce stade — application de démonstration/recherche. (à confirmer)

---

## 10. Roadmap et évolutions notables

> Historique compressé : grands jalons fonctionnels passés et à venir. Pas un changelog détaillé (qui vit dans `CHANGELOG.md` ou les releases).

### 10.1 Jalons passés

| Date | Version | Évolution majeure | PR / RFC |
|---|---|---|---|
| (à confirmer) | v1.0 | Mise en place initiale : Q&A interactif sur le papier RAG avec cache d'embeddings | — |

### 10.2 À venir (haut niveau)

| Échéance | Évolution | Statut |
|---|---|---|
| (à confirmer) | (à confirmer) | (à confirmer) |

---

## 11. Questions ouvertes côté métier

> Décisions fonctionnelles non tranchées. Distinct des questions techniques (qui vivent dans le glossaire ou les ADR).

| # | Question | Impact si non tranché | Owner | Ouverte depuis |
|---|---|---|---|---|
| QF-01 | Le PDF source doit-il rester celui du papier RAG original, ou l'application doit-elle supporter n'importe quel PDF ? | Définit si l'app est un outil générique ou un démonstrateur figé | (à confirmer) | 2026-05-21 |
| QF-02 | Faut-il gérer la mémoire de session (questions précédentes) pour des échanges multi-tours ? | Impacte fortement l'architecture du moteur de requêtes | (à confirmer) | 2026-05-21 |

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

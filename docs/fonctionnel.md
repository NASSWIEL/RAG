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

# Documentation fonctionnelle — RAG

| Champ | Valeur |
|---|---|
| **Dernière mise à jour** | 2026-05-20 |
| **Mise à jour par** | agent IA |
| **PR de référence** | (à confirmer) |
| **Owner produit** | (à confirmer) |
| **Statut** | Brouillon |

> **Pitch en 1 phrase** : Moteur de questions-réponses interactif sur des documents PDF — télécharge un PDF depuis une URL, en indexe le contenu via des embeddings sémantiques, puis répond aux questions en langage naturel grâce au LLM Gemini 2.5 Flash.

---

## 1. Vision et raison d'être

### 1.1 Problème adressé

Lire et interroger un long document PDF (article de recherche, rapport) prend du temps. Ce système permet à un utilisateur de poser des questions en langage naturel sur le contenu d'un PDF sans avoir à le parcourir manuellement. Il télécharge le document, le découpe en fragments sémantiques, génère des embeddings et les met en cache ; les requêtes suivantes sont instantanées. La réponse est générée par Gemini en s'appuyant uniquement sur les passages du document les plus pertinents.

### 1.2 Bénéfices clés

| Pour | Bénéfice | Mesuré par |
|---|---|---|
| **Utilisateur CLI** | Obtenir une réponse précise sur un PDF sans lire l'intégralité du document | (à confirmer) |
| **Développeur / chercheur** | Réutilisation du cache d'embeddings entre sessions — pas de re-traitement coûteux | Présence du dossier `./storage/index_<md5>` après la première exécution |

### 1.3 Hors-scope

> Ce que le produit ne fait PAS — délibérément. Inclure les fonctionnalités souvent confondues avec ce produit.

| Ce qui est hors-scope | Pourquoi | Alternative |
|---|---|---|
| Interrogation de plusieurs PDFs simultanément | Un seul PDF est configuré dans `main.py` (à confirmer) | Modifier `pdf_url` dans `main.py` pour changer de document |
| Interface graphique / web | Produit CLI uniquement | (à confirmer) |
| Persistance de l'historique des conversations | Chaque question est indépendante ; pas de mémoire de session | (à confirmer) |
| Indexation de PDFs locaux sans URL | Le chargement se fait via URL HTTP | Adapter `pdf_loader.py` (à confirmer) |

---

## 2. Personas et utilisateurs

| Persona | Rôle | Fréquence d'usage | Objectif principal | Compétences attendues |
|---|---|---|---|---|
| **Utilisateur CLI** | Utilisateur final (chercheur, développeur, lecteur) | Ponctuel | Poser des questions sur un document PDF et obtenir des réponses contextuelles | Savoir lancer un script Python en ligne de commande |
| **Développeur / intégrateur** | Appelant programmatique de la bibliothèque | (à confirmer) | Utiliser les capacités de génération, résumé et réordonnancement de passages sans passer par le mode interactif | Connaissance de Python ; capacité à importer et appeler `gemini_client` |

> Pour chaque persona, lister les **cas d'usage prioritaires** dans §3.

---

## 3. Cas d'usage

### 3.1 Carte des cas d'usage

| ID | Cas d'usage | Persona | Fréquence | Criticité |
|---|---|---|---|---|
| UC-01 | Poser une question sur un PDF en mode interactif | Utilisateur CLI | Ponctuel | Critique |
| UC-02 | Relancer la session en réutilisant le cache d'embeddings | Utilisateur CLI | Ponctuel | Standard |
| UC-03 | Utiliser les capacités de génération, résumé et réordonnancement par appel programmatique | Développeur / intégrateur | (à confirmer) | Standard |

### 3.2 Détail par cas d'usage

> Un bloc par UC critique. Forme abrégée pour les UC standards.

#### UC-01 — Poser une question sur un PDF en mode interactif

| Méta | Valeur |
|---|---|
| **Persona** | Utilisateur CLI |
| **Pré-conditions** | `GOOGLE_API_KEY` configuré dans `config.py` ; accès réseau à l'URL du PDF ; dépendances Python installées |
| **Post-conditions** | Succès : une réponse en langage naturel est affichée dans le terminal. Échec : message d'erreur Python |
| **Déclencheur** | Lancement de `python main.py` puis saisie d'une question |
| **Fréquence** | Ponctuelle |
| **Volumétrie** | (à confirmer) |
| **SLA** | (à confirmer) — dépend de la latence Gemini API et de la taille du PDF |
| **Endpoint(s) / écran(s)** | `RAGEngine.query()` dans [`rag_engine.py`](../rag_engine.py) |
| **Règles métier appliquées** | Aucune RG métier formalisée à ce stade (à confirmer) |

**Scénario nominal**

1. L'utilisateur lance `python main.py`.
2. Au premier lancement, le PDF (`https://arxiv.org/pdf/2005.11401.pdf`) est téléchargé, découpé en fragments (chunk size 512, overlap 50), les embeddings `BAAI/bge-small-en-v1.5` sont générés et sauvegardés dans `./storage/`.
3. Le moteur affiche `Interactive mode - Enter your questions (type 'q' to exit):`.
4. L'utilisateur saisit une question.
5. Les 3 fragments les plus pertinents sont récupérés par recherche sémantique (`similarity_top_k=3`).
6. Gemini 2.5 Flash génère une réponse à partir de ces fragments (`response_mode="compact"`).
7. La réponse est affichée dans le terminal.
8. La boucle reprend à l'étape 4 jusqu'à ce que l'utilisateur saisisse `q`.

**Scénarios alternatifs**

| Variante | Condition | Comportement |
|---|---|---|
| Cache existant | Le dossier `./storage/index_<md5(url)>` existe déjà | Le téléchargement et la génération d'embeddings sont ignorés ; l'index est chargé depuis le disque (étape 2 remplacée par un chargement) |

**Cas d'erreur métier**

| Cas | Détection | Message utilisateur | Effet |
|---|---|---|---|
| PDF inaccessible (URL invalide ou réseau indisponible) | Lors du téléchargement à l'étape 2 | Exception Python dans le terminal | Arrêt du programme |
| Clé API Gemini absente ou invalide | Lors de l'initialisation du LLM | Exception Python dans le terminal | Arrêt du programme |

**Pièges connus**

- ⚠️ La saisie de `q` est insensible à la casse (`user_query.lower() in ["q"]`) mais `quit` ou `exit` ne terminent pas la session — ils sont envoyés comme questions au moteur.
- ⚠️ Le PDF source est codé en dur dans `main.py:5` (`https://arxiv.org/pdf/2005.11401.pdf`). Changer de document nécessite une modification du code source.

#### UC-02 — Relancer la session en réutilisant le cache

Le cache est identifié par le hash MD5 de l'URL du PDF. Si `./storage/index_<md5>` existe, les embeddings sont chargés sans re-traitement. Le comportement de requête est identique à UC-01 à partir de l'étape 3.

#### UC-03 — Utiliser les capacités de génération, résumé et réordonnancement par appel programmatique

Un développeur ou un intégrateur importe `gemini_client` directement et exploite trois capacités sans passer par le mode interactif CLI :

- **Génération directe** : soumettre un prompt libre à Gemini et obtenir une réponse textuelle, sans passer par la chaîne de récupération RAG.
- **Résumé** : fournir un texte et obtenir un résumé condensé dont la longueur cible est paramétrable.
- **Réordonnancement de passages** : fournir une question et une liste de passages récupérés ; Gemini les réordonne du plus au moins pertinent. En cas d'échec d'analyse de la réponse, l'ordre original est conservé.

| Méta | Valeur |
|---|---|
| **Persona** | Développeur / intégrateur |
| **Pré-conditions** | `GOOGLE_API_KEY` configuré ; dépendances Python installées ; `gemini_client` importé |
| **Post-conditions** | Succès : valeur textuelle retournée. Échec : exception Python (prompt vide → `ValueError`) |
| **Déclencheur** | Appel direct à l'une des fonctions exposées par `gemini_client` |
| **Endpoint(s)** | `generate_text()`, `summarize()`, `rerank_passages()` dans [`gemini_client.py`](../gemini_client.py) |
| **Règles métier appliquées** | Un prompt ou texte vide provoque une erreur explicite ; le réordonnancement dégrade silencieusement vers l'ordre original si Gemini retourne un résultat non parsable |

---

## 4. Règles de gestion

> Aucune règle de gestion métier formalisée à ce stade. Les paramètres techniques (`similarity_top_k=3`, `chunk_size=512`, `response_mode="compact"`) sont des choix d'implémentation, pas des RG. (à confirmer)

---

## 5. Workflows et machines à états

Non applicable — le système ne comporte pas d'entité avec cycle de vie dans la version actuelle.

---

## 6. Données métier de référence

> Référentiels et énumérations qui ont une signification métier. Le détail technique vit dans [data-model.md](data-model.md).

| Référentiel | Valeurs | Source de vérité | Mise à jour |
|---|---|---|---|
| **PDF source par défaut** | `https://arxiv.org/pdf/2005.11401.pdf` (article RAG de Lewis et al., 2020) | `main.py:5` | Manuel — modifier `pdf_url` dans `main.py` |

---

## 7. Cas limites et pathologiques

> Comportements aux extrémités du fonctionnel. Ces cas sont la principale source de bugs en production — les documenter explicitement.

| Cas | Comportement attendu | Justification |
|---|---|---|
| **PDF inaccessible** | Exception Python, arrêt du programme | Aucune gestion d'erreur réseau dans `pdf_loader.py` (à confirmer) |
| **Cache corrompu** | Erreur au chargement de l'index ; comportement non défini | LlamaIndex `load_index_from_storage` peut lever une exception (à confirmer) |
| **Question vide** | La question vide est envoyée à Gemini comme n'importe quelle autre saisie | Pas de validation de la saisie dans `main.py` |
| **Saisie `q` en majuscule ou `Q`** | Termine la session (`lower()` appliqué) | Voir `main.py:12` |
| **Saisie `quit` ou `exit`** | Envoyée comme question au moteur RAG — ne termine pas la session | Seul `q` est géré comme sortie |
| **Reprise après panne** | Relancer `python main.py` — si le cache est intact, l'index est rechargé (à confirmer) | |

---

## 8. Conformité et contraintes externes

| Contrainte | Origine | Implémentation |
|---|---|---|
| **Conditions d'utilisation Gemini API** | Google (à confirmer) | Clé API dans `config.py` |
| **Licence du PDF interrogé** | Dépend du document source | Responsabilité de l'utilisateur (à confirmer) |

---

## 9. KPIs et indicateurs métier

Non applicable à ce stade — outil CLI sans tableau de bord ni métriques instrumentées. (à confirmer)

---

## 10. Roadmap et évolutions notables

### 10.1 Jalons passés

| Date | Version | Évolution majeure | PR / RFC |
|---|---|---|---|
| (à confirmer) | v1.0 | Lancement du moteur RAG interactif (LlamaIndex + Gemini 2.5 Flash + cache embeddings) | (à confirmer) |
| 2026-05-20 | (à confirmer) | Ajout des capacités programmatiques : génération directe, résumé et réordonnancement de passages (`gemini_client.py`) | (à confirmer) |

### 10.2 À venir (haut niveau)

| Échéance | Évolution | Statut |
|---|---|---|
| (à confirmer) | Support multi-PDF, interface web, persistance de l'historique | Hypothèse |

---

## 11. Questions ouvertes côté métier

> Décisions fonctionnelles non tranchées. Distinct des questions techniques (qui vivent dans le glossaire ou les ADR).

| # | Question | Impact si non tranché | Owner | Ouverte depuis |
|---|---|---|---|---|
| QF-01 | Quel document PDF doit être indexé par défaut ? L'URL est codée en dur. | Changer de document demande une modification du code | (à confirmer) | 2026-05-20 |
| QF-02 | Faut-il gérer la saisie `quit` / `exit` comme synonymes de `q` ? | Expérience utilisateur dégradée | (à confirmer) | 2026-05-20 |

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

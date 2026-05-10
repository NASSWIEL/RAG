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

# Documentation fonctionnelle — RAG Question-Answering

| Champ | Valeur |
|---|---|
| **Dernière mise à jour** | 2026-05-10 |
| **Mise à jour par** | agent IA |
| **PR de référence** | 04b7fd4 |
| **Owner produit** | (à confirmer) |
| **Statut** | Brouillon |

> **Pitch en 1 phrase** : Système interactif en ligne de commande qui permet à un utilisateur de poser des questions en langage naturel sur un document PDF distant et d'obtenir des réponses générées par un modèle de langage (Gemini) grâce à la recherche sémantique (RAG).

---

## 1. Vision et raison d'être

### 1.1 Problème adressé

Les documents longs (articles de recherche, rapports PDF) sont difficiles à consulter rapidement. Un utilisateur qui souhaite trouver une information précise doit parcourir l'intégralité du document. Ce système permet d'interroger directement un PDF en langage naturel et d'obtenir une réponse synthétique, sans lecture manuelle.

### 1.2 Bénéfices clés

| Pour | Bénéfice | Mesuré par |
|---|---|---|
| **Chercheur / analyste** | Obtenir une réponse ciblée sur un article PDF sans le lire en entier | Temps de recherche réduit (à confirmer) |
| **Développeur / expérimentateur** | Tester rapidement une pipeline RAG sur n'importe quel PDF distant | Facilité d'intégration d'un nouveau document |

### 1.3 Hors-scope

> Ce que le produit ne fait PAS — délibérément. Inclure les fonctionnalités souvent confondues avec ce produit.

| Ce qui est hors-scope | Pourquoi | Alternative |
|---|---|---|
| Interrogation de plusieurs PDFs simultanément | Le moteur est initialisé avec une URL unique par session | (à confirmer) — lancer plusieurs instances |
| Interface graphique (UI web) | L'interaction se fait exclusivement en ligne de commande | (à confirmer) |
| Modification ou annotation du PDF source | Le système est en lecture seule | Outil d'annotation dédié |
| Authentification / gestion multi-utilisateurs | Pas de gestion de session | (à confirmer) |

---

## 2. Personas et utilisateurs

| Persona | Rôle | Fréquence d'usage | Objectif principal | Compétences attendues |
|---|---|---|---|---|
| **Utilisateur CLI** | Toute personne souhaitant interroger un PDF | Ponctuel / ad hoc | Poser des questions en langage naturel et obtenir des réponses | Savoir lancer un script Python en ligne de commande |
| **Développeur** | Expérimentateur ou intégrateur de la pipeline RAG | Ponctuel | Adapter ou étendre le moteur à d'autres documents | Connaissance Python, compréhension des LLM |

> Pour chaque persona, lister les **cas d'usage prioritaires** dans §3.

---

## 3. Cas d'usage

### 3.1 Carte des cas d'usage

| ID | Cas d'usage | Persona | Fréquence | Criticité |
|---|---|---|---|---|
| UC-01 | Interroger un PDF distant en langage naturel | Utilisateur CLI | Ponctuel | Critique |
| UC-02 | Réutiliser les embeddings mis en cache pour un PDF déjà indexé | Utilisateur CLI / Développeur | Fréquent après la première exécution | Standard |

### 3.2 Détail par cas d'usage

> Un bloc par UC critique. Forme abrégée pour les UC standards.

#### UC-01 — Interroger un PDF distant en langage naturel

| Méta | Valeur |
|---|---|
| **Persona** | Utilisateur CLI |
| **Pré-conditions** | Python installé, clé API Gemini configurée, accès réseau au PDF (`https://arxiv.org/pdf/2005.11401.pdf`) |
| **Post-conditions** | L'utilisateur a reçu une réponse textuelle à sa question ; la session continue jusqu'à saisie de `q` |
| **Déclencheur** | Lancement de `main.py` puis saisie d'une question au prompt |
| **Fréquence** | Ponctuel / ad hoc |
| **Volumétrie** | Une question par interaction utilisateur (à confirmer) |
| **SLA** | (à confirmer) — dépend du temps de génération Gemini et de la taille du PDF |
| **Endpoint(s) / écran(s)** | Interface ligne de commande — `main.py` ; voir [contracts.md](contracts.md) |
| **Règles métier appliquées** | RG-01, RG-02 (voir §4) |

**Scénario nominal**

1. L'utilisateur lance `python main.py`.
2. Le moteur télécharge le PDF depuis l'URL configurée et construit l'index vectoriel (ou charge le cache si disponible).
3. Le prompt `Your question:` s'affiche.
4. L'utilisateur saisit une question en langage naturel.
5. Le moteur effectue une recherche sémantique (`similarity_top_k=3`) et génère une réponse via Gemini.
6. La réponse s'affiche sous la forme `Answer: <texte>`.
7. Le prompt réapparaît ; l'utilisateur peut poser une nouvelle question ou saisir `q` pour quitter.

**Scénarios alternatifs**

| Variante | Condition | Comportement |
|---|---|---|
| Cache disponible (UC-02) | Un index pour ce PDF a déjà été calculé et sauvegardé dans `./storage/` | Étape 2 : le moteur charge l'index depuis le disque sans re-télécharger ni re-indexer le PDF |

**Cas d'erreur métier**

| Cas | Détection | Message utilisateur | Effet |
|---|---|---|---|
| PDF inaccessible | Erreur réseau lors du téléchargement | (à confirmer) — exception Python non interceptée | La session ne démarre pas |
| Clé API Gemini absente ou invalide | Lors de l'initialisation du LLM | (à confirmer) — exception Python non interceptée | La session ne démarre pas |

**Pièges connus**

- ⚠️ La première exécution sur un PDF peut être longue (téléchargement + indexation + génération des embeddings). Les exécutions suivantes sur le même PDF sont significativement plus rapides grâce au cache.
- ⚠️ L'URL du PDF est codée en dur dans `main.py` (`https://arxiv.org/pdf/2005.11401.pdf`). Changer de document nécessite de modifier le code source.

---

## 4. Règles de gestion

> Toute règle métier nommée. Une RG = une assertion sur le système, formulée en langage métier, idéalement testable.

### 4.1 Catalogue

| ID | Nom | Domaine | Cas d'usage liés | Statut |
|---|---|---|---|---|
| RG-01 | Interrogation en mode boucle interactive | Interaction utilisateur | UC-01 | Active |
| RG-02 | Mise en cache des embeddings par URL | Performance / stockage | UC-01, UC-02 | Active |

### 4.2 Détail par règle

#### RG-01 — Interrogation en mode boucle interactive

- **Énoncé** : Le système accepte des questions successives de l'utilisateur dans une boucle continue ; la session ne se termine que si l'utilisateur saisit `q`.
- **Pré-conditions** : Moteur RAG initialisé avec succès (index chargé ou construit).
- **Effet** : Toute saisie autre que `q` est transmise au moteur de requête ; la réponse est affichée avant de revenir au prompt.
- **Origine** : Conception de l'interface CLI — [`main.py`](../main.py)
- **Implémentation** : [`main.py:13-18`](../main.py#L13)
- **Tests fonctionnels** : (à confirmer)
- **Évolutions** :
  | Date | PR | Changement |
  |---|---|---|
  | 2026-05-10 | 04b7fd4 | Création |

#### RG-02 — Mise en cache des embeddings par URL

- **Énoncé** : Le moteur calcule un hash MD5 de l'URL du PDF et stocke l'index vectoriel dans `./storage/index_<hash>/`. Si ce répertoire existe déjà au démarrage, l'index est chargé depuis le cache sans re-traiter le PDF.
- **Pré-conditions** : L'URL du PDF est connue au démarrage du moteur.
- **Effet** : Évite de re-télécharger et re-indexer un document déjà traité ; réduit le temps de démarrage.
- **Origine** : Optimisation de performance — [`rag_engine.py`](../rag_engine.py)
- **Implémentation** : [`rag_engine.py:31-51`](../rag_engine.py#L31)
- **Tests fonctionnels** : (à confirmer)
- **Évolutions** :
  | Date | PR | Changement |
  |---|---|---|
  | 2026-05-10 | 04b7fd4 | Création |

---

## 5. Workflows et machines à états

> Ce système n'expose pas d'entité métier avec cycle de vie (pas de commande, dossier ou contrat). La session utilisateur est éphémère : elle commence au lancement de `main.py` et se termine à la saisie de `q`. Aucune machine à états persistante n'est applicable.

---

## 6. Données métier de référence

> Référentiels et énumérations qui ont une signification métier. Le détail technique vit dans [data-model.md](data-model.md).

| Référentiel | Valeurs | Source de vérité | Mise à jour |
|---|---|---|---|
| **Commande de sortie** | `q` — quitte la session interactive | [`main.py:15`](../main.py#L15) | Manuel (code) |
| **Répertoire de cache** | `./storage/` (sous-dossier par hash MD5 de l'URL) | [`rag_engine.py:16`](../rag_engine.py#L16) | Automatique à la première indexation |

---

## 7. Cas limites et pathologiques

> Comportements aux extrémités du fonctionnel. Ces cas sont la principale source de bugs en production — les documenter explicitement.

| Cas | Comportement attendu | Justification |
|---|---|---|
| **PDF vide ou non lisible** | (à confirmer) — exception non interceptée, la session ne démarre pas | Aucune gestion explicite dans le code actuel |
| **Question vide (entrée vide)** | (à confirmer) — transmise au moteur de requête comme chaîne vide | Aucun filtrage de la saisie dans `main.py` |
| **PDF très volumineux** | Indexation longue à la première exécution ; mises en cache pour les suivantes | RG-02 |
| **Réseau coupé en cours de téléchargement** | (à confirmer) — exception non interceptée | Aucune gestion de reprise partielle |
| **Cache corrompu** | (à confirmer) — erreur au chargement de l'index | Aucune détection de corruption du répertoire `./storage/` |
| **Caractères spéciaux dans la question** | Transmis tels quels au moteur ; comportement dépend de Gemini | (à confirmer) |

---

## 8. Conformité et contraintes externes

| Contrainte | Origine | Implémentation |
|---|---|---|
| Licence du document PDF interrogé | Le PDF source (`arxiv.org`) est soumis à la licence des auteurs | L'utilisateur est responsable de l'usage du contenu ; le système ne vérifie pas les droits |
| Conditions d'utilisation de l'API Gemini | Google Cloud / Gemini API | Clé API fournie par l'utilisateur ; aucune gestion côté application |
| Données personnelles dans les questions | (à confirmer) — les questions sont transmises à l'API Gemini (service tiers) | Aucun mécanisme d'anonymisation dans le code actuel |

---

## 9. KPIs et indicateurs métier

| KPI | Définition | Cible | Source | Dashboard |
|---|---|---|---|---|
| Temps de réponse par question | Délai entre la saisie et l'affichage de la réponse | (à confirmer) | Logs console | Aucun dashboard défini |
| Taux de succès des requêtes Gemini | Proportion de questions ayant obtenu une réponse sans erreur | (à confirmer) | (à confirmer) | Aucun dashboard défini |

> Note : ces KPIs orientent les décisions produit. Toute fonctionnalité majeure devrait s'y rattacher.

---

## 10. Roadmap et évolutions notables

> Historique compressé : grands jalons fonctionnels passés et à venir. Pas un changelog détaillé (qui vit dans `CHANGELOG.md` ou les releases).

### 10.1 Jalons passés

| Date | Version | Évolution majeure | PR / RFC |
|---|---|---|---|
| 2026-05-10 | (à confirmer) | Mise en place de la pipeline RAG interactive avec cache d'embeddings et intégration Gemini | 03de3db |

### 10.2 À venir (haut niveau)

| Échéance | Évolution | Statut |
|---|---|---|
| (à confirmer) | Support de plusieurs PDFs simultanés | En discussion |
| (à confirmer) | Interface web ou API REST | Hypothèse |

---

## 11. Questions ouvertes côté métier

> Décisions fonctionnelles non tranchées. Distinct des questions techniques (qui vivent dans le glossaire ou les ADR).

| # | Question | Impact si non tranché | Owner | Ouverte depuis |
|---|---|---|---|---|
| QF-01 | Quelle est la politique de rétention du cache d'embeddings ? Faut-il prévoir une purge automatique ? | Croissance non bornée du répertoire `./storage/` | (à confirmer) | 2026-05-10 |
| QF-02 | Le système doit-il supporter d'autres PDF que `2005.11401.pdf` sans modification du code ? | Actuellement l'URL est codée en dur | (à confirmer) | 2026-05-10 |

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

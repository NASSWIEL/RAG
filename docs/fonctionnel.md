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

> **Pitch en 1 phrase** : Démo en ligne de commande qui indexe un PDF (par défaut l'article RAG, arxiv 2005.11401) et répond à des questions en langage naturel via un pipeline LlamaIndex + Gemini 2.5 Flash.

---

## 1. Vision et raison d'être

### 1.1 Problème adressé

Interroger un document PDF long en langage naturel est laborieux : l'utilisateur doit lire, chercher, recouper manuellement. Ce système construit un index vectoriel du PDF et délègue la recherche sémantique + la génération de réponses à un LLM (Gemini 2.5 Flash), offrant un accès conversationnel immédiat au contenu.

### 1.2 Bénéfices clés

| Pour | Bénéfice | Mesuré par |
|---|---|---|
| Chercheur / développeur | Interroge un PDF long sans le lire intégralement | (à confirmer) |
| Démonstrateur technique | Valide un pipeline RAG complet en quelques minutes | (à confirmer) |

### 1.3 Hors-scope

> Ce que le produit ne fait PAS — délibérément.

| Ce qui est hors-scope | Pourquoi | Alternative |
|---|---|---|
| Interface graphique ou web | Démo CLI uniquement (à confirmer) | (à confirmer) |
| Indexation multi-documents | Un seul PDF par session, URL codée en dur | (à confirmer) |
| Authentification / gestion d'utilisateurs | Démo locale sans serveur | (à confirmer) |

---

## 2. Personas et utilisateurs

| Persona | Rôle | Fréquence d'usage | Objectif principal | Compétences attendues |
|---|---|---|---|---|
| **Utilisateur CLI** | Développeur / chercheur qui lance `main.py` | Ponctuel | Obtenir une réponse textuelle à partir d'un PDF | Python, terminal |

> Pour chaque persona, lister les **cas d'usage prioritaires** dans §3.

---

## 3. Cas d'usage

### 3.1 Carte des cas d'usage

| ID | Cas d'usage | Persona | Fréquence | Criticité |
|---|---|---|---|---|
| UC-01 | Session interactive de questions sur un PDF | Utilisateur CLI | Ponctuel | Critique |
| UC-02 | Rechargement de l'index depuis le cache | Utilisateur CLI | Ponctuel | Standard |

### 3.2 Détail par cas d'usage

> Un bloc par UC critique. Forme abrégée pour les UC standards.

#### UC-01 — Session interactive de questions sur un PDF

| Méta | Valeur |
|---|---|
| **Persona** | Utilisateur CLI |
| **Pré-conditions** | `GOOGLE_API_KEY` configurée ; accès internet pour télécharger le PDF si cache absent |
| **Post-conditions** | L'utilisateur a reçu une réponse textuelle à chaque question posée |
| **Déclencheur** | Exécution de `python main.py` |
| **Fréquence** | Ponctuelle |
| **Volumétrie** | 1 session / exécution ; N questions par session |
| **SLA** | (à confirmer) |
| **Endpoint(s) / écran(s)** | CLI — stdin / stdout |
| **Règles métier appliquées** | RG-01 (voir §4) |

**Scénario nominal**

1. L'utilisateur lance `python main.py`.
2. Le moteur vérifie si un index vectoriel existe en cache pour l'URL `https://arxiv.org/pdf/2005.11401.pdf` (clé MD5 de l'URL sous `./storage/`).
3. Si aucun cache : téléchargement du PDF, extraction du texte, construction de l'index vectoriel, sauvegarde sur disque.
4. Si cache présent : chargement direct de l'index (étapes de téléchargement et d'embedding sautées).
5. Le système affiche l'invite `Your question:`.
6. L'utilisateur saisit une question en langage naturel.
7. Le moteur effectue une recherche sémantique (top_k = 3) dans l'index et génère une réponse via Gemini 2.5 Flash.
8. La réponse est affichée sous `Answer:`.
9. Le système réaffiche l'invite ; l'utilisateur peut poser une nouvelle question ou saisir `q` pour quitter.

**Scénarios alternatifs**

| Variante | Condition | Comportement |
|---|---|---|
| Quitter la session | L'utilisateur saisit `q` | La boucle se termine immédiatement, aucune réponse générée |

**Cas d'erreur métier**

| Cas | Détection | Message utilisateur | Effet |
|---|---|---|---|
| Téléchargement PDF échoué | Exception `requests` | (à confirmer) | Interruption de l'initialisation |
| Clé API invalide | Exception Gemini | (à confirmer) | Interruption de l'initialisation |

**Pièges connus**

- ⚠️ Le PDF source est codé en dur dans `main.py` (`pdf_url = "https://arxiv.org/pdf/2005.11401.pdf"`). Changer de document nécessite de modifier le code.
- ⚠️ Le cache d'embeddings est lié à l'URL via MD5 : si l'URL change d'un caractère, l'index est intégralement recalculé.

---

## 4. Règles de gestion

> Toute règle métier nommée. Une RG = une assertion sur le système, formulée en langage métier, idéalement testable.

### 4.1 Catalogue

| ID | Nom | Domaine | Cas d'usage liés | Statut |
|---|---|---|---|---|
| RG-01 | Cache par URL | Indexation | UC-01, UC-02 | Active |

### 4.2 Détail par règle

#### RG-01 — Cache par URL

- **Énoncé** : Si un index vectoriel a déjà été construit pour l'URL donnée (clé = MD5 de l'URL), le système le charge depuis le disque sans re-télécharger ni re-embedder le PDF.
- **Pré-conditions** : Le répertoire `./storage/index_<md5>` existe et est lisible.
- **Effet** : L'étape de téléchargement, d'extraction et d'embedding est intégralement sautée ; le temps d'initialisation est réduit à la lecture du cache.
- **Origine** : Décision technique — éviter les appels d'embedding répétés et coûteux (à confirmer).
- **Implémentation** : [rag_engine.py:64-74](rag_engine.py)
- **Tests fonctionnels** : (à confirmer)
- **Évolutions** :
  | Date | PR | Changement |
  |---|---|---|
  | 2026-05-20 | (à confirmer) | Création |

---

## 5. Workflows et machines à états

> Le système est une application CLI sans entité persistante avec cycle de vie propre. Section non applicable pour cette démo. (à confirmer)

---

## 6. Données métier de référence

> Référentiels et énumérations qui ont une signification métier. Le détail technique vit dans [data-model.md](data-model.md).

| Référentiel | Valeurs | Source de vérité | Mise à jour |
|---|---|---|---|
| **Commande de sortie** | `q` — quitte la session interactive | [`main.py:14`](main.py) | Manuel |

---

## 7. Cas limites et pathologiques

> Comportements aux extrémités du fonctionnel. Ces cas sont la principale source de bugs en production — les documenter explicitement.

| Cas | Comportement attendu | Justification |
|---|---|---|
| **Question vide** | (à confirmer) — la boucle retransmet à Gemini | Aucune validation de l'entrée dans le code actuel |
| **PDF inaccessible** | Interruption avec exception `requests` | Timeout 30 s configuré dans `pdf_loader.py:21` |
| **Cache corrompu** | (à confirmer) — LlamaIndex tente le chargement et lève une exception | Aucune vérification d'intégrité du cache actuellement |
| **Concurrence** | Non géré — usage mono-utilisateur local | Application CLI sans serveur |
| **Rejeu d'une même URL** | L'index est rechargé depuis le cache, aucun re-embedding | RG-01 |

---

## 8. Conformité et contraintes externes

| Contrainte | Origine | Implémentation |
|---|---|---|
| Licence d'usage Gemini API | Conditions Google | Clé API via `config.py` — usage soumis aux CGU Google AI |
| Licence du PDF indexé | Dépend du document source | L'article arxiv 2005.11401 est sous licence auteur arXiv (à confirmer) |

---

## 9. KPIs et indicateurs métier

| KPI | Définition | Cible | Source | Dashboard |
|---|---|---|---|---|
| Temps de première réponse | Délai entre la saisie de la question et l'affichage de la réponse | (à confirmer) | stdout (mesure manuelle) | (à confirmer) |

> Note : démo technique — aucun KPI produit formalisé à ce stade. (à confirmer)

---

## 10. Roadmap et évolutions notables

> Historique compressé : grands jalons fonctionnels passés et à venir. Pas un changelog détaillé (qui vit dans `CHANGELOG.md` ou les releases).

### 10.1 Jalons passés

| Date | Version | Évolution majeure | PR / RFC |
|---|---|---|---|
| 2026-05-20 | 0.0.0 | Mise en place du pipeline RAG interactif (LlamaIndex + Gemini 2.5 Flash) | (à confirmer) |

### 10.2 À venir (haut niveau)

| Échéance | Évolution | Statut |
|---|---|---|
| (à confirmer) | Support multi-documents / URL configurable | Hypothèse |

---

## 11. Questions ouvertes côté métier

> Décisions fonctionnelles non tranchées. Distinct des questions techniques (qui vivent dans le glossaire ou les ADR).

| # | Question | Impact si non tranché | Owner | Ouverte depuis |
|---|---|---|---|---|
| QF-01 | L'URL du PDF doit-elle être configurable sans modifier le code ? | Limite l'usage à un seul document | (à confirmer) | 2026-05-20 |
| QF-02 | Quel comportement adopter si le cache est corrompu ou incomplet ? | Risque d'erreur opaque pour l'utilisateur | (à confirmer) | 2026-05-20 |

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

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
| **Dernière mise à jour** | 2026-05-21 |
| **Mise à jour par** | agent IA (doc-patcher) |
| **PR de référence** | (à confirmer) |
| **Owner produit** | (à confirmer) |
| **Statut** | Brouillon |

> **Pitch en 1 phrase** : Système de question-réponse en ligne de commande qui permet à un utilisateur de poser des questions en langage naturel sur le contenu d'un document PDF, en exploitant un index vectoriel et le modèle Gemini pour générer les réponses.

---

## 1. Vision et raison d'être

### 1.1 Problème adressé

Consulter manuellement un long document PDF pour trouver une information précise est laborieux et peu reproductible. Ce système permet d'interroger le contenu d'un PDF en langage naturel et d'obtenir une réponse générée, sans devoir lire l'intégralité du document. À la première utilisation, l'index vectoriel est construit et mis en cache localement pour éviter de retraiter le PDF à chaque session.

### 1.2 Bénéfices clés

| Pour | Bénéfice | Mesuré par |
|---|---|---|
| Utilisateur CLI | Obtenir une réponse ciblée sur un PDF sans le lire intégralement | (à confirmer) |
| Utilisateur récurrent | Réponses rapides grâce au cache local des embeddings | Présence du répertoire `./storage/index_<hash>` |

### 1.3 Hors-scope

> Ce que le produit ne fait PAS — délibérément. Inclure les fonctionnalités souvent confondues avec ce produit.

| Ce qui est hors-scope | Pourquoi | Alternative |
|---|---|---|
| Indexation de plusieurs PDFs simultanément | L'URL est codée en dur dans `main.py`, une seule source par exécution | Modifier `main.py` pour passer l'URL en argument |
| Interface graphique ou web | Application CLI uniquement | (à confirmer) |
| Réponses en streaming | `query_engine.query()` retourne une réponse complète | (à confirmer) |
| Gestion multi-utilisateurs ou multi-sessions | Pas de serveur, usage local mono-utilisateur | (à confirmer) |

---

## 2. Personas et utilisateurs

| Persona | Rôle | Fréquence d'usage | Objectif principal | Compétences attendues |
|---|---|---|---|---|
| **Utilisateur CLI** | Utilisateur final posant des questions sur un PDF via le terminal | Ponctuel / quotidien | Obtenir des réponses précises sur le contenu d'un PDF | Savoir lancer un script Python, accès internet pour téléchargement du PDF |

> Pour chaque persona, lister les **cas d'usage prioritaires** dans §3.

---

## 3. Cas d'usage

### 3.1 Carte des cas d'usage

| ID | Cas d'usage | Persona | Fréquence | Criticité |
|---|---|---|---|---|
| UC-01 | Poser une question sur le contenu du PDF indexé | Utilisateur CLI | À chaque session interactive | Critique |
| UC-02 | Indexation initiale du PDF (premier lancement) | Utilisateur CLI | Ponctuel (une fois par URL) | Critique |

### 3.2 Détail par cas d'usage

> Un bloc par UC critique. Forme abrégée pour les UC standards.

#### UC-01 — Poser une question sur le contenu du PDF indexé

| Méta | Valeur |
|---|---|
| **Persona** | Utilisateur CLI |
| **Pré-conditions** | L'index vectoriel existe en cache (`./storage/index_<hash>`) ou le PDF est accessible via l'URL configurée dans `main.py` |
| **Post-conditions** | Succès : une réponse en langage naturel est affichée dans le terminal. Échec : exception Python propagée si le PDF est inaccessible |
| **Déclencheur** | L'utilisateur saisit une question dans le terminal après le prompt `Your question:` |
| **Fréquence** | Autant de fois que souhaité dans une session |
| **Volumétrie** | (à confirmer) |
| **SLA** | (à confirmer) |
| **Endpoint(s) / écran(s)** | Interface CLI — boucle interactive dans `main.py` |
| **Règles métier appliquées** | RG-01, RG-02 (voir §4) |

**Scénario nominal**

1. L'utilisateur lance `python main.py` ; le moteur RAG charge l'index depuis le cache.
2. Le prompt `Your question:` s'affiche dans le terminal.
3. L'utilisateur saisit sa question en langage naturel.
4. Le moteur effectue une recherche sémantique (`similarity_top_k=3`) et génère une réponse via Gemini.
5. La réponse est affichée sous `Answer:`.
6. La boucle reprend à l'étape 2 jusqu'à ce que l'utilisateur tape `q`.

**Scénarios alternatifs**

| Variante | Condition | Comportement |
|---|---|---|
| Première exécution (pas de cache) | Aucun répertoire `./storage/index_<hash>` trouvé | Le PDF est téléchargé, l'index est construit et sauvegardé avant d'entrer en boucle interactive (UC-02) |

**Cas d'erreur métier**

| Cas | Détection | Message utilisateur | Effet |
|---|---|---|---|
| PDF inaccessible à l'URL | Au moment du téléchargement (première exécution) | Exception Python propagée dans le terminal | Arrêt du programme |

**Pièges connus**

- ⚠️ L'URL du PDF est codée en dur dans `main.py` (`https://arxiv.org/pdf/2005.11401.pdf`). Changer l'URL sans supprimer le répertoire `./storage` correspondant peut causer une confusion si un hash MD5 devait collisionner.
- ⚠️ Taper `q` est le seul moyen de quitter proprement la boucle interactive.

---

## 4. Règles de gestion

> Toute règle métier nommée. Une RG = une assertion sur le système, formulée en langage métier, idéalement testable.

### 4.1 Catalogue

| ID | Nom | Domaine | Cas d'usage liés | Statut |
|---|---|---|---|---|
| RG-01 | Cache d'index par URL | Indexation | UC-01, UC-02 | Active |
| RG-02 | Sortie par mot-clé `q` | Interface utilisateur | UC-01 | Active |

### 4.2 Détail par règle

#### RG-01 — Cache d'index par URL

- **Énoncé** : L'index vectoriel d'un PDF est identifié par un hash MD5 de son URL et mis en cache localement ; si ce cache existe, il est réutilisé sans retélécharger ni réindexer le PDF.
- **Pré-conditions** : S'applique à chaque démarrage du moteur RAG.
- **Effet** : Évite de reconstruire l'index à chaque session ; réduit le temps de démarrage pour les usages récurrents.
- **Origine** : Contrainte de performance / optimisation développeur.
- **Implémentation** : [`rag_engine.py:31-51`](../rag_engine.py)
- **Tests fonctionnels** : (à confirmer)
- **Évolutions** :
  | Date | PR | Changement |
  |---|---|---|
  | 2026-05-21 | (à confirmer) | Création |

#### RG-02 — Sortie par mot-clé `q`

- **Énoncé** : La saisie de `q` (insensible à la casse) dans le prompt de question met fin à la session interactive.
- **Pré-conditions** : S'applique pendant la boucle interactive de `main.py`.
- **Effet** : Arrête proprement le programme sans erreur.
- **Origine** : Convention CLI.
- **Implémentation** : [`main.py:15`](../main.py)
- **Tests fonctionnels** : (à confirmer)
- **Évolutions** :
  | Date | PR | Changement |
  |---|---|---|
  | 2026-05-21 | (à confirmer) | Création |

---

---

## 7. Cas limites et pathologiques

> Comportements aux extrémités du fonctionnel. Ces cas sont la principale source de bugs en production — les documenter explicitement.

| Cas | Comportement attendu | Justification |
|---|---|---|
| **PDF vide ou invalide** | (à confirmer) — exception probable à l'indexation | Le chargeur PDF n'a pas de garde-fou explicite dans le code |
| **Question vide** | (à confirmer) — la chaîne vide est transmise au moteur de requête | Pas de validation de la saisie dans `main.py` |
| **URL inaccessible** | Arrêt du programme avec exception | Aucune retry ni message utilisateur explicite |
| **Cache corrompu** | (à confirmer) — l'index tente d'être chargé depuis un répertoire invalide | Pas de vérification d'intégrité du cache |
| **Concurrence** | Non applicable — usage mono-utilisateur local | Pas de serveur |
| **Rejeu (même question)** | Réponse identique ou proche ; idempotent | La recherche sémantique est déterministe |

---

## 8. Conformité et contraintes externes

| Contrainte | Origine | Implémentation |
|---|---|---|
| Licence du modèle d'embedding `BAAI/bge-small-en-v1.5` | MIT (à confirmer) | Utilisé via `llama-index-embeddings-huggingface` |
| Conditions d'utilisation de l'API Gemini | Google ToS | Via `gemini_client.py` (à confirmer) |

---

## 9. KPIs et indicateurs métier

| KPI | Définition | Cible | Source | Dashboard |
|---|---|---|---|---|
| (à confirmer) | Pas de KPI défini à ce stade | — | — | — |

> Note : ces KPIs orientent les décisions produit. Toute fonctionnalité majeure devrait s'y rattacher.

---

## 10. Roadmap et évolutions notables

> Historique compressé : grands jalons fonctionnels passés et à venir. Pas un changelog détaillé (qui vit dans `CHANGELOG.md` ou les releases).

### 10.1 Jalons passés

| Date | Version | Évolution majeure | PR / RFC |
|---|---|---|---|
| 2026-05-21 | (à confirmer) | Mise en place du moteur RAG avec cache d'embeddings et boucle interactive | (à confirmer) |

### 10.2 À venir (haut niveau)

| Échéance | Évolution | Statut |
|---|---|---|
| (à confirmer) | (à confirmer) | hypothèse |

---

## 11. Questions ouvertes côté métier

> Décisions fonctionnelles non tranchées. Distinct des questions techniques (qui vivent dans le glossaire ou les ADR).

| # | Question | Impact si non tranché | Owner | Ouverte depuis |
|---|---|---|---|---|
| QF-01 | L'URL du PDF doit-elle rester codée en dur ou devenir un paramètre CLI ? | Limite la réutilisabilité du système à un seul document | (à confirmer) | 2026-05-21 |
| QF-02 | Quelle politique appliquer si le cache est invalide ou corrompu ? | Risque de réponses incorrectes sans avertissement | (à confirmer) | 2026-05-21 |

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

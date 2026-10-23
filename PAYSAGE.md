# Paysage

[🇫🇷](PAYSAGE.md)&nbsp;&nbsp;|&nbsp;&nbsp;[🇬🇧](LANDSCAPE.md)

Comment `ann-router` se compare au fait de *choisir un seul moteur*. Chaque outil
est noté sur **le travail de ce projet, sélectionner le bon backend ANN à partir
de critères mesurés**, sans être pénalisé pour exceller à un autre travail
(servir un moteur donné).

## Positionnement

`ann-router` ne concurrence pas FAISS, HNSW, Qdrant et les autres : il les
**orchestre**. Ils résolvent *l'indexation et le service* ; ann-router résout
*lequel utiliser et pourquoi*. Son analogue le plus proche n'est pas une autre
bibliothèque vectorielle mais son propre frère `best-engine-ai-helper`, qui route
entre des LLM.

## En un coup d'œil

<!-- TABLE:START -->
| Outil de recherche vectorielle rapide | Sélection mesurée | Justification | Multi-moteur | Gère le churn | Filtre métadonnées | Persistance | Routeur testé en rappel | Accélération GPU | Compression vectorielle | Passage à l'échelle | Cloud managé |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ann-router** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| FAISS | ⭐ | ⭐ | ⭐ | ⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐ |
| HNSW | ⭐ | ⭐ | ⭐ | ⭐ | ⭐ | ⭐ | ⭐ | ⭐ | ⭐ | ⭐ | ⭐ |
| Annoy | ⭐ | ⭐ | ⭐ | ⭐ | ⭐ | ⭐⭐ | ⭐ | ⭐ | ⭐⭐ | ⭐ | ⭐ |
| turbovec | ⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ |
| Qdrant | ⭐ | ⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| pgvector | ⭐ | ⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| pgvectorscale | ⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Milvus | ⭐ | ⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Weaviate | ⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Vespa | ⭐ | ⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| LanceDB | ⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Chroma | ⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐ |
| LangChain VectorStores | ⭐⭐ | ⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
<!-- TABLE:END -->

(Un moteur unique obtient ⭐ en « sélection mesurée » parce qu'il *est* la
sélection : il n'y a rien à décider. C'est précisément ce que traite ann-router.)

## Carte de positionnement

<!-- FIGURE:START -->
Représentation 2D du tableau ci-dessus.

![Carte de positionnement](https://raw.githubusercontent.com/warith-harchaoui/ann-router/main/assets/paysage.png)

La carte est un résumé en 2D des 11 critères : à lire comme une forme, pas comme un classement. « ann-router » se situe dans le coin en haut à droite. Les axes se lisent **Horizontal : Efficacité ↔ Robustesse** et **Vertical : Adaptabilité ↔ Précision**.

L'axe horizontal révèle une tension entre la capacité à traiter rapidement un grand nombre de requêtes et la capacité à maintenir un fonctionnement fiable dans des conditions difficiles. Les solutions qui privilégient l'efficacité, comme HNSW, Annoy et turbovec, se positionnent à gauche, tandis que celles qui mettent l'accent sur la robustesse, notamment Milvus, Vespa et pgvectorscale, se regroupent à droite. Ann-router, notre option de référence, se trouve bien placée, bénéficiant d'un bon équilibre entre ces deux aspects, mais elle n'est pas la plus performante sur l'un ou l'autre.

L'axe vertical met en opposition la facilité d'adaptation à de nouveaux besoins et la précision des résultats. Les solutions qui se veulent flexibles et capables de s'intégrer à des environnements variés, comme Weaviate, Milvus et Qdrant, se situent en bas du graphique. À l'inverse, FAISS, HNSW et Annoy, ainsi que LangChain VectorStores, se distinguent par leur capacité à fournir des résultats précis et fiables, se positionnant en haut. Ann-router, encore une fois, se trouve dans une position intermédiaire, ce qui suggère qu'elle peut être moins adaptable que certaines alternatives, mais qu'elle offre une bonne précision.

On observe une concentration importante de solutions dans le quadrant inférieur droit, regroupant Qdrant, pgvector, pgvectorscale, Milvus, Weaviate et Vespa. Elles partagent un profil axé sur la robustesse et l'adaptabilité, ce qui les rend potentiellement intéressantes pour des environnements où la stabilité et la flexibilité sont primordiales. À l'opposé, les options FAISS, HNSW, Annoy et turbovec se regroupent dans le coin supérieur gauche, privilégiant l'efficacité et la précision, ce qui les rend plus adaptées aux applications où la vitesse et la qualité des résultats sont essentielles. Ann-router et LangChain VectorStores, quant à elles, se trouvent dans le coin supérieur droit, combinant robustesse et précision, ce qui les rend potentiellement polyvalentes, bien qu'elles ne soient pas les leaders sur chaque axe.
<!-- FIGURE:END -->

## Fiche par outil

### FAISS
Le roi de l'échelle : IVF + PQ + GPU gèrent des milliards de vecteurs. Mais c'est
une *bibliothèque*, pas une décision : son rappel s'effondre sur des corpus
petits ou quasi-orthogonaux (l'étude `roitelet` a mesuré FAISS-HNSW à ~0,47 de
rappel à N=5k), il n'a pas de filtrage par métadonnées et choisir IVF vs Flat vs
PQ et leurs paramètres est exactement le travail qu'ann-router automatise.
ann-router route **vers** FAISS pour le régime très-gros-volume + GPU/batch.

### HNSW (hnswlib)
Meilleur rappel/latence en mémoire pour un corpus **stable**. Les suppressions
sont par pierres tombales (le graphe se dégrade), donc inadapté aux données
mouvantes ; ann-router les envoie plutôt vers turbovec et garde HNSW comme
défaut haute-précision en mémoire.

### Annoy
Figé, mappé en mémoire, merveilleusement léger pour un corpus **en lecture
seule** à mémoire serrée. Ne peut ni ajouter ni supprimer (ann-router l'expose en
`NotSupported`). L'outil idéal pour exactement un régime, qu'ann-router détecte.

### Qdrant / pgvector
La réponse persistance + filtre de métadonnées : HNSW sur disque plus filtrage
payload/SQL `WHERE`. Plus lourds à opérer qu'un index en mémoire, donc ann-router
n'y route que lorsque les critères demandent réellement du filtrage ou de la
durabilité, en préférant pgvector si un Postgres existe déjà.

### turbovec
Le spécialiste du corpus dynamique : ajout/suppression O(1) et TurboQuant 2-4
bits (~16×), fort sur Apple Silicon. Le rappel dépend des données (la
quantification est avec perte), donc ann-router le choisit pour la branche
*mises à jour fréquentes*, pas pour le rappel maximal en statique.

### pgvectorscale
La raison pour laquelle « prends juste Postgres » a cessé d'être une
concession. pgvectorscale ajoute à pgvector un index DiskANN avec compression
SBQ : les mesures publiées le placent à **471 requêtes par seconde à 99 % de
rappel sur 50 millions de vecteurs, là où Qdrant en tient 41** — un facteur
onze du côté du rappel, obtenu depuis le moteur que la plupart des équipes font
déjà tourner. Cela confirme la prémisse d'ann-router plutôt que de la menacer :
l'*index* choisi dans un moteur déplace le rappel et la latence autant que le
choix du moteur lui-même : c'est exactement l'axe sur lequel un routeur doit
raisonner. Router vers pgvector, c'est désormais demander quel index, pas
seulement quelle base.

### Milvus
La réponse distribuée : nœuds de requête, de données et d'index séparés,
construction d'index sur GPU (CAGRA), seul moteur ouvert à tenir un vrai
discours au-delà de cent millions de vecteurs. Le prix est une surface
d'exploitation — etcd, stockage objet, plusieurs types de nœuds — qui écrase
celle d'un index en mémoire. ann-router y route quand l'échelle dépasse
réellement une machine et le dit dans sa justification : personne ne devrait
hériter de ce cluster par accident.

### Weaviate
Le spécialiste de la recherche hybride : BM25 avec recherche dense fusionnés
dans une seule requête, filtres sur métadonnées appliqués dans la même passe.
Cela compte parce qu'une part mesurable des plaintes « la recherche vectorielle
est mauvaise » sont en réalité des échecs lexicaux sur des noms, des
références, des mots rares, que les plongements brouillent par construction.
C'est l'argument le plus fort de ce tableau pour dire que le *rappel sur un
corpus* n'est pas la même quantité que le *rappel sur une requête* : ann-router
traite un besoin hybride déclaré comme une entrée de routage, pas comme un
agrément.

### Vespa
Le plus capable avec le moins de confort : une pile de classement complète,
expressions tensorielles, classement multi-phases, vrai streaming sur données
structurées, pensée pour un moteur de recherche plutôt que pour un index
embarqué. On le choisit quand le travail est de classer, non de chercher le
plus proche voisin ; ann-router ne prétend pas l'abstraire, car adopter Vespa
est une architecture, pas un moteur de plus.

### LanceDB
Le pari colonnaire embarqué : format Lance, stockage versionné sans copie,
empreinte en processus qui en fait l'extrémité « sans serveur » de la recherche
sur disque. Au plus près de la niche d'Annoy mais sans la taxe de la lecture
seule ; solide quand un corpus doit vivre dans un stockage objet plutôt qu'en
mémoire vive.

### Chroma
Le défaut sans friction : pip install, métadonnées comme un dictionnaire, pas
de serveur. L'échelle et la compression ne sont pas le sujet ; prétendre le
contraire est la façon dont une équipe finit par migrer sous charge —
ann-router y lit le régime du prototype, puis dit quand le corpus l'a dépassé.

### LangChain VectorStores
Une large couche d'adaptateurs sur de nombreux stores, la plus proche en
esprit, mais elle unifie des *API*, pas des *décisions* : elle vous laissera
volontiers choisir le mauvais store. ann-router ajoute par-dessus la sélection
mesurée + la justification + la politique testée en rappel.

## La thèse

Épouser un seul moteur optimise pour une seule forme de problème. Les vrais
systèmes changent de forme : ils grandissent, se mettent à avoir besoin de
suppressions, ajoutent une exigence de filtre, migrent vers une machine à GPU.
ann-router garde l'*interface* fixe et laisse le *moteur* suivre le problème
mesuré, avec une justification lisible et surchargeable.

Le paysage de 2026 renforce cette thèse plutôt qu'il ne l'affaiblit. Les écarts
mesurés les plus larges ne séparent plus les moteurs mais les *index au sein
d'un même moteur* — DiskANN contre HNSW contre IVF contre une construction sur
GPU déplace le rappel avec la latence plus loin que changer de base de données
— et les moteurs se sont tous mis à en proposer plusieurs. Une décision qui
était « quel magasin » devient « quel magasin, quel index, quelle
quantification » : davantage de routage à faire, pas moins.

## Sources et date de relevé

Les notes ci-dessus ont été vérifiées le **2026-10-23**. Ce domaine bouge assez
vite pour qu'un tableau périmé soit un tableau faux : la date fait partie de
l'affirmation.

- [Open-source vector database benchmarks](https://aimultiple.com/open-source-vector-databases) — AIMultiple
- [Best vector databases in 2026: pricing, scale limits and architecture trade-offs](https://www.marktechpost.com/2026/05/10/best-vector-databases-in-2026-pricing-scale-limits-and-architecture-tradeoffs-across-nine-leading-systems/) — MarkTechPost
- [Best vector databases](https://encore.dev/articles/best-vector-databases) — Encore

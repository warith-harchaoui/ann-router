# Exemples ann-router

[🇫🇷](https://github.com/warith-harchaoui/ann-router/blob/main/EXEMPLES.md)&nbsp;&nbsp;|&nbsp;&nbsp;[🇬🇧](https://github.com/warith-harchaoui/ann-router/blob/main/EXAMPLES.md)

Un livre de recettes autonome. Chaque extrait de cette page a été exécuté
contre `ann-router 0.3.0` avec l'extra `[all]` installé, sur un Mac Apple
Silicon, le 2026-10-27 : les sorties en commentaire sont celles qui sont
revenues. Quand un chiffre dépend de la machine ou des données, c'est dit.

## Table des matières

- [Mise en place](#mise-en-place)
- [1. Route : quel moteur, pourquoi ?](#1-route--quel-moteur-pourquoi-)
- [2. auto_index : router, construire et chercher en un seul appel](#2-auto_index--router-construire-et-chercher-en-un-seul-appel)
- [3. Piloter un moteur précis, directement](#3-piloter-un-moteur-précis-directement)
- [4. Mesurer le rappel face à la vérité terrain exacte](#4-mesurer-le-rappel-face-à-la-vérité-terrain-exacte)
- [5. Recherche filtrée par métadonnées (Qdrant)](#5-recherche-filtrée-par-métadonnées-qdrant)
- [6. Capacités et disponibilité](#6-capacités-et-disponibilité)
- [7. Gérer honnêtement les modes d'échec](#7-gérer-honnêtement-les-modes-déchec)
- [8. Régler la politique sans toucher au code](#8-régler-la-politique-sans-toucher-au-code)
- [Ligne de commande](#ligne-de-commande)
- [Dépannage](#dépannage)

## Mise en place

```bash
pip install 'ann-router[all]'     # ou, depuis un clone : pip install -e '.[all]'
```

Chaque extrait Python ci-dessous suppose ces deux imports, rien de plus :

```python
import numpy as np
import ann_router as ar
```

La première chose à afficher reste quels moteurs cette machine atteint
réellement : le routeur ne renverra jamais qu'un moteur installé.

```python
print(ar.__version__)
# 0.3.0
print(ar.available_backends())
# ['exact', 'turbovec', 'hnsw', 'faiss', 'annoy', 'qdrant', 'pgvector']
print(ar.detect_hardware())
# apple_silicon
```

`available_backends()` est la ligne dont la sortie variera le plus chez vous :
sans aucun extra installé elle renvoie `['exact']` et tous les extraits de
routage ci-dessous tournent quand même, en se rabattant sur la référence en
force brute.

## 1. Route : quel moteur, pourquoi ?

```python
# target_recall abaissé sous HIGH_RECALL=0.9, le plafond calibré de turbovec
# (bench/results/calibrated_policy.yaml) -- sinon la politique l'écarte.
choice = ar.route(ar.Criteria(n_vectors=2_000_000, dim=768, dynamic=True, target_recall=0.85))
print(choice.backend)
# turbovec
print(choice.rationale)  # (la justification est toujours renvoyée en anglais)
# corpus receives frequent updates: turbovec offers O(1) add_with_ids/remove(id)
# with no index rebuild, plus TurboQuant 2-4 bit (~16x) compression — graph
# indexes degrade under deletes, so they are avoided here. target_recall=0.85
# is below HIGH_RECALL=0.9, within turbovec's calibrated range.
print([c["backend"] for c in choice.considered])
# ['turbovec', 'hnsw']

# Au target_recall=0.95 par défaut (>= HIGH_RECALL), ce même corpus dynamique
# route vers hnsw à la place -- turbovec n'est même plus dans la liste, pas
# parce qu'il n'est pas installé, mais parce qu'il ne peut pas atteindre ce rappel.
choice = ar.route(ar.Criteria(n_vectors=2_000_000, dim=768, dynamic=True))
print(choice.backend)
# hnsw
print([c["backend"] for c in choice.considered])
# ['hnsw']
```

Le routeur ne renvoie jamais qu'un moteur **installé** et `considered` porte le
drapeau de disponibilité qui a servi à décider :

```python
choice = ar.route(ar.Criteria(n_vectors=500_000, dim=768, dynamic=True, target_recall=0.85))
print([(c["backend"], c["available"], c["chosen"]) for c in choice.considered])
# [('turbovec', True, True), ('hnsw', True, False)]
```

Si le premier choix de la politique n'est *pas* installé, le routeur se rabat
puis le signale dans la justification. On peut lire cette formulation exacte
sans rien désinstaller, en demandant à un adaptateur de se déclarer absent :

```python
from ann_router.backends.turbovec_backend import TurboVecIndex

TurboVecIndex.is_available = classmethod(lambda cls: False)  # faire comme si l'extra manquait

choice = ar.route(ar.Criteria(n_vectors=500_000, dim=768, dynamic=True, target_recall=0.85))
print(choice.backend)
# hnsw
print(choice.rationale)
# Preferred backend 'turbovec' is not installed here, so the router fell back.
# dynamic corpus but turbovec unavailable: HNSW is the working fallback, though
# its tombstone deletes degrade the graph over time.
print([(c["backend"], c["available"], c["chosen"]) for c in choice.considered])
# [('turbovec', False, False), ('hnsw', True, True)]
```

(Ce détournement sert à lire le chemin de repli, ce n'est pas une API prise en
charge. Repartez d'un interpréteur neuf ensuite.)

## 2. auto_index : router, construire et chercher en un seul appel

```python
rng = np.random.default_rng(0)

# Sous l'EXACT_MAX_N=1000 calibré, le conseil du routeur est un balayage brut.
small = rng.standard_normal((800, 128)).astype("float32")
index, choice = ar.auto_index(small, ar.Criteria(n_vectors=800, dim=128))
print(choice.backend)
# exact
ids, distances = index.search(small[:2], k=5)
print(ids.shape)
# (2, 5)
print(int(ids[0, 0]))
# 0        (le plus proche voisin d'un point est lui-même)

# Cinq mille vecteurs dépassent déjà ce seuil : le même appel route donc vers
# hnsw. « Petit corpus » veut dire 10^3 ici, pas 10^4.
big = rng.standard_normal((5_000, 128)).astype("float32")
index, choice = ar.auto_index(big, ar.Criteria(n_vectors=5_000, dim=128))
print(choice.backend)
# hnsw
```

`auto_index` croit le tableau plutôt que les critères : si `n_vectors`/`dim`
contredisent `vectors.shape`, c'est la forme qui gagne — pour la décision de
routage aussi, pas seulement pour l'index construit.

## 3. Piloter un moteur précis, directement

```python
from ann_router.backends.hnsw import HNSWIndex

vecs = np.random.default_rng(1).standard_normal((10_000, 256)).astype("float32")
idx = HNSWIndex(dim=256, metric="cosine", M=16, ef=64).build(vecs)
ids, dist = idx.search(vecs[:1], k=10)
print(ids.shape, dist.shape)
# (1, 10) (1, 10)

idx.save("/tmp/my.hnsw")
idx2 = HNSWIndex(dim=256).load("/tmp/my.hnsw")
ids2, _ = idx2.search(vecs[:1], k=10)
print(int(ids2[0, 0]) == int(ids[0, 0]))
# True     (l'index rechargé répond à l'identique)
```

## 4. Mesurer le rappel face à la vérité terrain exacte

`ExactIndex` est la référence : un balayage complet en force brute, de rappel
1.0 par construction. Tout moteur approché peut être noté contre lui.

```python
from ann_router.backends.exact import ExactIndex
from ann_router.backends.turbovec_backend import TurboVecIndex

rng = np.random.default_rng(7)
corpus = rng.standard_normal((8_000, 128)).astype("float32")
q = rng.standard_normal((50, 128)).astype("float32")
truth, _ = ExactIndex(dim=128).build(corpus).search(q, 10)
pred, _ = TurboVecIndex(dim=128).build(corpus).search(q, 10)
recall = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(truth, pred)])
print(round(float(recall), 2))
# 0.85     (dépend des données ; TurboQuant est un procédé avec pertes)
```

Ou plus simplement, lancer le banc d'essai intégré sur chaque moteur installé :

```python
from ann_router._core_cli import do_bench

print(do_bench(n=5000, dim=128, k=10)["results"])
# {'exact':    {'recall': 1.0,   'build_s': 0.002, 'query_ms': 0.072},
#  'turbovec': {'recall': 0.736, 'build_s': 0.1,   'query_ms': 0.073},
#  'hnsw':     {'recall': 1.0,   'build_s': 0.174, 'query_ms': 0.011},
#  'faiss':    {'recall': 1.0,   'build_s': 0.02,  'query_ms': 0.028},
#  'annoy':    {'recall': 0.0,   'build_s': 0.088, 'query_ms': 0.015},
#  'qdrant':   {'recall': 1.0,   'build_s': 0.396, 'query_ms': 2.079}}
```

La colonne de rappel se reproduit à l'identique (le corpus est semé) ;
`build_s` avec `query_ms` sont le temps d'une exécution sur une machine, ils ne
se reproduiront pas. Deux détails de cette sortie relèvent de l'environnement,
pas du moteur : tous deux sont expliqués au [Dépannage](#dépannage). `annoy` à
0.0 de rappel, c'est le wheel PyPI cassé ; `pgvector` manque entièrement du
dictionnaire parce qu'aucun PostgreSQL ne tournait.

## 5. Recherche filtrée par métadonnées (Qdrant)

Qdrant est le moteur vers lequel le routeur se tourne quand la question est
« les plus proches voisins **où** `lang == 'fr'` ». L'adaptateur utilise par
défaut le client Qdrant embarqué en mémoire : cet extrait n'a besoin d'aucun
serveur.

```python
from ann_router.backends.qdrant_backend import QdrantIndex

vecs = np.random.default_rng(2).standard_normal((1_000, 64)).astype("float32")
payloads = [{"lang": "fr" if i % 2 else "en"} for i in range(1_000)]
idx = QdrantIndex(dim=64).build(vecs, payloads=payloads)

# plus proches voisins restreints aux documents en français :
ids, scores = idx.search_filter(vecs[:1], k=5, where={"lang": "fr"})
print(ids.tolist())
# [[361, 869, 409, 555, 85]]
print(all(int(i) % 2 == 1 for i in ids[0]))
# True     (chaque réponse est d'indice impair, donc lang == 'fr')
```

`pgvector` expose le même `search_filter` via un `WHERE payload->>...` SQL,
mais il lui faut un DSN vivant (`dsn=...` ou `ANN_ROUTER_PG_DSN`) : il n'est
donc pas exécutable depuis un simple clone.

## 6. Capacités et disponibilité

```python
print(ar.available_backends())
# ['exact', 'turbovec', 'hnsw', 'faiss', 'annoy', 'qdrant', 'pgvector']

caps = ar.all_capabilities()
print(caps["annoy"].supports_remove)
# False    (figé après construction)
print(caps["qdrant"].supports_filter)
# True
print(caps["annoy"].pip_extra)
# annoy    (le correctif est donc : pip install 'ann-router[annoy]')
```

`all_capabilities()` répond pour **tous** les moteurs enregistrés, installés ou
non : les descripteurs sont de simples données, lisibles sans la dépendance, ce
qui permet à `route()` d'expliquer un repli au lieu de s'écraser sur un import.

## 7. Gérer honnêtement les modes d'échec

```python
from ann_router.backends.annoy_backend import AnnoyIndex
from ann_router.base import NotSupported

idx = AnnoyIndex(dim=8).build(np.random.rand(100, 8).astype("float32"))
try:
    idx.remove(np.array([1]))
except NotSupported as e:
    print(e)
# annoy: remove() unsupported — rebuild the index without those ids
```

Les deux exceptions répondent délibérément à deux questions différentes.
`NotSupported` veut dire *ce moteur ne sait pas le faire, jamais* : reconstruire,
ou router ailleurs. `BackendUnavailable` veut dire *ce moteur n'est pas installé
ici* et porte la ligne `pip install` qui corrige.

## 8. Régler la politique sans toucher au code

```python
# élève à 100k le seuil de bascule exact -> ANN, pour cet appel :
choice = ar.route(ar.Criteria(n_vectors=50_000, dim=128), thresholds={"EXACT_MAX_N": 100_000})
print(choice.backend)
# exact
print(choice.rationale)
# n=50,000 < 100,000 (EXACT_MAX_N=100,000 scaled to a 10 ms budget): a
# brute-force scan is already instant and exact (recall 1.0), so approximation
# would only add build cost.

# Les mêmes critères sous la politique livrée :
print(ar.route(ar.Criteria(n_vectors=50_000, dim=128)).backend)
# hnsw
```

Ou exporter `ANN_ROUTER_POLICY=/path/to/policy.yaml` pour remplacer les seuils
livrés, pour tout le processus. Partez de
[`ann_router/policy.yaml`](https://github.com/warith-harchaoui/ann-router/blob/main/ann_router/policy.yaml) :
chaque seuil y porte la note de calibrage qui l'a produit.

## Ligne de commande

Une seule commande click, cinq sous-commandes : `route`, `build`, `search`,
`bench`, `capabilities`.

```bash
ann-router route --n-vectors 2000000 --dim 768 --dynamic --markdown
```

```markdown
# ann-router decision: `hnsw`

**Rationale.** dynamic corpus, but target_recall=0.95 is at/above
HIGH_RECALL=0.9 — turbovec's calibrated benchmarks consistently undershoot that
recall (see policy.py's HIGH_RECALL docstring), so HNSW's better recall is used
despite the tombstone-delete downside on a churning corpus.

**Recommended config.** `{'metric': 'cosine', 'M': 16, 'ef_construction': 200, 'ef': 64}`

## Considered (priority order)

| backend | available | chosen | reason |
| --- | --- | --- | --- |
| `hnsw` | yes | **<-** | dynamic corpus, but target_recall=0.95 is at/above HIGH_RECALL=0.9 ... |

_policy version 1.2.0_
```

Construire un index à partir d'un fichier `.npy`, puis le chercher. `build`
écrit l'index plus un petit fichier JSON à côté (`my.idx.meta.json`) qui note
le moteur avec la géométrie, pour que `search` reconstruise le bon adaptateur
sans qu'on lui dise qui a écrit le fichier :

```bash
python -c "import numpy as np; np.save('c.npy', np.random.rand(20000,128).astype('float32'))"
ann-router build --n-vectors 20000 --dim 128 --vectors c.npy --index my.idx
# {
#   "backend": "hnsw",
#   "rationale": "stable in-memory corpus, high precision wanted: hnswlib gives the
#                 best recall/latency of the in-memory engines when the index rarely
#                 changes (note: it deletes only via tombstones, hence 'stable').",
#   "index_path": "my.idx",
#   "n": 20000,
#   "dim": 128,
#   "build_seconds": 1.4306
# }

python -c "import numpy as np; np.save('q.npy', np.random.rand(3,128).astype('float32'))"
ann-router search --index my.idx --queries q.npy -k 10
# {"ids": [[ ...10 identifiants... ], ...], "distances": [[0.17006641626358032, ...], ...]}
```

```bash
# Comparer les moteurs installés à la vérité terrain exacte (les chiffres du §4)
ann-router bench --n 5000 --dim 128 -k 10

# Matrice de disponibilité et de capacités, fusionnée avec le catalogue en prose
ann-router capabilities

# -V/--version comme -h/--help marchent, sur le groupe et sur chaque sous-commande
ann-router --version
# ann-router, version 0.3.0
```

Les portes HTTP et MCP appellent exactement les mêmes fonctions de cœur : elles
ne peuvent donc pas contredire la ligne de commande. `uvicorn
ann_router.api:app` sert `POST /route`, `GET /capabilities` et `GET /bench`,
pendant qu'`ann-router-mcp` monte ces trois-là comme outils MCP sur
`http://127.0.0.1:8019/mcp`.

## Dépannage

**`annoy` annonce un rappel de 0.0.** Ce n'est pas un défaut de ce paquet. Le
wheel PyPI d'`annoy` est compilé depuis les sources et, sur Apple Silicon, il
renvoie un seul voisin puis du remplissage `-1` quel que soit `k` : c'est
exactement le 0.0 du banc au §4. Remplacez-le par la version conda-forge :

```bash
pip uninstall -y annoy
conda install -c conda-forge python-annoy -y
```

`annoy` n'a par ailleurs aucune ABI numpy-2 : épinglez `numpy<2` dans tout
environnement qui l'utilise. Les deux mises en garde sont détaillées dans
[INSTALL.md](https://github.com/warith-harchaoui/ann-router/blob/main/INSTALL.md).

**`pgvector` est absent des résultats du banc.**
`PgVectorIndex.is_available()` vérifie seulement que `psycopg` avec `pgvector`
s'importent ; s'en servir réellement demande un PostgreSQL vivant. Sans DSN, il
lève `NotSupported: pgvector: no DSN — pass dsn=... or set ANN_ROUTER_PG_DSN to
a live Postgres` et le banc le saute. Lancez-en un avec
`docker compose --profile backends up`.

**`BackendUnavailable` sur un moteur attendu.** Le message porte le correctif,
par exemple `qdrant-client not installed. Run: pip install
'ann-router[qdrant]'`. Le routeur n'a pas besoin que vous agissiez : il
contourne le manque puis le dit dans sa justification (§1).

**Le routeur répond toujours `hnsw`.** C'est le défaut voulu pour un corpus
stable en mémoire. Lisez `choice.sensitivity` pour voir quel critère était le
plus près de changer le verdict ; une liste vide signifie qu'aucun ne le fait
basculer à 25 % près.

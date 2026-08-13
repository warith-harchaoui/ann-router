# Exemples ann-router

Un livre de recettes autonome. Chaque extrait s'exécute contre le paquet installé
(`pip install -e '.[all]'`) ; les sorties sont indiquées par des commentaires `# =>`.

## 1. Route : quel moteur, pourquoi ?

```python
import ann_router as ar

# target_recall abaissé sous HIGH_RECALL=0.9, le plafond calibré de turbovec
# (bench/results/calibrated_policy.yaml) -- sinon la politique l'écarte.
choice = ar.route(ar.Criteria(n_vectors=2_000_000, dim=768, dynamic=True, target_recall=0.85))
print(choice.backend)      # => turbovec
print(choice.rationale)    # => "corpus receives frequent updates: turbovec offers O(1) ..."
print([c["backend"] for c in choice.considered])   # => ['turbovec', 'hnsw']

# Au target_recall=0.95 par défaut (>= HIGH_RECALL), ce même corpus dynamique
# route vers hnsw à la place -- turbovec n'est même plus dans la liste, pas
# parce qu'il n'est pas installé, mais parce qu'il ne peut pas atteindre ce rappel.
choice = ar.route(ar.Criteria(n_vectors=2_000_000, dim=768, dynamic=True))
print(choice.backend)      # => hnsw
print([c["backend"] for c in choice.considered])   # => ['hnsw']
```

Le routeur ne renvoie jamais qu'un moteur **installé**. Si le premier choix de la
politique n'est pas installé, il se rabat sur un autre et le signale : sur une
machine où l'extra `turbovec` n'est pas installé, un corpus dynamique et à rappel
détendu se rabat par exemple sur `hnsw` :

```python
choice = ar.route(ar.Criteria(n_vectors=500_000, dim=768, dynamic=True, target_recall=0.85))
print(choice.backend)      # => hnsw   (si turbovec n'est pas installé ici)
print(choice.rationale)    # => "Preferred backend 'turbovec' is not installed here, so ..."
```

## 2. auto_index : router, construire et chercher en un seul appel

```python
import numpy as np, ann_router as ar

rng = np.random.default_rng(0)
vectors = rng.standard_normal((5_000, 128)).astype("float32")

index, choice = ar.auto_index(vectors, ar.Criteria(n_vectors=5_000, dim=128))
print(choice.backend)                      # => exact  (petit corpus)
ids, distances = index.search(vectors[:2], k=5)
print(ids.shape)                           # => (2, 5)
print(int(ids[0, 0]))                      # => 0  (le plus proche voisin d'un point est lui-même)
```

## 3. Piloter un moteur précis, directement

```python
from ann_router.backends.hnsw import HNSWIndex
import numpy as np

vecs = np.random.default_rng(1).standard_normal((10_000, 256)).astype("float32")
idx = HNSWIndex(dim=256, metric="cosine", M=16, ef=64).build(vecs)
ids, dist = idx.search(vecs[:1], k=10)
idx.save("/tmp/my.hnsw")
idx2 = HNSWIndex(dim=256).load("/tmp/my.hnsw")
```

## 4. Mesurer le rappel face à la vérité terrain exacte

```python
import numpy as np, ann_router as ar
from ann_router.backends.exact import ExactIndex
from ann_router.backends.turbovec_backend import TurboVecIndex

rng = np.random.default_rng(7)
corpus = rng.standard_normal((8_000, 128)).astype("float32")
q = rng.standard_normal((50, 128)).astype("float32")
truth, _ = ExactIndex(dim=128).build(corpus).search(q, 10)
pred, _ = TurboVecIndex(dim=128).build(corpus).search(q, 10)
recall = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(truth, pred)])
print(round(float(recall), 2))   # => ~0.85 (dépend des données ; TurboQuant est un procédé avec pertes)
```

Ou plus simplement, lancer le banc d'essai intégré sur chaque moteur installé :

```python
import ann_router as ar
from ann_router._core_cli import do_bench
print(do_bench(n=5000, dim=128, k=10)["results"])
# => {'exact': {'recall': 1.0, ...}, 'hnsw': {'recall': 1.0, ...}, 'turbovec': {...}, ...}
```

## 5. Recherche filtrée par métadonnées (Qdrant)

```python
import numpy as np
from ann_router.backends.qdrant_backend import QdrantIndex

vecs = np.random.default_rng(2).standard_normal((1_000, 64)).astype("float32")
payloads = [{"lang": "fr" if i % 2 else "en"} for i in range(1_000)]
idx = QdrantIndex(dim=64).build(vecs, payloads=payloads)

# plus proches voisins restreints aux documents en français :
ids, scores = idx.search_filter(vecs[:1], k=5, where={"lang": "fr"})
```

`pgvector` expose le même `search_filter` via un `WHERE payload->>...` SQL.

## 6. Capacités et disponibilité

```python
import ann_router as ar

print(ar.available_backends())
# => ['exact', 'turbovec', 'hnsw', 'faiss', 'annoy', 'qdrant', 'pgvector']

caps = ar.all_capabilities()
print(caps["annoy"].supports_remove)   # => False (figé après construction)
print(caps["qdrant"].supports_filter)  # => True
```

## 7. Gérer honnêtement les modes d'échec

```python
import numpy as np
from ann_router.backends.annoy_backend import AnnoyIndex
from ann_router.base import NotSupported

idx = AnnoyIndex(dim=8).build(np.random.rand(100, 8).astype("float32"))
try:
    idx.remove(np.array([1]))
except NotSupported as e:
    print(e)   # => "annoy: remove() unsupported (frozen index)"
```

## 8. Régler la politique sans toucher au code

```python
import ann_router as ar

# élève à 100k le seuil de bascule exact -> ANN, pour cet appel :
choice = ar.route(ar.Criteria(n_vectors=50_000, dim=128),
                  thresholds={"EXACT_MAX_N": 100_000})
print(choice.backend)   # => exact
```

Ou exporter `ANN_ROUTER_POLICY=/path/to/policy.yaml` pour remplacer les seuils
livrés, pour tout le processus.

## 9. En ligne de commande

```bash
# Décider, puis expliquer
ann-router route --n-vectors 2000000 --dim 768 --dynamic --markdown

# Construire un index à partir d'un fichier .npy, puis le chercher
python -c "import numpy as np; np.save('c.npy', np.random.rand(20000,128).astype('float32'))"
ann-router build --n-vectors 20000 --dim 128 --vectors c.npy --index my.idx
python -c "import numpy as np; np.save('q.npy', np.random.rand(3,128).astype('float32'))"
ann-router search --index my.idx --queries q.npy -k 10

# Comparer les moteurs installés à la vérité terrain exacte
ann-router bench --n 5000 --dim 128 -k 10
```

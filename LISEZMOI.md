# ann-router

[🇫🇷](LISEZMOI.md)&nbsp;&nbsp;|&nbsp;&nbsp;[🇬🇧](README.md)

![Licence](https://img.shields.io/badge/licence-BSD--3--Clause-blue)
![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)
![Local-first](https://img.shields.io/badge/local--first-oui-brightgreen)

`ann-router` fait partie de la suite **AI Helpers**. C'est un *routeur* :
vous décrivez votre problème de recherche de plus proches voisins approchés
(ANN) en termes *mesurés*, et il sélectionne, **justifie** et peut **instancier**
le bon moteur, au lieu de vous marier à une seule bibliothèque.

Rechercher les vecteurs les plus proches d'un vecteur de requête dans une base de nombreux vecteurs est un problème très courant en intelligence artificielle. Naïvement le problème a une complexité linéaire en le nombre de vecteurs dans la base. Souvent ce n'est pas acceptable donc on fait une recherche approchée pour avec une complexité beaucoup plus faible en le nombre de vecteurs dans la vase et raisonnable pour nos applications à millions voire milliards de vecteurs.

**C'est un composant indispensable pour les RAG.**

C'est le pendant « recherche vectorielle » de
[`best-engine-ai-helper`](https://github.com/warith-harchaoui/best-engine-ai-helper)
(qui choisit le meilleur LLM local pour une machine).

Même philosophie :
**mesurer les critères → choisir le moteur → renvoyer une justification
discutable.**

Les moteurs entre lesquels il arbitre :

+ naïf, *exact* (force brute)

+ turbovec

+ HNSW (hnswlib)

+ FAISS (IVF/PQ)

+ Annoy

+ ScaNN

+ Qdrant

+ pgvector

Importer le paquet est peu coûteux et sans dépendance : aucune dépendance
optionnelle de moteur n'est chargée à l'import, donc `import ann_router`
fonctionne avec numpy seul, et un backend dont la dépendance est absente se
signale simplement indisponible pendant que le routeur le contourne. Il s'agit donc de _lazy import_.

## Pourquoi router plutôt que choisir FAISS d'office (ou un autre) ?

Parce que le bon moteur est une *fonction du problème*, et le problème change :
un corpus de 5 000 vecteurs veut un balayage exact (instantané, rappel 1.0) ;
un corpus avec insertions/suppressions constantes veut turbovec (mutation
O(1)) ; un besoin de filtres SQL `WHERE` veut pgvector ; un corpus figé et
contraint en mémoire veut Annoy. Coder en dur une seule bibliothèque en réussit
un cas et rate les autres. Voir [PAYSAGE.md](PAYSAGE.md).

## Installation

Cœur (bibliothèque + CLI argparse toujours disponible) :

```bash
pip install -e ~/os-helper      # fondation de la suite
pip install -e .                # ou : pip install ann-router
```

Ajoutez les moteurs au besoin (extras par backend), ou tout d'un coup :

```bash
pip install 'ann-router[hnsw]'      # un moteur
pip install 'ann-router[all]'       # tous les moteurs installables + cli + api
```

Les instructions complètes et spécifiques à la plateforme — dont la mise en
garde **annoy sur Apple Silicon** et les notes **ScaNN**/**pgvector** — sont dans
[INSTALL.md](INSTALL.md).

## Démarrage rapide (bibliothèque)

```python
import numpy as np
import ann_router as ar

# 1. Décrire le problème en termes mesurés.
criteres = ar.Criteria(
    n_vectors=2_000_000, dim=768,
    dynamic=True,              # ajouts/suppressions fréquents
    target_recall=0.95,
    hardware=ar.detect_hardware(),
)

# 2. Demander quel backend — et pourquoi.
choix = ar.route(criteres)
print(choix.backend)          # 'turbovec'
print(choix.rationale)        # « le corpus reçoit des mises à jour fréquentes : ... »

# 3. Ou router + construire en un appel, puis chercher.
vecteurs = np.random.default_rng(0).standard_normal((5_000, 768)).astype("float32")
index, choix = ar.auto_index(vecteurs, ar.Criteria(n_vectors=5_000, dim=768))
ids, distances = index.search(vecteurs[:3], k=10)
```

Chaque backend parle la même interface `ANNIndex` :

```python
index.build(vecteurs, ids=None)
index.add(vecteurs); index.add_with_ids(vecteurs, ids); index.remove(ids)
ids, distances = index.search(requetes, k)
index.save(chemin); index.load(chemin)
Backend.capabilities()        # supports_remove / supports_filter / persistent / needs_gpu ...
```

Une opération qu'un backend ne peut réellement pas faire (ex. `Annoy.remove`)
lève un `NotSupported` clair ; un backend dont la dépendance manque lève un
`BackendUnavailable` avec la ligne `pip install` qui corrige.

## Les cinq portes (un cœur, cinq surfaces)

1. **Bibliothèque** — tout ce qui précède (`ann_router`).
2. **CLI** — `ann-router` (argparse, toujours disponible) et le jumeau
   `ann-router-click` (extra `[cli]`). Sous-commandes : `route`, `build`,
   `search`, `bench`, `capabilities`.
3. **API HTTP** — `uvicorn ann_router.api:app` (extra `[api]`) : `POST /route`,
   `GET /capabilities`, `GET /bench`.
4. **Serveur MCP** — `python -m ann_router.mcp_server` (extra `[mcp]`) : expose
   `route`, `capabilities`, `bench` comme outils d'agent.
5. **Skill** — `skills/ann-router/SKILL.md`, pour qu'un agent sache quand
   dégainer le routeur.

```bash
ann-router route --n-vectors 2000000 --dim 768 --dynamic --markdown
ann-router bench --n 5000 --dim 128 -k 10
ann-router capabilities
```

## Comment fonctionne la sélection

L'arbre de décision (réglable via `policy.yaml` / `ANN_ROUTER_POLICY`) :

| # | Si les critères disent… | Router vers | Parce que |
| - | ----------------------- | ----------- | --------- |
| 1 | `n < 10 000` | **exact** | un balayage force brute est déjà instantané et exact (rappel 1.0) |
| 2 | mises à jour fréquentes | **turbovec** | ajout/suppression O(1), sans reconstruction ; TurboQuant 2-4 bits (~16×) |
| 3 | très gros volume + GPU/batch | **FAISS** | IVF+PQ passe à l'échelle ; débit GPU par lots |
| 4 | persistance + filtres de métadonnées | **Qdrant / pgvector** | HNSW sur disque + filtrage payload/SQL `WHERE` |
| 5 | rappel maximal à l'échelle | **ScaNN** | quantification anisotrope (sensible au score) |
| 6 | lecture seule + mémoire serrée | **Annoy** | figé, mappé en mémoire, très léger |
| 7 | corpus stable en mémoire (défaut) | **HNSW** | meilleur rappel/latence quand l'index change rarement |

Le routeur renvoie non seulement le nom mais les **critères qui l'ont dicté** et
les **alternatives considérées** (y compris un moteur préféré mais non installé
dont il a dû se rabattre), pour que le choix soit auditable et surchargeable.

## Les critères (la spécification d'entrée)

`n_vectors`, `dim`, `target_recall`, `latency_budget_ms`, `memory_budget_gb`,
`dynamic`, `metadata_filtering`, `hardware` (`cpu`/`gpu`/`apple_silicon`,
auto-détectable), `persistence`, `batch_queries`, `metric`
(`cosine`/`l2`/`ip`). Seuls `n_vectors` et `dim` sont obligatoires.

## Pour aller plus loin

- [EXAMPLES.md](EXAMPLES.md) — un livre de recettes exécutable.
- [PAYSAGE.md](PAYSAGE.md) — comment ann-router se compare au choix d'un seul moteur.
- [CODING.md](CODING.md) — le standard de code que ce dépôt s'impose.
- [CONTRIBUTING.md](CONTRIBUTING.md) · [CHANGELOG.md](CHANGELOG.md) · [TRIGGERS.md](TRIGGERS.md)

## Auteur

[Warith HARCHAOUI](https://harchaoui.org/warith), Ph.D.

## Licence

BSD-3-Clause — voir [LICENSE](LICENSE).

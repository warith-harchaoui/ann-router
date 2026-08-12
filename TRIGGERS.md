# When to reach for ann-router

A discoverability catalogue (mirrors the sibling `best-engine-ai-helper`'s
`TRIGGERS.md`): the intents that should fire ann-router, in English and French,
plus what it does and does not do.

## What it does
Given a *measured* description of a vector-search problem, it selects the right
ANN backend (exact / turbovec / HNSW / FAISS / Annoy / Qdrant /
pgvector), explains why, and can build the index, all behind one interface.

## What it does NOT do
- It does not embed text/images (bring your own vectors).
- It does not serve a vector database or manage infrastructure.
- It does not replace FAISS/Qdrant/etc: it *orchestrates* them.

## Commands and how to invoke them

| Intent | CLI | Library |
| --- | --- | --- |
| Which backend for this problem? | `ann-router route ...` | `ann_router.route(Criteria(...))` |
| Route + build an index | `ann-router build ...` | `ann_router.auto_index(vectors, Criteria(...))` |
| Search a built index | `ann-router search ...` | `index.search(queries, k)` |
| Benchmark backends vs exact truth | `ann-router bench ...` | `ann_router._core_cli.do_bench(...)` |
| List backends + capabilities | `ann-router capabilities` | `ann_router.all_capabilities()` |

## Natural-language phrasings that should fire

**English**
- "Which ANN library / vector index should I use for N vectors?"
- "Should I use FAISS or HNSW or Annoy here?"
- "My corpus has constant inserts/deletes: what vector index handles that?"
- "I need metadata-filtered vector search."
- "Pick the right nearest-neighbour engine for this recall/latency/memory budget."
- "Is brute force enough or do I need an ANN index?"
- "Benchmark the recall of my vector backends."

**Français**
- « Quelle bibliothèque ANN / quel index vectoriel utiliser pour N vecteurs ? »
- « FAISS ou HNSW ou Annoy ici ? »
- « Mon corpus a des ajouts/suppressions constants, quel index gère ça ? »
- « J'ai besoin d'une recherche vectorielle filtrée par métadonnées. »
- « Choisis le bon moteur de plus proches voisins pour ce budget rappel/latence/mémoire. »
- « La force brute suffit-elle ou faut-il un index ANN ? »
- « Mesure le rappel de mes backends vectoriels. »

## Files read and written

| Reads | Writes |
| --- | --- |
| `ann_router/policy.yaml`, `backends.yaml`, `hardware.yaml`; `$ANN_ROUTER_POLICY` | index files + `*.meta.json` sidecars (via `build`) |

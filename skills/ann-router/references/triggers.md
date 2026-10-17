# ann-router skill — exhaustive trigger catalogue

Auditable superset of the phrasings that should fire this skill, in English and
French. The `description:` in `SKILL.md` is what a host model actually sees and
is capped at 1024 characters; this file carries the full list. Keep the two in
sync — and keep this file in sync with the repo-root `TRIGGERS.md`, the
user-facing catalogue it mirrors.

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

## When NOT to use ann-router (SKIP)

- **Embedding** text or images into vectors — bring your own; ann-router never
  calls a model.
- **Serving** a vector database, provisioning it, or operating it in
  production. It chooses and builds; it is not infrastructure.
- Replacing FAISS / Qdrant / HNSW — it *orchestrates* them, and a user who has
  already measured their way to one engine does not need a router.
- Picking a **local LLM or VLM** → `best-engine-ai-helper`, the sibling that
  answers the same shape of question for models.
- Deciding how many clusters or where a curve bends → `elbow-helper`.
- Plain similarity maths with no index in play.

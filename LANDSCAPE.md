# Landscape

[🇫🇷](PAYSAGE.md)&nbsp;&nbsp;|&nbsp;&nbsp;[🇬🇧](LANDSCAPE.md)

How `ann-router` compares to *just picking one engine*. Each tool is rated on
**this project's job, selecting the right ANN backend from measured criteria**,
not penalised for excelling at a different job (serving one engine well).

## Positioning

`ann-router` does not compete with FAISS, HNSW, Qdrant or the rest: it
**orchestrates** them. They solve *indexing and serving*; ann-router solves
*which one to use, and why*. The closest analogue is not another vector library
but its own sibling [best-engine-ai-helper](https://github.com/warith-harchaoui/best-engine-ai-helper), which routes over LLMs.

## At a glance

<!-- TABLE:START -->
| Fast Vector Search Tool | Measured selection | Justified rationale | Multi-engine | Handles churn | Metadata filter | Persistence | Recall-tested router | GPU acceleration | Vector compression | Distributed scaling | Managed cloud |
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

(A single engine scores ⭐ on "measured selection" because it *is* the selection:
there is nothing to decide. That is the point ann-router addresses.)

## Positioning map

<!-- FIGURE:START -->
2D representation of the table above.

![Positioning map](https://raw.githubusercontent.com/warith-harchaoui/ann-router/main/assets/landscape.png)

The map is a 2-D summary of the eleven criteria, so read it as a shape, not a scoreboard. `ann-router` is at the top-right corner. The axes read **Horizontal: Robustness ↔ Thoroughness** and **Vertical: Stability ↔ Precision**.

The map reveals a clear distinction between approaches prioritizing careful design and those emphasizing resilience. Options clustered on the left side of the map demonstrate a focus on robustness, with criteria like distributed scaling and handling churn playing a significant role in their positioning. Conversely, those on the right emphasize thoroughness, driven by factors such as measured selection and the inclusion of metadata filters. The reference option, ann-router, occupies the top-right corner, suggesting a balance of both thoroughness and precision, though it’s worth noting that several other options, including LangChain VectorStores, share this space, indicating a similar emphasis on detailed design and accurate results.

Several options find themselves grouped in the bottom-right corner, including Milvus, Weaviate, Vespa, Qdrant, LanceDB, and pgvectorscale. These systems prioritize stability and a comprehensive approach, but perhaps at the expense of the precision seen in the top-right quadrant. In contrast, FAISS, HNSW, Annoy, and turbovec all occupy the top-left corner, demonstrating a strong focus on robustness and precision, though they may not offer the same level of stability as the bottom-right group. Chroma sits alone in the bottom-left, highlighting a unique combination of robustness and stability that sets it apart from the other options. 

Ultimately, the map illustrates that different vector databases and tools make different trade-offs. While ann-router leads in overall performance, the other options demonstrate distinct strengths, and the best choice will depend on the specific priorities of the application. The clustering of several options in the right-bottom corner suggests a common design philosophy focused on comprehensive features and stability, while the concentration of options in the top-left corner indicates a shared emphasis on robustness and precision.
<!-- FIGURE:END -->

## Per-tool write-up

### FAISS
The scale king: IVF + PQ + GPU handle billions of vectors. But it is a *library*,
not a decision: its recall collapses on small or near-orthogonal corpora (the
`roitelet` study measured FAISS-HNSW recall ~0.47 at N=5k), it has no metadata
filtering, and choosing IVF vs Flat vs PQ and their parameters is exactly the
work ann-router automates. ann-router routes **to** FAISS for the very-large +
GPU/batch regime where it wins.

### HNSW (hnswlib)
Best recall/latency in memory for a **stable** corpus. Deletes are tombstone-only
(the graph degrades), so it is wrong for churny data; ann-router sends those to
turbovec instead, and reaches for HNSW as the high-precision in-memory default.

### Annoy
Frozen, memory-mapped, wonderfully lean for a **read-only** corpus under tight
memory. Cannot add or remove at all (ann-router surfaces this as `NotSupported`).
The right tool for exactly one regime, which ann-router detects.

### Qdrant / pgvector
The persistence + metadata-filter answer: on-disk HNSW plus payload/SQL `WHERE`
filtering. Heavier to operate than an in-memory index, so ann-router routes here
only when the criteria actually ask for filtering or durability, preferring
pgvector when a Postgres already exists.

### turbovec
The dynamic-corpus specialist: O(1) add/remove and 2-4 bit TurboQuant (~16×)
compression, strong on Apple Silicon. Recall is data-dependent (quantisation is
lossy), so ann-router picks it for the *frequent-updates* branch, not for
maximal-recall static workloads.

### pgvectorscale
The reason "just use Postgres" stopped being a concession. pgvectorscale adds a
DiskANN index and SBQ compression to pgvector, and the published benchmarks put
it at **471 QPS at 99% recall on 50M vectors where Qdrant managed 41** — an
11x gap at the recall end, from the engine most teams already run. It confirms
the premise rather than threatening it: the *index* inside an engine moves
recall and latency as much as the choice of engine, which is precisely the axis
ann-router has to reason about. Routing to pgvector now means asking which
index, not only which database.

### Milvus
The distributed answer: separated query/data/index nodes, GPU index builds
(CAGRA), and the one open engine with a real story past 100M vectors. The cost
is an operational surface — etcd, object storage, several node types — that
dwarfs an in-process index. ann-router routes here when scale genuinely exceeds
one machine, and says so in the rationale, because nobody should inherit that
cluster by accident.

### Weaviate
The hybrid-search specialist: BM25 and dense retrieval fused in a single query,
with metadata filters applied in the same pass. That matters because a
measurable share of "the vector search is bad" complaints are really lexical
misses on names, codes and rare tokens, which embeddings systematically blur.
It is the strongest argument in this table that *recall against a corpus* is
not the same quantity as *recall against a query*, and ann-router treats a
stated hybrid requirement as a routing input, not a nice-to-have.

### Vespa
The most capable and the least convenient: a full ranking stack with tensor
expressions, multi-phase ranking and genuine streaming over structured data,
designed for search engines rather than for an embedded index. Picked when the
job is ranking rather than nearest-neighbour lookup; ann-router does not
pretend to abstract it, since adopting Vespa is an architecture, not a backend.

### LanceDB
The embedded columnar bet: Lance format, zero-copy versioned storage, and an
in-process footprint that makes it the serverless-friendly end of on-disk
search. Closest to Annoy's niche but without the read-only tax, and strong
where a corpus has to live in object storage rather than in RAM.

### Chroma
The frictionless default: pip install, dict-like metadata, no server. Scale and
compression are not the point, and pretending otherwise is how teams end up
migrating under load — ann-router reads it as the prototype regime and says
when the corpus has outgrown it.

### LangChain VectorStores
A broad adapter layer over many stores, closest in spirit, but it unifies
*APIs*, not *decisions*: it will happily let you pick the wrong store. ann-router
adds the measured selection + justification + recall-tested policy on top.

## The thesis

Marrying one engine optimises for one shape of problem. Real systems change
shape: they grow, they start needing deletes, they add a filter requirement, they
move to a box with a GPU. ann-router keeps the *interface* fixed and lets the
*engine* follow the measured problem, with a rationale you can read and override.

The 2026 field strengthens that case rather than weakening it. The widest
measured gaps are no longer between engines but between *indexes inside one
engine* — DiskANN against HNSW against IVF against a GPU build swings recall
and latency further than swapping the database does — and the engines have
converged on offering several. A decision that used to be "which store" is now
"which store, which index, which quantisation", which is more routing to do,
not less.

## Sources and as-of date

The ratings above were last checked on **2026-10-23**. This field moves fast
enough that a stale table is a wrong table, so the date is part of the claim.

- [Open-source vector database benchmarks](https://aimultiple.com/open-source-vector-databases) — AIMultiple
- [Best vector databases in 2026: pricing, scale limits and architecture trade-offs](https://www.marktechpost.com/2026/05/10/best-vector-databases-in-2026-pricing-scale-limits-and-architecture-tradeoffs-across-nine-leading-systems/) — MarkTechPost
- [Best vector databases](https://encore.dev/articles/best-vector-databases) — Encore

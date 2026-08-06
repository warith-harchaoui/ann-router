# ann-router server image — the HTTP API door (see ann_router/api.py), fully
# featured: every pip-installable backend + the api/cli extras, so the served
# router can actually route to (and build) any of them, not just exact/turbovec.
#
# Build:  docker build -t ann-router .
# Run:    docker run --rm -p 8018:8018 ann-router
# Try:    curl -X POST localhost:8018/route -H 'content-type: application/json' \
#             -d '{"n_vectors": 500000, "dim": 768, "dynamic": true}'
#
# pgvector/qdrant servers are separate containers (see INSTALL.md); this image
# talks to them via ANN_ROUTER_PG_DSN / a Qdrant URL, it does not bundle them.

FROM python:3.11-slim

# hnswlib/faiss/annoy/turbovec occasionally need to build from source on a
# platform without a prebuilt wheel; build-essential keeps that path working
# without the image failing outright.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements.txt first (core deps only) so this layer caches independently
# of source changes; the full [all] extra (every pip-installable backend +
# api/cli) is installed once the source is in place, right below.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir '.[all]'

RUN useradd --create-home --uid 1000 ann-router
USER ann-router

EXPOSE 8018
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8018/capabilities', timeout=3)" || exit 1

CMD ["uvicorn", "ann_router.api:app", "--host", "0.0.0.0", "--port", "8018"]

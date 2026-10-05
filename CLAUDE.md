# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A GraphRAG proof of concept: documents are ingested into a Neo4j knowledge graph (Document/Entity nodes) and mirrored into a ChromaDB vector collection; a rule-based router sends each query to vector, graph, or hybrid (RRF) retrieval. A Streamlit UI (`demo_app.py`) wraps upload + query. Everything LLM-related goes through OpenAI (chat model for entity extraction, embedding model for ChromaDB); there are no local ML models. See `AGENTS.md` for style/commit conventions and `README.md` for the long-form walkthrough.

## Commands

### Local (no Docker)
```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
cp .env.sample .env            # fill in NEO4J_* , OPENAI_API_KEY, DEMO_USERNAME_1/DEMO_PASSWORD_1

streamlit run demo_app.py                                  # UI on :8501
python runtime_processor.py ./data/uploads                 # ingest a directory into Neo4j
python runtime_processor.py ./data/uploads --force-reprocess   # ignore the hash cache
python query_router_system.py --query "How is X related to Y?" # route + retrieve one query from the CLI
python query_router_system.py --batch                      # run the built-in demo query set
```
Run these from the repo root: `Config`/`RouterConfig` hardcode relative paths (`./cache`, `./chroma_db`, `./data`) and ignore the `DATA_DIR`/`CACHE_DIR`/`CHROMA_DB_PATH` env vars (only `demo_app.py` reads `DATA_DIR`).

### Docker (recommended path, what the README documents)
```bash
docker-compose build && docker-compose up -d
docker-compose ps
docker-compose logs -f graphrag-app      # service name; container name is graphrag-app
docker-compose logs -f neo4j

docker exec graphrag-app python runtime_processor.py /app/data/uploads [--force-reprocess]
docker exec graphrag-neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" "MATCH (n) RETURN labels(n), count(n)"
```
- Compose overrides `NEO4J_URI=bolt://neo4j:7687` on top of `.env`; the Neo4j password comes from `NEO4J_PASSWORD` with no default.
- `runtime_processor.py`, `data_ingestion_pipeline.py`, and `query_router_system.py` are bind-mounted read-only into the container, so edits to them take effect on `docker-compose restart`. `demo_app.py` and everything else are baked in and need `docker-compose build`.
- Ports 8501/7474/7687 are bound to 127.0.0.1 only (nginx config in `nginx-graphrag.conf` fronts it in production).

### Tests / sanity checks
There is no pytest suite. `tests/` holds standalone scripts that need a live Neo4j and a valid `.env`:
```bash
python tests/neo4j_connection_test.py   # RETURN 1 + node count
python tests/connection_test.py         # node/doc/entity counts
python tests/check_graph.py             # lists documents, entities, relationships
```
Inside Docker prefix with `docker exec graphrag-app`. Add new checks as self-contained scripts in `tests/`.

## Architecture

### Ingestion (`data_ingestion_pipeline.py`, driven by `runtime_processor.py`)
`LazyGraphBuilder.build_from_directory()` is the entry point:
1. `DataIngestionPipeline.ingest_directory()` loads files with LlamaIndex `SimpleDirectoryReader` (pdf/csv/txt/md/docx, recursive). Each file's MD5 is checked against `cache/file_hashes.json`; unchanged files are skipped. `--force-reprocess` just deletes that file.
2. Each document becomes one `ProcessedDocument` (`id = "doc_" + md5(text)[:12]`). **No chunking happens** despite `CHUNK_SIZE`/`CHUNK_OVERLAP` and the `Chunk` constraint in the schema; the whole text is stored on the `Document` node.
3. `extract_entities()` sends the first 2000 chars to `OPENAI_CHAT_MODEL` and parses a JSON array (max `MAX_ENTITIES_PER_CHUNK`=10). Failures are swallowed and return `[]`.
4. `extract_relationships()` is regex pattern matching over the text (IS_A, WORKS_FOR, FOUNDED, LOCATED_IN, OWNS), matched back to extracted entities.
5. `Neo4jGraphManager` MERGEs nodes/edges and creates constraints, indexes, and two full-text indexes (`entity_search`, `document_search`) on init.

Ingestion never writes to ChromaDB (see below).

### Neo4j schema (as actually written)
- `(:Document {id, title, content, source_path, document_type, content_hash, created_at, metadata})`
- `(:Entity {id, text, label, confidence, embedding})` with `id = "ent_" + md5(doc_id+text+label)[:12]`
- `(:Document)-[:CONTAINS]->(:Entity)`
- `(:Entity)-[:RELATED {type, confidence, context, document_id}]->(:Entity)` — relationship kind is the `type` **property**, not the edge label.

### ChromaDB sync model (`query_router_system.py`)
`VectorRetriever.__init__` opens `./chroma_db`, collection `documents`, with `OpenAIEmbeddingFunction(OPENAI_EMBEDDING_MODEL)`, then calls `_ensure_documents_indexed()`: it compares `count(:Document)` in Neo4j with the collection count and, on any mismatch, **drops and rebuilds the collection** from Neo4j (one embedding per whole document). So ChromaDB is a derived cache of Neo4j, refreshed whenever a `VectorRetriever` is constructed. `demo_app.py` relies on this by re-instantiating `VectorRetriever` after an upload; outside the UI, restarting the app (or constructing `HybridRetriever`) is the resync.

Changing `OPENAI_EMBEDDING_MODEL` makes the existing collection incompatible; delete `chroma_db/` and let it rebuild.

### Query routing (`query_router_system.py`)
- `QueryFeatureExtractor`: entities = capitalized words; complexity = token count (≤0.3) + entity count (≤0.2) + multi-hop (+0.25) + relationship words (+0.15) + reasoning words (+0.1), capped at 1.0.
- `AgenticQueryRouter._make_routing_decision()`, in order: VECTOR if complexity ≤ 0.3 and no multi-hop/relationship words; GRAPH if relationship words and ≥2 entities and complexity ≥ 0.4; HYBRID if complexity ≥ 0.6 or (multi-hop and >1 entity); else VECTOR at lower confidence.
- `GraphRetriever`: entity match via `(:Document)-[:CONTAINS]->(:Entity {text IN $entities})` → full-text `document_search` → plain `CONTAINS` substring search, each as fallback for the previous.
- `HybridRetriever.retrieve()` is the single entry point used by the UI and CLI; HYBRID runs vector (k*2, ≤20) and graph (k*2, ≤15) then `_reciprocal_rank_fusion()` with k=60, deduplicating on `source filename + first 100 chars`.
- Thresholds (`VECTOR_MAX_COMPLEXITY` etc.), `DEFAULT_K`, `CHUNK_SIZE`, `BATCH_SIZE` are dataclass defaults in `RouterConfig`/`Config`. They are **not** read from the environment even though `.env.example` and the README list them; change them in code.

### Streamlit app (`demo_app.py`)
- Auth: `DEMO_USERNAME_N`/`DEMO_PASSWORD_N` for N=1..9; if none are set it falls back to hardcoded demo credentials.
- `initialize_system()` is `@st.cache_resource` and builds one `HybridRetriever` per process.
- Upload saves to `$DATA_DIR/uploads` then runs `runtime_processor.py` as a subprocess (300 s timeout), then rebuilds `VectorRetriever`.

### Config / env
`Config` (ingestion) and `RouterConfig` (router) both `load_dotenv()` at import and raise `ValueError` at module import if `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, or `OPENAI_API_KEY` is missing. Importing either module therefore requires a configured `.env`. Model defaults: `OPENAI_EMBEDDING_MODEL=text-embedding-3-small`, `OPENAI_CHAT_MODEL=gpt-5-nano`. `.env.sample` is the current template; `.env.example` is stale (SentenceTransformer-era).

## Legacy scripts: read before running
- `fix_chromadb_sync.py` and `sync_databases.py` predate the OpenAI migration. They build the ChromaDB collection with `SentenceTransformerEmbeddingFunction`, which is no longer in `requirements.txt` and is incompatible with the OpenAI-embedded collection `VectorRetriever` expects. `fix_chromadb_sync.py` additionally rewrites `/app/query_router_system.py` and `/app/demo_app.py` in place. Prefer deleting `chroma_db/` and restarting (automatic resync) over running these.
- `kpi_tracker.py` is standalone and not imported anywhere.
- `data_ingestion_pipeline.py`'s `__main__` block downloads LlamaIndex demo datasets and clears the hash cache; it is a dev experiment, not the ingestion CLI.
- `runtime_processor.py` builds its JSON report from `config.EMBEDDING_MODEL`, which does not exist on `Config` (the attribute is `OPENAI_EMBEDDING_MODEL`). If you see processing "fail" right after the graph stats print, that is the cause; the graph writes have already succeeded.

## Conventions (from AGENTS.md)
- Python 3.10 (Docker image); PEP 8, type hints encouraged, imports ordered stdlib → third-party → local.
- Commits: short imperative subject ≤72 chars, one change per commit.
- Generated artifacts (`processing_report_*.json`, `graphrag_processing_*.log`, `cache/`, `chroma_db/`, `data/uploads/`) are gitignored; don't commit them.

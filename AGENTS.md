# Repository Guidelines

## Project Structure & Modules
- `demo_app.py`: Streamlit UI for querying and monitoring.
- `query_router_system.py`: Routing + retrieval (vector/graph/hybrid).
- `data_ingestion_pipeline.py`: Document parsing, entity/relationship extraction, Neo4j/Chroma sync.
- `runtime_processor.py`: Batch processing CLI.
- `cypher/`: Cypher helpers and query snippets.
- `tests/`: Connection and graph sanity scripts.
- `data/`, `logs/`, `cache/`: Runtime data and outputs.
- `docker-compose.yml`, `Dockerfile`, `.env.example`: Containerization and config.

## Build, Run, and Test
- Local setup: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Run UI: `streamlit run demo_app.py` (uses `.env`).
- Process a directory: `python runtime_processor.py ./data/sample`
- Docker (recommended): `docker-compose up -d` then access `http://localhost:8501`.
- Neo4j/Chroma sanity checks:
  - `python tests/neo4j_connection_test.py`
  - `python tests/connection_test.py`
  - `python tests/check_graph.py`

## Coding Style & Naming
- Python 3.9+: PEP 8, 4‑space indent.
- Names: `snake_case` for functions/vars, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Imports: standard → third‑party → local.
- Type hints encouraged; keep functions small and cohesive.

## Testing Guidelines
- Framework: lightweight scripts under `tests/` (no pytest config).
- Connection tests require valid `.env` (NEO4J_URI/USERNAME/PASSWORD).
- Add new checks as self‑contained scripts under `tests/` (e.g., `tests/my_feature_check.py`).
- Run via `python tests/<script>.py`; prefer deterministic outputs.

## Commit & PR Guidelines
- Commits: short, imperative subject (≤72 chars). Examples: “Fix vector retrieval scoring”, “Add Docker setup”.
- Scope one change per commit; reference issues with `#ID` when applicable.
- PRs: clear description, what/why, steps to test, linked issues, screenshots of Streamlit UI if UI changes, and note any env or schema changes.

## Security & Config
- Copy `.env.example` to `.env`; never commit secrets.
- Required env: `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`; optional: `OPENAI_API_KEY`.
- Volumes: local `./chroma_db`, `./data`, `./logs`; clean with caution.
- Before merging, confirm Neo4j and Chroma counts align (see README “Verification”).

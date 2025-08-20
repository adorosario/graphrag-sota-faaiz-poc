# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Docker Operations
```bash
# Build and start the system
docker-compose build
docker-compose up -d

# Check system status
docker-compose ps

# View logs
docker-compose logs -f app
docker-compose logs -f neo4j

# Restart services
docker-compose restart

# Stop system
docker-compose down
```

### Document Processing
```bash
# Process documents in a directory
docker exec graphrag-app python runtime_processor.py /app/data/uploads

# Force reprocess all documents (ignores cache)
docker exec graphrag-app python runtime_processor.py /app/data/uploads --force-reprocess

# Process sample data
docker exec graphrag-app python runtime_processor.py /app/data/sample
```

### Database Synchronization
```bash
# Fix ChromaDB sync issues (run when vector search fails)
docker exec graphrag-app python fix_chromadb_sync.py

# Manual database sync
docker exec graphrag-app python sync_databases.py
```

### Testing and Verification
```bash
# Test Neo4j connection
docker exec graphrag-app python tests/neo4j_connection_test.py

# Check database sync status
docker exec graphrag-app python tests/connection_test.py

# View graph structure
docker exec graphrag-neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" "MATCH (n) RETURN labels(n), count(n)"
```

## System Architecture

### Core Components

**GraphRAG** is an intelligent document processing system that combines Neo4j graph databases with ChromaDB vector storage for optimal retrieval performance.

#### Data Flow
1. **Documents** → `data_ingestion_pipeline.py` → **Entity Extraction** (spaCy) → **Neo4j Graph**
2. **Documents** → **Text Embedding** (SentenceTransformers) → **ChromaDB Vector Store**
3. **User Query** → `query_router_system.py` → **Route Decision** → **Optimal Retrieval Method**

#### Key Modules

**Data Ingestion** (`data_ingestion_pipeline.py`):
- `DataIngestionPipeline`: File hash tracking, multi-format reader (PDF, CSV, TXT, DOCX, MD)
- `Neo4jGraphManager`: Graph schema creation, batch processing, indexing
- Entity extraction using OpenAI GPT models (configurable)
- Relationship extraction via pattern matching

**Query Router** (`query_router_system.py`):
- `QueryFeatureExtractor`: Analyzes query complexity (8 features)
- `AgenticQueryRouter`: Routes to Vector (simple), Graph (relationships), or Hybrid (complex)
- `VectorRetriever`: ChromaDB semantic search
- `GraphRetriever`: Neo4j traversal with entity co-occurrence
- `HybridRetriever`: Reciprocal Rank Fusion (RRF) combining both methods

**Demo Interface** (`demo_app.py`):
- Streamlit web UI with authentication
- Document upload with automatic processing
- Real-time query routing visualization
- Performance analytics dashboard

**Runtime Processor** (`runtime_processor.py`):
- CLI tool for batch document processing
- Directory validation and progress tracking
- JSON report generation

### Database Schema

**Neo4j Nodes**:
- `Document`: Contains metadata, content, file_path, hash
- `Entity`: NER-extracted entities (PERSON, ORG, GPE, DATE)

**Neo4j Relationships**:
- `CONTAINS`: Document → Entity
- Entity relationship types: IS_A, WORKS_FOR, FOUNDED, LOCATED_IN, OWNS

**ChromaDB Collections**:
- `documents`: Synchronized with Neo4j Document nodes
- Embeddings: OpenAI text-embedding-3-small (configurable)

### Query Routing Logic

**Vector Route** (complexity < 0.3):
- Simple factual queries
- Uses ChromaDB cosine similarity search

**Graph Route** (≥2 entities, complexity ≥ 0.4):
- Relationship-focused queries
- Neo4j Cypher traversal with full-text search fallback

**Hybrid Route** (complexity ≥ 0.6):
- Complex multi-hop queries
- RRF fusion of vector + graph results (k=60)

## Environment Configuration

Required `.env` variables (Optimized for GraphRAG speed/cost):
```env
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-secure-password
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-5-nano
DEMO_USERNAME_1=admin
DEMO_PASSWORD_1=your-admin-password
```

## Common Issues and Solutions

### ChromaDB Sync Problems
**Symptoms**: Documents appear in Neo4j but not in search results
**Solution**: Run `docker exec graphrag-app python fix_chromadb_sync.py`

### Memory Issues
**Solution**: Adjust Docker Compose memory limits:
```yaml
deploy:
  resources:
    limits:
      memory: 4g
```

### Processing Failures
**Solution**: Clear cache and force reprocess:
```bash
docker exec graphrag-app rm -f /app/cache/file_hashes.json
docker exec graphrag-app python runtime_processor.py /app/data/uploads --force-reprocess
```

### OpenAI API Issues
**Symptoms**: Entity extraction or embedding failures
**Solution**: Check your OpenAI API key and model availability:
- Verify `OPENAI_API_KEY` is set correctly
- Ensure sufficient API credits
- Check model names are valid (text-embedding-3-small, gpt-4o-mini)

## File Structure

Key directories:
- `data/uploads/`: User-uploaded documents
- `chroma_db/`: ChromaDB persistent storage
- `cache/`: File hash cache for incremental processing
- `logs/`: Processing and deployment logs
- `cypher/`: Neo4j query examples and utilities
- `tests/`: Connection and system tests
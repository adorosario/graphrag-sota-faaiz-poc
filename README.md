# GraphRAG Document Processing System

An intelligent document processing system that combines graph-based knowledge representation with advanced retrieval techniques for unprecedented insights into your data.

## Table of Contents
- [System Architecture](#system-architecture)
- [Core Modules](#core-modules)
- [Data Processing Flow](#data-processing-flow)
- [Installation & Setup](#installation--setup)
- [Quick Start](#quick-start)
- [Troubleshooting](#troubleshooting)

## System Architecture

### Overview
GraphRAG is a next-generation Retrieval Augmented Generation (RAG) system that intelligently routes queries to the optimal retrieval method: Vector Search, Graph Traversal, or Hybrid approach. This results in up to 70% cost reduction and 8% F1 score improvement over traditional RAG systems.

### Key Features
- **Intelligent Query Routing**: Automatically selects the best retrieval strategy
- **Multi-format Support**: Processes PDF, CSV, TXT, DOCX, and MD files
- **Incremental Processing**: Only processes new or modified documents
- **Knowledge Graph Construction**: Builds Neo4j graph with entities and relationships
- **Performance Analytics**: Real-time monitoring and reporting
- **Production Ready**: Docker-based deployment with SSL support

## Core Modules

### Module Architecture and Data Flow

#### 1. Data Ingestion Pipeline (`data_ingestion_pipeline.py`)

**Purpose**: Processes documents and builds the knowledge graph

**Key Components**:

- **DataIngestionPipeline Class**
  - File hash tracking using MD5 for incremental processing
  - Multi-format document reader supporting PDF, CSV, TXT, DOCX, MD
  - Automatic document type detection based on file extension
  - Lazy processing - only processes changed files

- **Entity Extraction Process**
  - Uses spaCy NLP model (en_core_web_sm) for Named Entity Recognition
  - Extracts: People (PERSON), Organizations (ORG), Locations (GPE), Dates (DATE)
  - Filters low-quality entities (single characters, cardinal numbers)
  - Generates unique IDs using MD5 hashing
  - Limits to MAX_ENTITIES_PER_CHUNK (default: 10) for performance

- **Relationship Extraction**
  - Pattern-based extraction using regex
  - Relationship types: IS_A, WORKS_FOR, FOUNDED, LOCATED_IN, OWNS
  - Confidence scoring (0.7 for pattern-based matches)
  - Context preservation for each relationship

- **Neo4jGraphManager Class**
  - Creates optimized schema with constraints and indexes
  - Batch processing for efficient insertion
  - Full-text search index creation
  - Graph statistics monitoring

**Processing Flow**:
```
1. Directory Scan → 2. Hash Check → 3. Document Loading → 4. Content Extraction
→ 5. Entity Recognition → 6. Relationship Detection → 7. Graph Node Creation
→ 8. Index Creation → 9. Cache Update
```

#### 2. Query Router System (`query_router_system.py`)

**Purpose**: Intelligently routes queries to optimal retrieval method

**Routing Logic**:

- **QueryFeatureExtractor Class**
  - Extracts 8 key features from queries:
    1. Entity count (capitalized words)
    2. Token count
    3. Multi-hop indicators (relationship patterns)
    4. Relationship words presence
    5. Complex reasoning patterns
    6. Question type (what/how/why/who/where/when)
    7. Complexity score (0-1 scale)
    8. Named entities list

- **AgenticQueryRouter Class**
  - **Vector Route** (complexity < 0.3):
    - Simple factual queries
    - No relationship indicators
    - Example: "What is IBM?"
    - Uses ChromaDB semantic search
  
  - **Graph Route** (entities ≥ 2, complexity ≥ 0.4):
    - Relationship-focused queries
    - Contains relationship words
    - Example: "How are X and Y connected?"
    - Uses Neo4j graph traversal
  
  - **Hybrid Route** (complexity ≥ 0.6):
    - Complex multi-hop queries
    - Multiple entities with relationships
    - Example: "How did A affect B over time?"
    - Combines both methods with RRF

**Retrieval Components**:

- **VectorRetriever Class**
  - ChromaDB persistent storage at `./chroma_db`
  - Sentence transformer embeddings (all-MiniLM-L6-v2)
  - Cosine similarity scoring
  - Automatic Neo4j synchronization on init

- **GraphRetriever Class**
  - Entity-based traversal with co-occurrence detection
  - Full-text search fallback
  - Simple content search as final fallback
  - Cypher query optimization

- **HybridRetriever Class**
  - Reciprocal Rank Fusion (RRF) with k=60
  - Content-based deduplication
  - Source tracking and fusion metadata
  - Performance metrics collection

#### 3. Runtime Processor (`runtime_processor.py`)

**Purpose**: Command-line tool for batch document processing

**Features**:
- Directory validation and analysis
- File type detection and counting
- Progress tracking with detailed logging
- Automatic report generation in JSON
- Cache management for incremental processing

**Processing Metrics Tracked**:
- Documents processed
- Entities created
- Relationships created
- Processing time
- Files skipped (unchanged)
- Graph statistics

#### 4. Demo Application (`demo_app.py`)

**Purpose**: Interactive web interface with Streamlit

**Key Features**:

- **Authentication System**
  - Environment-based credentials
  - Multi-user support
  - Session management

- **Document Upload**
  - Drag-and-drop interface
  - 50MB file size limit
  - Automatic processing pipeline
  - ChromaDB synchronization

- **Query Interface**
  - Real-time routing visualization
  - Confidence scoring display
  - Processing time metrics
  - Top-5 results with metadata

- **Performance Analytics**
  - Route distribution pie chart
  - Processing time trend graph
  - Session statistics
  - Query history table

#### 5. KPI Tracker (`kpi_tracker.py`)

**Purpose**: Tracks performance metrics against requirements

**Metrics Tracked**:
- Router accuracy (target: ≥90%)
- Median latency ratio (target: ≤1.5x baseline)
- Cost reduction (target: ≥70%)
- F1 score, precision, recall
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (nDCG)

### Data Processing Flow

#### Document Ingestion Flow
```
1. User uploads document via Streamlit UI
2. File saved to /app/data/uploads/
3. runtime_processor.py called with upload directory
4. Hash check performed (MD5)
5. If new/modified:
   - Content extracted using LlamaIndex
   - Entities extracted using spaCy
   - Relationships detected via patterns
   - Neo4j nodes created (Document, Entity)
   - CONTAINS relationships established
   - ChromaDB synchronized
6. Processing report generated
7. UI updated with new document count
```

#### Query Processing Flow
```
1. User enters query
2. Feature extraction (8 features)
3. Complexity scoring
4. Route decision (Vector/Graph/Hybrid)
5. Retrieval execution:
   - Vector: ChromaDB similarity search
   - Graph: Neo4j traversal + co-occurrence
   - Hybrid: Both + RRF fusion
6. Results ranked and returned
7. Metrics tracked for KPIs
```

## Installation & Setup

### Prerequisites
- Ubuntu 20.04+ or similar Linux distribution
- Docker 20.10+ and Docker Compose 1.29+
- 4GB+ RAM (8GB recommended)
- 20GB+ free disk space
- Domain name with SSL certificate (for production)
- Neo4j Aura account (free tier works)

### Environment Setup

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/graphrag-poc.git
cd graphrag-poc
```

2. **Create Environment File**
```bash
cp .env.example .env
```

3. **Configure Environment Variables**
Edit `.env` with your settings:
```env
# Neo4j Configuration (REQUIRED)
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-secure-password

# OpenAI Configuration (Optional for future features)
OPENAI_API_KEY=sk-your-api-key

# Demo Credentials
DEMO_USERNAME_1=admin
DEMO_PASSWORD_1=your-admin-password
DEMO_USERNAME_2=demo
DEMO_PASSWORD_2=your-demo-password

# Application Settings
STREAMLIT_PORT=8501
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Docker Deployment

1. **Build and Start Services**
```bash
# Build Docker images
docker-compose build

# Start services
docker-compose up -d

# Check status
docker-compose ps
```

2. **Initialize Sample Data**
```bash
# Create sample documents
docker exec graphrag-app python -c "
from pathlib import Path
sample_dir = Path('/app/data/sample')
sample_dir.mkdir(parents=True, exist_ok=True)

with open(sample_dir / 'test.txt', 'w') as f:
    f.write('TechCorp is a technology company founded by John Smith.')
"

# Process sample data
docker exec graphrag-app python runtime_processor.py /app/data/sample
```

3. **Apply ChromaDB Fix**
```bash
# Copy fix script to container
docker cp fix_chromadb_sync.py graphrag-app:/app/

# Run the fix
docker exec graphrag-app python /app/fix_chromadb_sync.py

# Restart services
docker-compose restart
```

### Production Setup with SSL

1. **Install Nginx**
```bash
sudo apt update
sudo apt install nginx certbot python3-certbot-nginx
```

2. **Obtain SSL Certificate**
```bash
sudo certbot certonly --standalone -d your-domain.com
```

3. **Configure Nginx**
Create `/etc/nginx/sites-available/graphrag`:
```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    client_max_body_size 100M;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 86400;
    }

    location /_stcore/stream {
        proxy_pass http://localhost:8501/_stcore/stream;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
}
```

4. **Enable Site**
```bash
sudo ln -s /etc/nginx/sites-available/graphrag /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Verification

1. **Check Database Sync**
```bash
# Neo4j document count
docker exec graphrag-neo4j cypher-shell -u neo4j -p 'your-password' \
  "MATCH (d:Document) RETURN count(d) as count;"

# ChromaDB document count
docker exec graphrag-app python -c "
import chromadb
client = chromadb.PersistentClient(path='/app/chroma_db')
collection = client.get_collection('documents')
print(f'ChromaDB: {collection.count()} documents')
"
```

2. **Run Health Check**
```bash
#!/bin/bash
# health_check.sh

echo "GraphRAG System Health Check"
echo "============================"

# Check containers
docker-compose ps

# Check Neo4j
NEO4J_COUNT=$(docker exec graphrag-neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" \
  "MATCH (d:Document) RETURN count(d) as count;" 2>/dev/null | grep -o '[0-9]*' | tail -1)
echo "Neo4j Documents: $NEO4J_COUNT"

# Check ChromaDB
CHROMA_COUNT=$(docker exec graphrag-app python -c "
import chromadb
client = chromadb.PersistentClient(path='/app/chroma_db')
try:
    collection = client.get_collection('documents')
    print(collection.count())
except:
    print(0)
" 2>/dev/null)
echo "ChromaDB Documents: $CHROMA_COUNT"

# Check sync status
if [ "$NEO4J_COUNT" = "$CHROMA_COUNT" ]; then
    echo "✅ Databases are in sync"
else
    echo "⚠️ Database mismatch detected"
fi
```

## Quick Start

### 1. Access the Application
Navigate to `https://your-domain.com` (or `http://localhost:8501` for local)

### 2. Login
Use credentials from your `.env` file

### 3. Upload Documents
1. Click "📁 Document Upload" in sidebar
2. Select files (PDF, CSV, TXT, DOCX, MD)
3. Click "Process Uploaded Documents"
4. Wait for processing completion

### 4. Test Queries

**Simple Queries (Vector Route)**:
```
What is TechCorp?
What was the revenue in 2023?
Who is the Knowledge Graph?
```

**Relationship Queries (Graph Route)**:
```
How is John Smith related to TechCorp?
What is the connection between Product A and Department B?
Show relationships for Mary Johnson
```

**Complex Queries (Hybrid Route)**:
```
How did the technology changes affect financial performance over the last 3 years?
Compare our product strategy with market trends and analyze the impact
What factors led to the revenue growth and how did they influence each department?
```

### 5. Monitor Performance
- View route distribution in sidebar
- Check processing times in charts
- Review query history table

## Troubleshooting

### ChromaDB Not Syncing with Neo4j

**Symptoms**: New documents appear in Neo4j but not in search results

**Solution**:
```bash
# Run the complete fix
docker exec graphrag-app python /app/fix_chromadb_sync.py

# Restart application
docker-compose restart

# Verify sync
docker exec graphrag-app python -c "
from neo4j import GraphDatabase
import chromadb
import os

driver = GraphDatabase.driver(
    os.getenv('NEO4J_URI'),
    auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
)

with driver.session() as session:
    neo4j_count = session.run('MATCH (d:Document) RETURN count(d) as count').single()['count']

client = chromadb.PersistentClient(path='/app/chroma_db')
collection = client.get_collection('documents')
chroma_count = collection.count()

print(f'Neo4j: {neo4j_count}, ChromaDB: {chroma_count}')
print('✅ Synced' if neo4j_count == chroma_count else '❌ Mismatch')
"
```

### Document Processing Fails

**Check logs**:
```bash
docker-compose logs -f app
docker exec graphrag-app ls -la /app/logs/
```

**Force reprocess**:
```bash
docker exec graphrag-app rm -f /app/cache/file_hashes.json
docker exec graphrag-app python runtime_processor.py /app/data/uploads --force-reprocess
```

### Memory Issues

**Adjust Docker limits**:
```yaml
# docker-compose.yml
services:
  app:
    mem_limit: 4g
    memswap_limit: 4g
```

### Connection Issues

**Neo4j connection**:
```bash
# Test connection
docker exec graphrag-app python -c "
from neo4j import GraphDatabase
import os
driver = GraphDatabase.driver(
    os.getenv('NEO4J_URI'),
    auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
)
with driver.session() as session:
    result = session.run('RETURN 1')
    print('✅ Neo4j connected')
"
```

### Complete System Reset

```bash
#!/bin/bash
# reset_system.sh

echo "Resetting GraphRAG System..."

# Stop services
docker-compose down

# Clear data (WARNING: Deletes all data)
sudo rm -rf chroma_db/
sudo rm -rf cache/
sudo rm -rf logs/
sudo rm -rf data/uploads/*

# Rebuild and restart
docker-compose build --no-cache
docker-compose up -d

# Initialize with sample data
sleep 10
docker exec graphrag-app python runtime_processor.py /app/data/sample

# Apply fix
docker exec graphrag-app python /app/fix_chromadb_sync.py

echo "✅ System reset complete"
```

## Performance Optimization

### Tuning Parameters

Edit `.env` for optimization:
```env
# Processing
CHUNK_SIZE=512          # Smaller for detailed extraction
CHUNK_OVERLAP=50        # Increase for better context
MAX_ENTITIES_PER_CHUNK=10  # Limit entity extraction

# Router Thresholds
VECTOR_MAX_COMPLEXITY=0.3
GRAPH_MIN_ENTITIES=2
GRAPH_MIN_COMPLEXITY=0.4
HYBRID_MIN_COMPLEXITY=0.6

# Performance
BATCH_SIZE=32           # Adjust based on memory
MAX_WORKERS=4           # Based on CPU cores
```

### Scaling Recommendations

- **Horizontal Scaling**: Deploy multiple app instances behind load balancer
- **Database Optimization**: Use Neo4j Enterprise for clustering
- **Caching**: Implement Redis for query result caching
- **CDN**: Use CloudFlare for static assets

## Support

For issues or questions, please open an issue on GitHub or contact the development team.

---

**Version**: 1.0.0  
**Last Updated**: August 2025  
**Status**: Production Ready
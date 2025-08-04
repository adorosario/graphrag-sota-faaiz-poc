# GraphRAG POC - Data Ingestion Pipeline & Neo4j Schema Implementation

import os
import json
import logging
import asyncio
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import spacy
from sentence_transformers import SentenceTransformer
import chromadb
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.llama_dataset import download_llama_dataset
from llama_index.core.node_parser import SemanticSplitterNodeParser
from pydantic import BaseModel
import streamlit as st
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================


@dataclass
class Config:
    # Neo4j Configuration (Load from .env file)
    NEO4J_URI: str = os.getenv("NEO4J_URI")
    # Note: Using NEO4J_USERNAME to match your .env
    NEO4J_USER: str = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD")

    # LLM Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")

    # Embedding Model
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Processing Parameters
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    MAX_ENTITIES_PER_CHUNK: int = 10

    # Storage Paths
    DATA_DIR: str = "./data"
    CACHE_DIR: str = "./cache"

    # Performance Settings
    BATCH_SIZE: int = 32
    MAX_WORKERS: int = 4

    def __post_init__(self):
        """Validate required environment variables."""
        required_vars = {
            'NEO4J_URI': self.NEO4J_URI,
            'NEO4J_USERNAME': self.NEO4J_USER,
            'NEO4J_PASSWORD': self.NEO4J_PASSWORD,
            'OPENAI_API_KEY': self.OPENAI_API_KEY
        }

        missing_vars = [var for var,
                        value in required_vars.items() if not value]

        if missing_vars:
            logger.error(
                f"Missing required environment variables: {missing_vars}")
            logger.error(
                "Please create a .env file with the following variables:")
            for var in missing_vars:
                logger.error(f"  {var}=your_value_here")
            raise ValueError(
                f"Missing required environment variables: {missing_vars}")

        logger.info("✅ All required environment variables loaded successfully")


config = Config()

# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class ProcessedDocument:
    id: str
    title: str
    content: str
    source_path: str
    document_type: str
    created_at: datetime
    content_hash: str
    metadata: Dict[str, Any]


@dataclass
class Entity:
    id: str
    text: str
    label: str
    start_pos: int
    end_pos: int
    confidence: float
    embedding: Optional[List[float]] = None


@dataclass
class Relationship:
    source_entity: str
    target_entity: str
    relation_type: str
    confidence: float
    context: str
    document_id: str


@dataclass
class QueryRoute:
    route: str  # "vector", "graph", "hybrid"
    confidence: float
    reasoning: str
    features: Dict[str, Any]

# =============================================================================
# DATA INGESTION PIPELINE
# =============================================================================


class DataIngestionPipeline:
    """
    Handles ingestion of PDFs, CSVs, and text files with lazy processing.
    Only processes files that have changed since last run.
    """

    def __init__(self, config: Config):
        self.config = config
        self.cache_dir = Path(config.CACHE_DIR)
        self.cache_dir.mkdir(exist_ok=True)

        # Initialize embedding model
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)

        # Initialize NLP model for entity extraction
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except IOError:
            logger.warning(
                "spaCy model not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None

        # File hash cache for incremental processing
        self.file_hashes = self._load_file_hashes()

    def _load_file_hashes(self) -> Dict[str, str]:
        """Load cached file hashes to detect changes."""
        hash_file = self.cache_dir / "file_hashes.json"
        if hash_file.exists():
            with open(hash_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_file_hashes(self):
        """Save file hashes to cache."""
        hash_file = self.cache_dir / "file_hashes.json"
        with open(hash_file, 'w') as f:
            json.dump(self.file_hashes, f, indent=2)

    def _get_file_hash(self, file_path: str) -> str:
        """Generate hash for file to detect changes."""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _needs_processing(self, file_path: str) -> bool:
        """Check if file needs processing based on hash comparison."""
        current_hash = self._get_file_hash(file_path)
        cached_hash = self.file_hashes.get(file_path)

        if current_hash != cached_hash:
            self.file_hashes[file_path] = current_hash
            return True
        return False

    async def ingest_directory(self, directory_path: str) -> List[ProcessedDocument]:
        """
        Ingest all supported files from directory with lazy processing.
        """
        logger.info(f"Starting ingestion of directory: {directory_path}")

        # Use LlamaIndex SimpleDirectoryReader for broad file support
        reader = SimpleDirectoryReader(
            input_dir=directory_path,
            recursive=True,
            required_exts=[".pdf", ".csv", ".txt", ".md", ".docx"]
        )

        documents = reader.load_data()
        processed_docs = []

        for doc in documents:
            try:
                # Check if document needs processing
                source_path = doc.metadata.get('file_path', '')
                if source_path and not self._needs_processing(source_path):
                    logger.info(f"Skipping unchanged file: {source_path}")
                    continue

                processed_doc = await self._process_document(doc)
                processed_docs.append(processed_doc)

            except Exception as e:
                logger.error(f"Error processing document {doc.metadata}: {e}")
                continue

        self._save_file_hashes()
        logger.info(f"Processed {len(processed_docs)} documents")
        return processed_docs

    async def _process_document(self, doc: Document) -> ProcessedDocument:
        """Process individual document with semantic chunking."""

        # Generate document ID and hash
        content_hash = hashlib.md5(doc.text.encode()).hexdigest()
        doc_id = f"doc_{content_hash[:12]}"

        # Extract metadata
        source_path = doc.metadata.get('file_path', '')
        file_name = Path(source_path).name if source_path else 'unknown'

        # Determine document type
        doc_type = self._determine_document_type(source_path, doc.text)

        # Create processed document
        processed_doc = ProcessedDocument(
            id=doc_id,
            title=file_name,
            content=doc.text,
            source_path=source_path,
            document_type=doc_type,
            created_at=datetime.now(),
            content_hash=content_hash,
            metadata=doc.metadata
        )

        return processed_doc

    def _determine_document_type(self, file_path: str, content: str) -> str:
        """Determine document type based on file extension and content."""
        if not file_path:
            return "text"

        ext = Path(file_path).suffix.lower()
        type_mapping = {
            '.pdf': 'pdf',
            '.csv': 'csv',
            '.txt': 'text',
            '.md': 'markdown',
            '.docx': 'document'
        }
        return type_mapping.get(ext, 'unknown')

    def extract_entities(self, text: str, doc_id: str) -> List[Entity]:
        """Extract named entities from text using spaCy."""
        if not self.nlp:
            return []

        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            # Filter out low-quality entities
            if len(ent.text.strip()) < 2 or ent.label_ in ['CARDINAL', 'ORDINAL']:
                continue

            entity = Entity(
                id=f"ent_{hashlib.md5((doc_id + ent.text + ent.label_).encode()).hexdigest()[:12]}",
                text=ent.text.strip(),
                label=ent.label_,
                start_pos=ent.start_char,
                end_pos=ent.end_char,
                confidence=1.0  # spaCy doesn't provide confidence scores directly
            )
            entities.append(entity)

        return entities[:self.config.MAX_ENTITIES_PER_CHUNK]

    def extract_relationships(self, text: str, entities: List[Entity], doc_id: str) -> List[Relationship]:
        """Extract relationships between entities using pattern matching."""
        relationships = []

        # Simple pattern-based relationship extraction
        relation_patterns = [
            (r'\b(\w+)\s+(?:is|was|are|were)\s+(?:a|an|the)?\s*(\w+)', 'IS_A'),
            (r'\b(\w+)\s+(?:works for|employed by|part of)\s+(\w+)', 'WORKS_FOR'),
            (r'\b(\w+)\s+(?:founded|created|established)\s+(\w+)', 'FOUNDED'),
            (r'\b(\w+)\s+(?:located in|based in|from)\s+(\w+)', 'LOCATED_IN'),
            (r'\b(\w+)\s+(?:owns|has|possesses)\s+(\w+)', 'OWNS'),
        ]

        import re

        for pattern, relation_type in relation_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                source_text, target_text = match.groups()

                # Find matching entities
                source_entities = [
                    e for e in entities if e.text.lower() in source_text.lower()]
                target_entities = [
                    e for e in entities if e.text.lower() in target_text.lower()]

                for source_ent in source_entities:
                    for target_ent in target_entities:
                        if source_ent.id != target_ent.id:
                            relationship = Relationship(
                                source_entity=source_ent.id,
                                target_entity=target_ent.id,
                                relation_type=relation_type,
                                confidence=0.7,  # Pattern-based confidence
                                context=match.group(0),
                                document_id=doc_id
                            )
                            relationships.append(relationship)

        return relationships

# =============================================================================
# NEO4J SCHEMA & GRAPH CONSTRUCTION
# =============================================================================


class Neo4jGraphManager:
    """
    Manages Neo4j graph database operations with optimized schema.
    """

    def __init__(self, config: Config):
        self.config = config
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )
        self._initialize_schema()

    def close(self):
        """Close database connection."""
        self.driver.close()

    def _initialize_schema(self):
        """Create optimized Neo4j schema with constraints and indexes."""

        schema_queries = [
            # Create constraints for unique IDs
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",

            # Create indexes for performance
            "CREATE INDEX entity_text IF NOT EXISTS FOR (e:Entity) ON (e.text)",
            "CREATE INDEX entity_label IF NOT EXISTS FOR (e:Entity) ON (e.label)",
            "CREATE INDEX document_type IF NOT EXISTS FOR (d:Document) ON (d.document_type)",
            "CREATE INDEX chunk_embedding IF NOT EXISTS FOR (c:Chunk) ON (c.embedding)",

            # Create full-text search indexes
            "CREATE FULLTEXT INDEX entity_search IF NOT EXISTS FOR (e:Entity) ON EACH [e.text]",
            "CREATE FULLTEXT INDEX document_search IF NOT EXISTS FOR (d:Document) ON EACH [d.title, d.content]",
        ]

        with self.driver.session() as session:
            for query in schema_queries:
                try:
                    session.run(query)
                    logger.info(f"Executed schema query: {query}")
                except Exception as e:
                    logger.warning(
                        f"Schema query failed (may already exist): {e}")

    def create_document_node(self, doc: ProcessedDocument) -> str:
        """Create or update document node in Neo4j."""

        query = """
        MERGE (d:Document {id: $doc_id})
        SET d.title = $title,
            d.content = $content,
            d.source_path = $source_path,
            d.document_type = $document_type,
            d.created_at = $created_at,
            d.content_hash = $content_hash,
            d.metadata = $metadata
        RETURN d.id as id
        """

        with self.driver.session() as session:
            result = session.run(query,
                                 doc_id=doc.id,
                                 title=doc.title,
                                 content=doc.content,
                                 source_path=doc.source_path,
                                 document_type=doc.document_type,
                                 created_at=doc.created_at.isoformat(),
                                 content_hash=doc.content_hash,
                                 metadata=json.dumps(doc.metadata)
                                 )
            return result.single()["id"]

    def create_entity_nodes(self, entities: List[Entity], doc_id: str):
        """Batch create entity nodes with relationships to document."""

        if not entities:
            return

        # Create entities in batch - convert dataclass to dict manually
        entity_query = """
        UNWIND $entities as entity
        MERGE (e:Entity {id: entity.id})
        SET e.text = entity.text,
            e.label = entity.label,
            e.confidence = entity.confidence,
            e.embedding = entity.embedding
        """

        # Create relationships to document
        doc_relation_query = """
        UNWIND $entity_ids as entity_id
        MATCH (d:Document {id: $doc_id})
        MATCH (e:Entity {id: entity_id})
        MERGE (d)-[:CONTAINS]->(e)
        """

        # Convert dataclasses to dictionaries manually
        entity_data = []
        for entity in entities:
            entity_data.append({
                'id': entity.id,
                'text': entity.text,
                'label': entity.label,
                'confidence': entity.confidence,
                'embedding': entity.embedding
            })

        entity_ids = [entity.id for entity in entities]

        with self.driver.session() as session:
            session.run(entity_query, entities=entity_data)
            session.run(doc_relation_query,
                        entity_ids=entity_ids, doc_id=doc_id)

    def create_relationships(self, relationships: List[Relationship]):
        """Create relationships between entities."""

        if not relationships:
            return

        query = """
        UNWIND $relationships as rel
        MATCH (source:Entity {id: rel.source_entity})
        MATCH (target:Entity {id: rel.target_entity})
        MERGE (source)-[r:RELATED {type: rel.relation_type}]->(target)
        SET r.confidence = rel.confidence,
            r.context = rel.context,
            r.document_id = rel.document_id
        """

        # Convert dataclasses to dictionaries manually
        relationship_data = []
        for rel in relationships:
            relationship_data.append({
                'source_entity': rel.source_entity,
                'target_entity': rel.target_entity,
                'relation_type': rel.relation_type,
                'confidence': rel.confidence,
                'context': rel.context,
                'document_id': rel.document_id
            })

        with self.driver.session() as session:
            session.run(query, relationships=relationship_data)

    def get_graph_stats(self) -> Dict[str, int]:
        """Get graph statistics for monitoring."""

        stats_query = """
        MATCH (d:Document) 
        WITH count(d) as docs
        MATCH (e:Entity) 
        WITH docs, count(e) as entities
        MATCH ()-[r:RELATED]->() 
        WITH docs, entities, count(r) as relationships
        RETURN docs, entities, relationships
        """

        with self.driver.session() as session:
            result = session.run(stats_query)
            record = result.single()

            # Handle case where query returns None (empty database)
            if record is None:
                return {
                    "documents": 0,
                    "entities": 0,
                    "relationships": 0
                }

            return {
                "documents": record["docs"] or 0,
                "entities": record["entities"] or 0,
                "relationships": record["relationships"] or 0
            }

# =============================================================================
# LAZY GRAPH BUILDER
# =============================================================================


class LazyGraphBuilder:
    """
    Incrementally builds knowledge graph with cost optimization.
    """

    def __init__(self, config: Config):
        self.config = config
        self.ingestion_pipeline = DataIngestionPipeline(config)
        self.graph_manager = Neo4jGraphManager(config)
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)

    async def build_from_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        Build graph incrementally from directory with cost tracking.
        """
        start_time = datetime.now()
        logger.info(f"Starting lazy graph build from: {directory_path}")

        # Track costs and performance
        metrics = {
            "documents_processed": 0,
            "entities_created": 0,
            "relationships_created": 0,
            "processing_time": 0,
            "files_skipped": 0
        }

        try:
            # Ingest documents (only processes changed files)
            documents = await self.ingestion_pipeline.ingest_directory(directory_path)
            metrics["documents_processed"] = len(documents)

            # Process each document
            for doc in documents:
                await self._process_document(doc, metrics)

            # Calculate final metrics
            end_time = datetime.now()
            metrics["processing_time"] = (
                end_time - start_time).total_seconds()

            # Get graph statistics
            graph_stats = self.graph_manager.get_graph_stats()
            metrics.update(graph_stats)

            logger.info(f"Graph build completed: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Error in graph building: {e}")
            raise

    async def _process_document(self, doc: ProcessedDocument, metrics: Dict[str, Any]):
        """Process individual document and update graph."""

        # Create document node
        self.graph_manager.create_document_node(doc)

        # Extract entities
        entities = self.ingestion_pipeline.extract_entities(
            doc.content, doc.id)

        # Generate embeddings for entities
        if entities:
            entity_texts = [entity.text for entity in entities]
            embeddings = self.embedding_model.encode(entity_texts).tolist()

            for entity, embedding in zip(entities, embeddings):
                entity.embedding = embedding

        # Create entity nodes
        self.graph_manager.create_entity_nodes(entities, doc.id)
        metrics["entities_created"] += len(entities)

        # Extract and create relationships
        relationships = self.ingestion_pipeline.extract_relationships(
            doc.content, entities, doc.id
        )

        self.graph_manager.create_relationships(relationships)
        metrics["relationships_created"] += len(relationships)

        logger.info(
            f"Processed {doc.title}: {len(entities)} entities, {len(relationships)} relationships")

# =============================================================================
# DEMO DATASET LOADER
# =============================================================================


class DemoDatasetLoader:
    """
    Load and prepare LlamaIndex datasets for evaluation.
    """

    @staticmethod
    def load_llama_datasets() -> List[Tuple[str, Any, Any]]:
        """Load multiple LlamaIndex datasets for comprehensive testing."""

        datasets_to_load = [
            "PaulGrahamEssayDataset",
            # Add more datasets as available
        ]

        loaded_datasets = []

        for dataset_name in datasets_to_load:
            try:
                logger.info(f"Loading dataset: {dataset_name}")
                rag_dataset, documents = download_llama_dataset(
                    dataset_name, f"./datasets/{dataset_name}"
                )
                loaded_datasets.append((dataset_name, rag_dataset, documents))
                logger.info(f"Successfully loaded {dataset_name}")

            except Exception as e:
                logger.error(f"Failed to load {dataset_name}: {e}")

        return loaded_datasets

    @staticmethod
    def prepare_sample_data() -> str:
        """Prepare sample data directory for testing."""

        sample_dir = Path("./sample_data")
        sample_dir.mkdir(exist_ok=True)

        # Create sample documents for testing
        sample_files = {
            "company_overview.txt": """
            TechCorp Inc. is a leading technology company founded in 2010 by John Smith and Mary Johnson.
            The company is headquartered in San Francisco, California and specializes in artificial intelligence
            and machine learning solutions. TechCorp has offices in New York, London, and Tokyo.
            
            Key personnel:
            - John Smith: CEO and Co-founder
            - Mary Johnson: CTO and Co-founder  
            - Robert Davis: Head of Engineering
            - Sarah Wilson: VP of Marketing
            
            The company has raised $50 million in Series B funding led by Venture Capital Partners.
            TechCorp's main products include AI Platform, DataAnalyzer Pro, and SmartInsights.
            """,

            "financial_report.txt": """
            TechCorp Q4 2024 Financial Report
            
            Revenue: $25 million (up 150% YoY)
            Gross Profit: $18 million (72% margin)
            Net Income: $5 million
            
            Key Metrics:
            - Monthly Recurring Revenue: $3.2 million
            - Customer Acquisition Cost: $1,200
            - Customer Lifetime Value: $15,000
            
            The company's AI Platform generated 60% of total revenue, while DataAnalyzer Pro
            contributed 30% and SmartInsights 10%. TechCorp expanded its customer base to
            over 500 enterprise clients including Microsoft, Google, and Amazon.
            """,

            "research_paper.txt": """
            Title: Advanced Graph Neural Networks for Knowledge Representation
            Authors: Dr. Alice Chen, Prof. Michael Brown
            
            Abstract:
            This paper presents a novel approach to knowledge representation using advanced
            Graph Neural Networks (GNNs). Our method combines entity linking with relation
            extraction to create more accurate knowledge graphs.
            
            The research was conducted at Stanford University in collaboration with MIT.
            We evaluated our approach on the NELL dataset and achieved 15% improvement
            over baseline methods. The work was funded by NSF Grant #12345.
            
            Keywords: Knowledge Graphs, Graph Neural Networks, Entity Linking, Relation Extraction
            """
        }

        for filename, content in sample_files.items():
            with open(sample_dir / filename, 'w') as f:
                f.write(content.strip())

        logger.info(f"Created sample data in {sample_dir}")
        return str(sample_dir)

# =============================================================================
# MAIN EXECUTION & TESTING
# =============================================================================


if __name__ == "__main__":
    async def main():
        """Main execution for testing the pipeline."""

        # Initialize configuration
        config = Config()

        # Test with sample data first
        logger.info("Setting up sample data for testing...")
        sample_dir = DemoDatasetLoader.prepare_sample_data()

        # Initialize lazy graph builder
        builder = LazyGraphBuilder(config)

        try:
            # Clear cache to force reprocessing for demo
            cache_file = Path(
                builder.ingestion_pipeline.cache_dir) / "file_hashes.json"
            if cache_file.exists():
                cache_file.unlink()
                logger.info("🔄 Cleared cache to force reprocessing")

            # Build graph from sample data
            logger.info("Building graph from sample data...")
            metrics = await builder.build_from_directory(sample_dir)

            print("\n" + "="*50)
            print("GRAPH BUILD RESULTS")
            print("="*50)
            for key, value in metrics.items():
                print(f"{key}: {value}")

            # Test with LlamaIndex datasets (optional)
            logger.info("Loading LlamaIndex datasets...")
            try:
                datasets = DemoDatasetLoader.load_llama_datasets()

                if datasets:
                    dataset_name, rag_dataset, documents = datasets[0]
                    logger.info(
                        f"Processing {dataset_name} with {len(documents)} documents")

                    # Save documents to temporary directory
                    temp_dir = Path(f"./temp_{dataset_name}")
                    temp_dir.mkdir(exist_ok=True)

                    for i, doc in enumerate(documents):
                        with open(temp_dir / f"doc_{i}.txt", 'w') as f:
                            f.write(doc.text)

                    # Build graph from dataset
                    dataset_metrics = await builder.build_from_directory(str(temp_dir))

                    print("\n" + "="*50)
                    print(f"DATASET BUILD RESULTS - {dataset_name}")
                    print("="*50)
                    for key, value in dataset_metrics.items():
                        print(f"{key}: {value}")

            except Exception as e:
                logger.warning(
                    f"LlamaIndex dataset loading failed (optional): {e}")
                logger.info("Continuing with sample data only...")

        finally:
            builder.graph_manager.close()

    # Run the test
    asyncio.run(main())
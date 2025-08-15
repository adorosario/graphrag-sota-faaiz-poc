#!/usr/bin/env python3
"""
Database synchronization utility
Ensures ChromaDB stays in sync with Neo4j
"""

import os
import logging
from neo4j import GraphDatabase
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sync_chromadb_with_neo4j():
    """Force sync ChromaDB with Neo4j"""
    
    # Configuration
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Connect to Neo4j
    driver = GraphDatabase.driver(
        neo4j_uri,
        auth=(neo4j_user, neo4j_password)
    )
    
    documents = []
    metadatas = []
    ids = []
    
    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (d:Document)
                RETURN d.id as id, 
                       d.content as content, 
                       d.title as title,
                       d.source_path as source_path,
                       d.document_type as document_type
            """)
            
            for record in result:
                if record['content'] and len(record['content'].strip()) > 10:
                    documents.append(record['content'])
                    metadatas.append({
                        'title': record['title'] or 'Unknown',
                        'source_path': record['source_path'] or 'unknown',
                        'document_type': record['document_type'] or 'text'
                    })
                    ids.append(record['id'])
            
            logger.info(f"Found {len(documents)} documents in Neo4j")
    finally:
        driver.close()
    
    if not documents:
        logger.warning("No documents found in Neo4j")
        return 0
    
    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # Delete existing collection
    try:
        chroma_client.delete_collection("documents")
        logger.info("Deleted existing ChromaDB collection")
    except:
        pass
    
    # Create new collection
    collection = chroma_client.create_collection(
        name="documents",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
    )
    
    # Add documents in batches
    batch_size = 10
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        collection.add(
            documents=batch_docs,
            metadatas=batch_metas,
            ids=batch_ids
        )
    
    final_count = collection.count()
    logger.info(f"✅ ChromaDB sync complete: {final_count} documents")
    return final_count

if __name__ == "__main__":
    count = sync_chromadb_with_neo4j()
    print(f"Synced {count} documents")
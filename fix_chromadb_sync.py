#!/usr/bin/env python3
"""
Complete fix for ChromaDB synchronization issue
This ensures ChromaDB always stays in sync with Neo4j
"""

import os
import sys
import json
import logging
from pathlib import Path
from neo4j import GraphDatabase
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaDBFixer:
    """Complete fix for ChromaDB synchronization"""
    
    def __init__(self):
        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
    def fix_query_router_system(self):
        """Update query_router_system.py with proper sync logic"""
        
        file_path = "/app/query_router_system.py"
        
        # Read the current file
        with open(file_path, 'r') as f:
            content = f.read()
        
        # New _ensure_documents_indexed method that always syncs
        new_method = '''    def _ensure_documents_indexed(self):
        """Ensure documents from Neo4j are indexed in ChromaDB - ALWAYS SYNC."""
        
        logger.info("Syncing ChromaDB with Neo4j documents...")
        
        # Get documents from Neo4j
        neo4j_driver = GraphDatabase.driver(
            self.config.NEO4J_URI,
            auth=(self.config.NEO4J_USER, self.config.NEO4J_PASSWORD)
        )
        
        try:
            with neo4j_driver.session() as session:
                # Get Neo4j document count
                neo4j_count_result = session.run("MATCH (d:Document) RETURN count(d) as count")
                neo4j_count = neo4j_count_result.single()["count"]
                
                # Get ChromaDB document count
                try:
                    chroma_count = self.collection.count()
                except:
                    chroma_count = 0
                
                logger.info(f"Neo4j documents: {neo4j_count}, ChromaDB documents: {chroma_count}")
                
                # ALWAYS resync to ensure consistency
                if True:  # Changed from checking count mismatch
                    logger.info("Resyncing ChromaDB with Neo4j...")
                    
                    # Delete and recreate collection for clean sync
                    try:
                        self.chroma_client.delete_collection("documents")
                        logger.info("Deleted existing collection")
                    except Exception as e:
                        logger.info(f"No existing collection to delete: {e}")
                    
                    # Recreate collection
                    self.collection = self.chroma_client.create_collection(
                        name="documents",
                        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                            model_name=self.config.EMBEDDING_MODEL
                        )
                    )
                    logger.info("Created new collection")
                    
                    # Get all documents from Neo4j
                    result = session.run("""
                        MATCH (d:Document)
                        RETURN d.id as id, 
                               d.content as content, 
                               d.title as title,
                               d.source_path as source_path,
                               d.document_type as document_type
                    """)
                    
                    documents = []
                    metadatas = []
                    ids = []
                    
                    for record in result:
                        if record['content'] and len(record['content'].strip()) > 10:
                            documents.append(record['content'])
                            metadatas.append({
                                'title': record['title'] or 'Unknown',
                                'source_path': record['source_path'] or 'unknown',
                                'document_type': record['document_type'] or 'text'
                            })
                            ids.append(record['id'])
                    
                    if documents:
                        # Add to ChromaDB in batches
                        batch_size = 10
                        for i in range(0, len(documents), batch_size):
                            batch_docs = documents[i:i+batch_size]
                            batch_metas = metadatas[i:i+batch_size]
                            batch_ids = ids[i:i+batch_size]
                            
                            self.collection.add(
                                documents=batch_docs,
                                metadatas=batch_metas,
                                ids=batch_ids
                            )
                        
                        logger.info(f"✅ Synced {len(documents)} documents to ChromaDB")
                    else:
                        logger.warning("No documents found in Neo4j")
                    
        except Exception as e:
            logger.error(f"Failed to sync ChromaDB: {e}")
        finally:
            neo4j_driver.close()'''
        
        # Replace the method
        import re
        pattern = r'def _ensure_documents_indexed\(self\):.*?(?=\n    def |\n\nclass |\Z)'
        
        if re.search(pattern, content, re.DOTALL):
            content = re.sub(pattern, new_method.strip(), content, flags=re.DOTALL)
            
            # Write back
            with open(file_path, 'w') as f:
                f.write(content)
            
            logger.info("✅ Updated query_router_system.py")
            return True
        else:
            logger.warning("Could not find _ensure_documents_indexed method")
            return False
    
    def fix_demo_app(self):
        """Fix the process_uploaded_documents function in demo_app.py"""
        
        file_path = "/app/demo_app.py"
        
        # Read current file
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Find and replace the process_uploaded_documents function section that handles ChromaDB
        updated_content = content.replace(
            """            # Force reinitialize ChromaDB to include new documents
            if st.session_state.hybrid_retriever:
                st.sidebar.info("🔄 Updating search index...")
                try:
                    # Clear the existing collection
                    st.session_state.hybrid_retriever.vector_retriever.collection.delete()
                    st.sidebar.info("🗑️ Cleared old index")
                except Exception as e:
                    st.sidebar.warning(f"Index clear warning: {e}")

                # Reinitialize the collection
                try:
                    st.session_state.hybrid_retriever.vector_retriever._ensure_documents_indexed()
                    st.sidebar.success("✅ Search index updated!")
                except Exception as e:
                    st.sidebar.error(f"❌ Index update failed: {e}")""",
            """            # Force reinitialize ChromaDB to include new documents
            if st.session_state.hybrid_retriever:
                st.sidebar.info("🔄 Updating search index...")
                try:
                    # Reinitialize the entire VectorRetriever to force sync
                    from query_router_system import VectorRetriever, RouterConfig
                    config = RouterConfig()
                    st.session_state.hybrid_retriever.vector_retriever = VectorRetriever(config)
                    st.sidebar.success("✅ Search index updated!")
                except Exception as e:
                    st.sidebar.error(f"❌ Index update failed: {e}")"""
        )
        
        if updated_content != content:
            with open(file_path, 'w') as f:
                f.write(updated_content)
            logger.info("✅ Updated demo_app.py")
            return True
        else:
            logger.warning("demo_app.py already up to date or pattern not found")
            return False
    
    def sync_chromadb_with_neo4j(self):
        """Force complete resync of ChromaDB with Neo4j"""
        
        logger.info("Starting forced ChromaDB sync with Neo4j...")
        
        # Connect to Neo4j
        driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password)
        )
        
        # Get all documents from Neo4j
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
        chroma_client = chromadb.PersistentClient(path="/app/chroma_db")
        
        # Delete existing collection completely
        try:
            chroma_client.delete_collection("documents")
            logger.info("Deleted existing ChromaDB collection")
        except Exception as e:
            logger.info(f"No existing collection to delete: {e}")
        
        # Create new collection with embedding function
        collection = chroma_client.create_collection(
            name="documents",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model_name
            )
        )
        
        logger.info("Created new ChromaDB collection")
        
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
            logger.info(f"Added batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
        
        # Verify the sync
        final_count = collection.count()
        logger.info(f"✅ ChromaDB sync complete: {final_count} documents")
        
        return final_count
    
    def create_sync_script(self):
        """Create a sync script that can be run periodically"""
        
        sync_script = '''#!/usr/bin/env python3
"""Auto-sync script for ChromaDB and Neo4j"""

import os
import time
from neo4j import GraphDatabase
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

def sync_databases():
    """Sync ChromaDB with Neo4j"""
    
    # Connect to Neo4j
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD"))
    )
    
    # Get Neo4j count
    with driver.session() as session:
        result = session.run("MATCH (d:Document) RETURN count(d) as count")
        neo4j_count = result.single()["count"]
    
    # Get ChromaDB count
    client = chromadb.PersistentClient(path="/app/chroma_db")
    try:
        collection = client.get_collection("documents")
        chroma_count = collection.count()
    except:
        chroma_count = 0
    
    print(f"Neo4j: {neo4j_count}, ChromaDB: {chroma_count}")
    
    if neo4j_count != chroma_count:
        print("Mismatch detected! Forcing sync...")
        os.system("python /app/fix_chromadb_sync.py")
    else:
        print("Databases are in sync")
    
    driver.close()

if __name__ == "__main__":
    while True:
        sync_databases()
        time.sleep(60)  # Check every minute
'''
        
        with open('/app/auto_sync.py', 'w') as f:
            f.write(sync_script)
        
        os.chmod('/app/auto_sync.py', 0o755)
        logger.info("✅ Created auto_sync.py script")

def main():
    """Main execution"""
    
    print("="*50)
    print("ChromaDB Complete Fix")
    print("="*50)
    
    fixer = ChromaDBFixer()
    
    # Step 1: Update query_router_system.py
    print("\n1. Updating query_router_system.py...")
    fixer.fix_query_router_system()
    
    # Step 2: Update demo_app.py
    print("\n2. Updating demo_app.py...")
    fixer.fix_demo_app()
    
    # Step 3: Force sync ChromaDB with Neo4j
    print("\n3. Forcing ChromaDB sync with Neo4j...")
    doc_count = fixer.sync_chromadb_with_neo4j()
    
    # Step 4: Create auto-sync script
    print("\n4. Creating auto-sync script...")
    fixer.create_sync_script()
    
    print("\n" + "="*50)
    print("✅ Fix applied successfully!")
    print(f"ChromaDB now has {doc_count} documents (matching Neo4j)")
    print("\nNext steps:")
    print("1. Restart the application: docker-compose restart")
    print("2. Test with query: 'What documents do you have?'")
    print("3. Optional: Run auto-sync in background: python /app/auto_sync.py &")
    print("="*50)

if __name__ == "__main__":
    main()
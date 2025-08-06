#!/usr/bin/env python3
"""
GraphRAG Query Router & Hybrid Retrieval System
Intelligent routing for optimal RAG performance
"""

import asyncio
import json
import logging
import time
import argparse
import sys
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import re
import math

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from neo4j import GraphDatabase
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# QUERY ROUTING MODELS
# =============================================================================

class RouteType(str, Enum):
    VECTOR = "vector"
    GRAPH = "graph" 
    HYBRID = "hybrid"

@dataclass
class QueryFeatures:
    """Features extracted from query for routing decision."""
    entity_count: int
    token_count: int
    has_multi_hop_indicators: bool
    has_relationship_words: bool
    has_complex_reasoning: bool
    question_type: str  # "what", "how", "why", "who", "where", "when"
    complexity_score: float
    named_entities: List[str]

@dataclass
class RouteDecision:
    """Router decision with explainability."""
    route: RouteType
    confidence: float 
    reasoning: str
    features: QueryFeatures
    processing_time: float

@dataclass
class RetrievalResult:
    """Single retrieval result with metadata."""
    content: str
    score: float
    source: str
    metadata: Dict[str, Any]
    route_used: RouteType

@dataclass
class HybridRetrievalResponse:
    """Complete response from hybrid retrieval system."""
    query: str
    results: List[RetrievalResult]
    route_decision: RouteDecision
    total_processing_time: float
    performance_metrics: Dict[str, Any]

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RouterConfig:
    """Configuration for the query router system."""
    # Neo4j Configuration
    NEO4J_URI: str = os.getenv("NEO4J_URI")
    NEO4J_USER: str = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD")
    
    # Embedding Model
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Router Thresholds (tunable for optimization)
    VECTOR_MAX_COMPLEXITY: float = 0.3
    GRAPH_MIN_ENTITIES: int = 2
    GRAPH_MIN_COMPLEXITY: float = 0.4
    HYBRID_MIN_COMPLEXITY: float = 0.6
    
    # Retrieval Parameters
    DEFAULT_K: int = 10
    VECTOR_K_MULTIPLIER: float = 2.0  # Get more for fusion
    
    # Performance Settings
    MAX_TRAVERSAL_DEPTH: int = 3
    CYPHER_TIMEOUT: int = 30
    
    def __post_init__(self):
        """Validate configuration."""
        required_vars = {
            'NEO4J_URI': self.NEO4J_URI,
            'NEO4J_USERNAME': self.NEO4J_USER,
            'NEO4J_PASSWORD': self.NEO4J_PASSWORD
        }
        
        missing_vars = [var for var, value in required_vars.items() if not value]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")

config = RouterConfig()

class QueryFeatureExtractor:
    """Extract features from queries for routing decisions."""
    
    def __init__(self):
        # Multi-hop indicators
        self.multi_hop_patterns = [
            r'\b(how.*relate|relationship between|connection between|link between)\b',
            r'\b(what.*cause|why.*happen|how.*affect|impact of)\b',
            r'\b(compare|versus|difference between|similar to)\b',
            r'\b(chain|sequence|pathway|process)\b'
        ]
        
        # Relationship words
        self.relationship_words = [
            'relationship', 'connection', 'link', 'association', 'correlation',
            'cause', 'effect', 'impact', 'influence', 'depend', 'related',
            'connect', 'associate', 'interact', 'affect', 'lead to'
        ]
        
        # Complex reasoning indicators
        self.complex_reasoning_patterns = [
            r'\b(analyze|evaluate|assess|compare|contrast|explain why)\b',
            r'\b(what would happen|what if|hypothetical|scenario)\b',
            r'\b(pros and cons|advantages|disadvantages|benefits|drawbacks)\b'
        ]
        
        # Question type patterns
        self.question_patterns = {
            'what': r'\bwhat\b',
            'how': r'\bhow\b', 
            'why': r'\bwhy\b',
            'who': r'\bwho\b',
            'where': r'\bwhere\b',
            'when': r'\bwhen\b'
        }
    
    def extract_features(self, query: str) -> QueryFeatures:
        """Extract comprehensive features from query."""
        query_lower = query.lower()
        tokens = query.split()
        
        # Basic counts
        entity_count = self._count_entities(query)
        token_count = len(tokens)
        
        # Pattern matching
        has_multi_hop = any(re.search(pattern, query_lower) for pattern in self.multi_hop_patterns)
        has_relationship_words = any(word in query_lower for word in self.relationship_words)
        has_complex_reasoning = any(re.search(pattern, query_lower) for pattern in self.complex_reasoning_patterns)
        
        # Question type
        question_type = self._identify_question_type(query_lower)
        
        # Complexity score
        complexity_score = self._calculate_complexity_score(
            token_count, entity_count, has_multi_hop, 
            has_relationship_words, has_complex_reasoning
        )
        
        # Named entities (simplified)
        named_entities = self._extract_named_entities(query)
        
        return QueryFeatures(
            entity_count=entity_count,
            token_count=token_count,
            has_multi_hop_indicators=has_multi_hop,
            has_relationship_words=has_relationship_words,
            has_complex_reasoning=has_complex_reasoning,  
            question_type=question_type,
            complexity_score=complexity_score,
            named_entities=named_entities
        )
    
    def _count_entities(self, query: str) -> int:
        """Count potential entities in query (capitalized words, proper nouns)."""
        # Simple heuristic - count capitalized words that aren't at sentence start
        words = query.split()
        entity_count = 0
        
        for i, word in enumerate(words):
            # Skip first word unless it's clearly a proper noun
            if i == 0:
                continue
            
            # Count capitalized words (likely proper nouns/entities)
            if word[0].isupper() and len(word) > 1:
                entity_count += 1
        
        return entity_count
    
    def _identify_question_type(self, query: str) -> str:
        """Identify the type of question being asked."""
        for q_type, pattern in self.question_patterns.items():
            if re.search(pattern, query):
                return q_type
        return "other"
    
    def _calculate_complexity_score(self, token_count: int, entity_count: int, 
                                  has_multi_hop: bool, has_relationships: bool, 
                                  has_reasoning: bool) -> float:
        """Calculate query complexity score (0-1)."""
        
        score = 0.0
        
        # Token count contribution (normalized)
        score += min(token_count / 20.0, 0.3)
        
        # Entity count contribution
        score += min(entity_count / 5.0, 0.2)
        
        # Pattern-based contributions
        if has_multi_hop:
            score += 0.25
        if has_relationships:
            score += 0.15
        if has_reasoning:
            score += 0.1
        
        return min(score, 1.0)
    
    def _extract_named_entities(self, query: str) -> List[str]:
        """Extract potential named entities from query."""
        # Simple implementation - find capitalized words
        words = query.split()
        entities = []
        
        for word in words:
            # Remove punctuation and check if capitalized
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word and clean_word[0].isupper() and len(clean_word) > 1:
                entities.append(clean_word)
        
        return entities

# =============================================================================
# AGENTIC QUERY ROUTER
# =============================================================================

class AgenticQueryRouter:
    """
    Intelligent query router that determines optimal retrieval strategy.
    """
    
    def __init__(self, config):
        self.config = config
        self.feature_extractor = QueryFeatureExtractor()
        
        # Router decision thresholds (tunable)
        self.thresholds = {
            'vector_max_complexity': 0.3,
            'graph_min_entities': 2,
            'graph_min_complexity': 0.4,
            'hybrid_min_complexity': 0.6
        }
        
        # Performance tracking
        self.routing_history = []
    
    def route_query(self, query: str) -> RouteDecision:
        """Route query to optimal retrieval strategy."""
        
        start_time = time.time()
        
        # Extract features
        features = self.feature_extractor.extract_features(query)
        
        # Make routing decision
        route, confidence, reasoning = self._make_routing_decision(features)
        
        processing_time = time.time() - start_time
        
        decision = RouteDecision(
            route=route,
            confidence=confidence,
            reasoning=reasoning,
            features=features,
            processing_time=processing_time
        )
        
        # Track for analysis
        self.routing_history.append(decision)
        
        logger.info(f"Routed query to {route.value} (confidence: {confidence:.2f})")
        return decision
    
    def _make_routing_decision(self, features: QueryFeatures) -> Tuple[RouteType, float, str]:
        """Make routing decision based on extracted features."""
        
        # Decision logic based on query characteristics
        
        # Simple vector search for basic queries
        if (features.complexity_score <= self.thresholds['vector_max_complexity'] and
            not features.has_multi_hop_indicators and
            not features.has_relationship_words):
            
            return (RouteType.VECTOR, 0.9, 
                   f"Simple query (complexity: {features.complexity_score:.2f}) best served by vector search")
        
        # Graph search for relationship-heavy queries
        if (features.has_relationship_words and 
            features.entity_count >= self.thresholds['graph_min_entities'] and
            features.complexity_score >= self.thresholds['graph_min_complexity']):
            
            return (RouteType.GRAPH, 0.85,
                   f"Relationship query with {features.entity_count} entities needs graph traversal")
        
        # Hybrid for complex multi-hop queries
        if (features.complexity_score >= self.thresholds['hybrid_min_complexity'] or
            (features.has_multi_hop_indicators and features.entity_count > 1)):
            
            return (RouteType.HYBRID, 0.8,
                   f"Complex multi-hop query (complexity: {features.complexity_score:.2f}) requires hybrid approach")
        
        # Default to vector with medium confidence
        return (RouteType.VECTOR, 0.6,
               f"Default to vector search for query with complexity {features.complexity_score:.2f}")
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing performance statistics."""
        if not self.routing_history:
            return {}
        
        routes = [d.route.value for d in self.routing_history]
        confidences = [d.confidence for d in self.routing_history]
        
        return {
            'total_queries': len(self.routing_history),
            'route_distribution': {
                'vector': routes.count('vector'),
                'graph': routes.count('graph'),
                'hybrid': routes.count('hybrid')
            },
            'average_confidence': np.mean(confidences),
            'average_processing_time': np.mean([d.processing_time for d in self.routing_history])
        }

# =============================================================================
# VECTOR RETRIEVAL SYSTEM
# =============================================================================

class VectorRetriever:
    """Semantic vector-based retrieval using ChromaDB."""
    
    def __init__(self, config):
        self.config = config
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        
        # Initialize ChromaDB with persistent storage
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Get or create collection with embedding function
        try:
            self.collection = self.chroma_client.get_collection(
                name="documents",
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=config.EMBEDDING_MODEL
                )
            )
            logger.info("Using existing ChromaDB collection")
        except:
            self.collection = self.chroma_client.create_collection(
                name="documents",
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=config.EMBEDDING_MODEL
                )
            )
            logger.info("Created new ChromaDB collection")
            
        # Populate collection if empty
        self._ensure_documents_indexed()
    
    def _ensure_documents_indexed(self):
        """Ensure documents from Neo4j are indexed in ChromaDB."""
        
        # Check if collection has documents
        try:
            count = self.collection.count()
            if count > 0:
                logger.info(f"ChromaDB collection has {count} documents")
                return
        except:
            pass
        
        logger.info("Populating ChromaDB from Neo4j documents...")
        
        # Get documents from Neo4j
        neo4j_driver = GraphDatabase.driver(
            self.config.NEO4J_URI,
            auth=(self.config.NEO4J_USER, self.config.NEO4J_PASSWORD)
        )
        
        try:
            with neo4j_driver.session() as session:
                result = session.run("""
                    MATCH (d:Document)
                    RETURN d.id as id, 
                           d.content as content, 
                           d.title as title,
                           d.source_path as source_path,
                           d.document_type as document_type
                    LIMIT 100
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
                    
                    logger.info(f"Added {len(documents)} documents to ChromaDB")
                else:
                    logger.warning("No documents found in Neo4j to index")
                    
        except Exception as e:
            logger.error(f"Failed to populate ChromaDB: {e}")
        finally:
            neo4j_driver.close()
    
    def retrieve(self, query: str, k: int = 10) -> List[RetrievalResult]:
        """Retrieve most similar documents using vector search."""
        
        start_time = time.time()
        
        try:
            # Query the collection
            results = self.collection.query(
                query_texts=[query],
                n_results=min(k, self.collection.count())
            )
            
            # Format results
            retrieval_results = []
            
            if results['documents'] and results['documents'][0]:
                for i, (doc, distance, metadata) in enumerate(zip(
                    results['documents'][0],
                    results['distances'][0], 
                    results['metadatas'][0]
                )):
                    
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    score = max(0.0, 1.0 - distance) if distance is not None else 0.0
                    
                    result = RetrievalResult(
                        content=doc,
                        score=score,
                        source=metadata.get('source_path', 'unknown'),
                        metadata={
                            'title': metadata.get('title', 'Unknown Document'),
                            'document_type': metadata.get('document_type', 'text'),
                            'similarity_distance': distance,
                            'rank': i + 1
                        },
                        route_used=RouteType.VECTOR
                    )
                    retrieval_results.append(result)
            
            processing_time = time.time() - start_time
            logger.info(f"Vector retrieval completed in {processing_time:.3f}s")
            
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Vector retrieval failed: {e}")
            return []

# =============================================================================
# GRAPH RETRIEVAL SYSTEM  
# =============================================================================

class GraphRetriever:
    """Graph-based retrieval using Neo4j traversal."""
    
    def __init__(self, config):
        self.config = config
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )
    
    def retrieve(self, query: str, entities: List[str], k: int = 10) -> List[RetrievalResult]:
        """Retrieve documents through graph traversal."""
        
        start_time = time.time()
        
        if not entities:
            # Fallback to text search if no entities
            return self._text_to_cypher_fallback(query, k)
        
        # Try entity-based traversal first
        results = self._entity_traversal_query(entities, k)
        
        # If no results, fall back to text search
        if not results:
            logger.info("No results from entity traversal, falling back to text search")
            results = self._text_to_cypher_fallback(query, k)
        
        processing_time = time.time() - start_time
        logger.info(f"Graph retrieval completed in {processing_time:.3f}s")
        
        return results
    
    def _entity_traversal_query(self, entities: List[str], k: int) -> List[RetrievalResult]:
        """Query using entity relationships (handles case where no RELATED relationships exist)."""
        
        # First, try to find documents containing the entities
        cypher_query = """
        // Find entities matching query entities
        MATCH (e:Entity)
        WHERE e.text IN $entities
        
        // Find documents containing these entities
        MATCH (d:Document)-[:CONTAINS]->(e)
        
        // Also find other entities in the same documents (co-occurrence relationships)
        MATCH (d)-[:CONTAINS]->(other:Entity)
        WHERE other.text <> e.text
        
        // Calculate relevance score based on entity matches and co-occurrences
        WITH d, e, other,
             COUNT(DISTINCT e) as matched_entities,
             COUNT(DISTINCT other) as related_entities
        
        // Aggregate results per document
        WITH d,
             matched_entities,
             related_entities,
             (matched_entities * 2.0 + related_entities * 0.5) as relevance_score,
             COLLECT(DISTINCT e.text) as found_entities,
             COLLECT(DISTINCT other.text)[0..5] as related_entity_sample
        
        RETURN d.content as content,
               relevance_score as score,
               d.source_path as source,
               found_entities,
               related_entity_sample,
               d.title as title
        ORDER BY relevance_score DESC
        LIMIT $limit
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query, entities=entities, limit=k)
                
                retrieval_results = []
                for record in result:
                    content = record.get('content', '')
                    score = float(record.get('score', 0.5))
                    source = record.get('source', 'unknown')
                    title = record.get('title', 'Unknown Document')
                    found_entities = record.get('found_entities', [])
                    related_entities = record.get('related_entity_sample', [])
                    
                    if content:  # Only add if we have content
                        result_obj = RetrievalResult(
                            content=content,
                            score=score,
                            source=source,
                            metadata={
                                'title': title,
                                'found_entities': found_entities,
                                'related_entities': related_entities,
                                'graph_method': 'entity_co_occurrence'
                            },
                            route_used=RouteType.GRAPH
                        )
                        retrieval_results.append(result_obj)
                
                return retrieval_results
                
        except Exception as e:
            logger.error(f"Entity traversal query failed: {e}")
            return []
    
    def _text_to_cypher_fallback(self, query: str, k: int) -> List[RetrievalResult]:
        """Fallback to full-text search when graph traversal returns no results."""
        
        # Use Neo4j full-text search
        cypher_query = """
        CALL db.index.fulltext.queryNodes('document_search', $query) 
        YIELD node, score
        RETURN node.content as content,
               score,
               node.source_path as source,
               node.title as title
        LIMIT $limit
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query, query=query, limit=k)
                
                retrieval_results = []
                for record in result:
                    content = record.get('content', '')
                    score = float(record.get('score', 0.3))
                    source = record.get('source', 'unknown')
                    title = record.get('title', 'Unknown Document')
                    
                    if content:
                        result_obj = RetrievalResult(
                            content=content,
                            score=score,
                            source=source,
                            metadata={
                                'title': title,
                                'graph_method': 'fulltext_search'
                            },
                            route_used=RouteType.GRAPH
                        )
                        retrieval_results.append(result_obj)
                
                return retrieval_results
                
        except Exception as e:
            logger.error(f"Full-text search fallback failed: {e}")
            # Final fallback - simple content search
            return self._simple_content_search(query, k)
    
    def _simple_content_search(self, query: str, k: int) -> List[RetrievalResult]:
        """Simple content-based search as final fallback."""
        
        cypher_query = """
        MATCH (d:Document)
        WHERE toLower(d.content) CONTAINS toLower($query_term)
        RETURN d.content as content,
               0.4 as score,
               d.source_path as source,
               d.title as title
        LIMIT $limit
        """
        
        # Extract main terms from query
        query_terms = [term.strip() for term in query.lower().split() if len(term) > 2]
        
        retrieval_results = []
        
        try:
            with self.driver.session() as session:
                for term in query_terms[:3]:  # Try first 3 significant terms
                    result = session.run(cypher_query, query_term=term, limit=k//2)
                    
                    for record in result:
                        content = record.get('content', '')
                        if content and len(content) > 50:  # Filter out very short content
                            result_obj = RetrievalResult(
                                content=content,
                                score=0.4,
                                source=record.get('source', 'unknown'),
                                metadata={
                                    'title': record.get('title', 'Unknown Document'),
                                    'search_term': term,
                                    'graph_method': 'simple_content_search'
                                },
                                route_used=RouteType.GRAPH
                            )
                            retrieval_results.append(result_obj)
                
                # Remove duplicates based on source
                seen_sources = set()
                unique_results = []
                for result in retrieval_results:
                    if result.source not in seen_sources:
                        seen_sources.add(result.source)
                        unique_results.append(result)
                
                return unique_results[:k]
                
        except Exception as e:
            logger.error(f"Simple content search failed: {e}")
            return []

# =============================================================================
# HYBRID RETRIEVAL SYSTEM
# =============================================================================

class HybridRetriever:
    """
    Combines vector and graph retrieval using Reciprocal Rank Fusion.
    """
    
    def __init__(self, config):
        self.config = config
        self.vector_retriever = VectorRetriever(config)
        self.graph_retriever = GraphRetriever(config)
        self.router = AgenticQueryRouter(config)
    
    async def retrieve(self, query: str, k: int = 10) -> HybridRetrievalResponse:
        """Main retrieval method with intelligent routing."""
        
        start_time = time.time()
        
        # Route the query
        route_decision = self.router.route_query(query)
        
        # Execute retrieval based on routing decision
        if route_decision.route == RouteType.VECTOR:
            results = self.vector_retriever.retrieve(query, k)
            
        elif route_decision.route == RouteType.GRAPH:
            entities = route_decision.features.named_entities
            results = self.graph_retriever.retrieve(query, entities, k)
            
        else:  # HYBRID
            results = self._hybrid_retrieve(query, route_decision.features, k)
        
        total_time = time.time() - start_time
        
        # Performance metrics
        metrics = {
            'retrieval_time': total_time,
            'route_time': route_decision.processing_time,
            'results_count': len(results),
            'average_score': np.mean([r.score for r in results]) if results else 0.0,
            'route_used': route_decision.route.value
        }
        
        return HybridRetrievalResponse(
            query=query,
            results=results,
            route_decision=route_decision,
            total_processing_time=total_time,
            performance_metrics=metrics
        )
    
    def _hybrid_retrieve(self, query: str, features: QueryFeatures, k: int) -> List[RetrievalResult]:
        """Perform hybrid retrieval with vector narrowing + graph expansion."""
        
        # Step 1: Vector narrowing - get initial candidates
        vector_k = min(k * 2, 20)  # Get more for fusion
        vector_results = self.vector_retriever.retrieve(query, vector_k)
        
        # Step 2: Graph expansion - find related content through graph
        entities = features.named_entities
        graph_k = min(k * 2, 15)
        graph_results = self.graph_retriever.retrieve(query, entities, graph_k)
        
        # Step 3: Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(vector_results, graph_results, k)
        
        return fused_results
    
    def _reciprocal_rank_fusion(self, vector_results: List[RetrievalResult], 
                               graph_results: List[RetrievalResult], k: int) -> List[RetrievalResult]:
        """Combine results using Reciprocal Rank Fusion algorithm."""
        
        # Create content-based lookup for deduplication
        content_scores = {}
        
        # Process vector results
        for i, result in enumerate(vector_results):
            content_key = self._get_content_key(result.content, result.source)
            rrf_score = 1.0 / (60 + i + 1)  # RRF formula with k=60
            
            if content_key in content_scores:
                content_scores[content_key]['score'] += rrf_score
                content_scores[content_key]['sources'].append('vector')
            else:
                content_scores[content_key] = {
                    'score': rrf_score,
                    'result': result,
                    'sources': ['vector'],
                    'vector_rank': i + 1,
                    'graph_rank': None
                }
        
        # Process graph results  
        for i, result in enumerate(graph_results):
            content_key = self._get_content_key(result.content, result.source)
            rrf_score = 1.0 / (60 + i + 1)
            
            if content_key in content_scores:
                content_scores[content_key]['score'] += rrf_score
                content_scores[content_key]['sources'].append('graph')
                content_scores[content_key]['graph_rank'] = i + 1
            else:
                content_scores[content_key] = {
                    'score': rrf_score,
                    'result': result,
                    'sources': ['graph'],
                    'vector_rank': None,
                    'graph_rank': i + 1
                }
        
        # Sort by fused score and return top k
        sorted_items = sorted(content_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        
        fused_results = []
        for content_key, data in sorted_items[:k]:
            result = data['result']
            # Update score and route information
            result.score = data['score']
            result.route_used = RouteType.HYBRID
            result.metadata = result.metadata or {}
            result.metadata.update({
                'fusion_sources': data['sources'],
                'vector_rank': data['vector_rank'],
                'graph_rank': data['graph_rank'],
                'rrf_score': data['score']
            })
            fused_results.append(result)
        
        return fused_results
    
    def _get_content_key(self, content: str, source: str) -> str:
        """Generate key for content deduplication."""
        # Use source path as primary key, content snippet as secondary
        source_key = source.split('/')[-1] if source else 'unknown'
        content_snippet = content[:100].strip().lower()
        return f"{source_key}::{content_snippet}"

# =============================================================================
# PERFORMANCE EVALUATOR
# =============================================================================

class PerformanceEvaluator:
    """Evaluate system performance against benchmarks."""
    
    def __init__(self, hybrid_retriever: HybridRetriever):
        self.hybrid_retriever = hybrid_retriever
        self.evaluation_results = []
    
    async def run_evaluation(self, test_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run evaluation on test query set."""
        
        logger.info(f"Starting evaluation on {len(test_queries)} queries")
        
        results = {
            'routing_accuracy': [],
            'retrieval_latencies': [],
            'route_distribution': {'vector': 0, 'graph': 0, 'hybrid': 0},
            'confusion_matrix': {}
        }
        
        for query_data in test_queries:
            query = query_data['query']
            expected_route = query_data.get('expected_route')
            
            # Perform retrieval
            response = await self.hybrid_retriever.retrieve(query)
            
            # Track metrics
            results['retrieval_latencies'].append(response.total_processing_time)
            results['route_distribution'][response.route_decision.route.value] += 1
            
            # Check routing accuracy if ground truth available
            if expected_route:
                actual_route = response.route_decision.route.value
                is_correct = actual_route == expected_route
                results['routing_accuracy'].append(is_correct)
                
                # Update confusion matrix
                if expected_route not in results['confusion_matrix']:
                    results['confusion_matrix'][expected_route] = {}
                if actual_route not in results['confusion_matrix'][expected_route]:
                    results['confusion_matrix'][expected_route][actual_route] = 0
                results['confusion_matrix'][expected_route][actual_route] += 1
        
        # Calculate summary statistics
        summary = {
            'total_queries': len(test_queries),
            'average_latency': np.mean(results['retrieval_latencies']),
            'median_latency': np.median(results['retrieval_latencies']),
            'routing_accuracy': np.mean(results['routing_accuracy']) if results['routing_accuracy'] else None,
            'route_distribution': results['route_distribution'],
            'confusion_matrix': results['confusion_matrix']
        }
        
        logger.info(f"Evaluation completed: {summary}")
        return summary

# =============================================================================
# DEMO QUERIES AND TESTING
# =============================================================================

def get_demo_queries() -> List[Dict[str, Any]]:
    """Get demo queries for testing different routing strategies."""
    return [
        # Vector queries (simple, factual)
        {
            'query': 'What is IBM?',
            'expected_route': 'vector',
            'type': 'simple_factual',
            'description': 'Simple entity lookup - should route to vector search'
        },
        {
            'query': 'Lyft revenue 2021',
            'expected_route': 'vector', 
            'type': 'simple_factual',
            'description': 'Specific financial fact - vector search optimal'
        },
        
        # Graph queries (relationship-focused)
        {
            'query': 'What is the relationship between IBM and Fortran?',
            'expected_route': 'graph',
            'type': 'relationship',
            'description': 'Entity relationship query - needs graph traversal'
        },
        {
            'query': 'How are Rich Draves and Peter Albert Amjad connected?',
            'expected_route': 'graph',
            'type': 'relationship', 
            'description': 'Person-to-person connections - graph search required'
        },
        
        # Hybrid queries (complex, multi-hop)
        {
            'query': 'How did IBM\'s technology development affect the Fortran programming language and its adoption?',
            'expected_route': 'hybrid',
            'type': 'complex_multi_hop',
            'description': 'Multi-hop causal reasoning - hybrid approach needed'
        },
        {
            'query': 'Compare Lyft\'s financial performance with industry trends and analyze the impact on their technology investments',
            'expected_route': 'hybrid',
            'type': 'complex_analysis',
            'description': 'Complex comparative analysis - requires hybrid fusion'
        },
        
        # Financial document specific queries  
        {
            'query': 'What were the budget allocations in 2021?',
            'expected_route': 'vector',
            'type': 'financial_lookup',
            'description': 'Financial fact lookup from budget documents'
        },
        {
            'query': 'How did budget changes from 2019 to 2023 affect different departments?',
            'expected_route': 'hybrid', 
            'type': 'temporal_analysis',
            'description': 'Multi-year trend analysis requiring hybrid approach'
        }
    ]

async def run_router_demo(data_directory: str = None):
    """Run interactive demo of the query router."""
    
    print("="*70)
    print("🧠 GraphRAG Intelligent Query Router Demo")
    print("="*70)
    print("Demonstrating automatic routing between Vector, Graph, and Hybrid retrieval")
    print()
    
    try:
        # Initialize system
        logger.info("Initializing GraphRAG system...")
        hybrid_retriever = HybridRetriever(config)
        
        # Test data directory if provided
        if data_directory:
            logger.info(f"Testing with data from: {data_directory}")
        
        # Get demo queries
        demo_queries = get_demo_queries()
        
        print("🎯 Demo Queries Available:")
        for i, query_data in enumerate(demo_queries):
            print(f"{i+1}. {query_data['description']}")
            print(f"   Query: \"{query_data['query']}\"")
            print(f"   Expected Route: {query_data['expected_route'].upper()}")
            print()
        
        # Interactive or batch mode
        mode = input("Choose mode: (1) Interactive, (2) Batch all queries, (3) Custom query: ")
        
        if mode == "1":
            # Interactive mode
            while True:
                print("\n" + "-"*50)
                choice = input(f"Enter query number (1-{len(demo_queries)}) or 'q' to quit: ")
                
                if choice.lower() == 'q':
                    break
                
                try:
                    query_idx = int(choice) - 1
                    if 0 <= query_idx < len(demo_queries):
                        query_data = demo_queries[query_idx]
                        await demo_single_query(hybrid_retriever, query_data)
                    else:
                        print("Invalid query number!")
                except ValueError:
                    print("Please enter a valid number!")
        
        elif mode == "2":
            # Batch mode
            print("\n🚀 Running all demo queries...\n")
            routing_results = []
            
            for i, query_data in enumerate(demo_queries):
                print(f"Query {i+1}/{len(demo_queries)}: {query_data['query']}")
                result = await demo_single_query(hybrid_retriever, query_data, show_details=False)
                routing_results.append(result)
                print("-"*50)
            
            # Show summary
            show_routing_summary(routing_results, demo_queries)
        
        elif mode == "3":
            # Custom query mode
            while True:
                custom_query = input("\nEnter your custom query (or 'q' to quit): ")
                if custom_query.lower() == 'q':
                    break
                
                custom_query_data = {
                    'query': custom_query,
                    'type': 'custom',
                    'description': 'Custom user query'
                }
                await demo_single_query(hybrid_retriever, custom_query_data)
        
        print("\n✅ Demo completed!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")

async def demo_single_query(hybrid_retriever, query_data: Dict[str, Any], show_details: bool = True):
    """Demo a single query with detailed output."""
    
    query = query_data['query']
    
    if show_details:
        print(f"\n🔍 Query: \"{query}\"")
        print(f"📝 Type: {query_data.get('type', 'unknown')}")
        if 'expected_route' in query_data:
            print(f"🎯 Expected Route: {query_data['expected_route'].upper()}")
    
    try:
        # Execute query
        start_time = time.time()
        response = await hybrid_retriever.retrieve(query)
        total_time = time.time() - start_time
        
        # Display routing decision
        route_decision = response.route_decision
        
        if show_details:
            print(f"\n⚡ ROUTING DECISION:")
            print(f"  Route: {route_decision.route.value.upper()}")
            print(f"  Confidence: {route_decision.confidence:.1%}")
            print(f"  Reasoning: {route_decision.reasoning}")
            print(f"  Processing Time: {route_decision.processing_time:.3f}s")
            
            # Query features
            features = route_decision.features
            print(f"\n🔧 QUERY ANALYSIS:")
            print(f"  Entities Found: {features.entity_count}")
            print(f"  Token Count: {features.token_count}")
            print(f"  Question Type: {features.question_type}")
            print(f"  Complexity Score: {features.complexity_score:.3f}")
            print(f"  Multi-hop Indicators: {features.has_multi_hop_indicators}")
            print(f"  Relationship Words: {features.has_relationship_words}")
            
            if features.named_entities:
                print(f"  Named Entities: {', '.join(features.named_entities[:5])}")
            
            # Results summary
            print(f"\n📊 RETRIEVAL RESULTS:")
            print(f"  Results Found: {len(response.results)}")
            print(f"  Total Time: {total_time:.3f}s")
            
            if response.results:
                avg_score = np.mean([r.score for r in response.results])
                print(f"  Average Score: {avg_score:.3f}")
                
                print(f"\n📄 TOP RESULTS:")
                for i, result in enumerate(response.results[:3]):
                    print(f"  {i+1}. Source: {result.source}")
                    print(f"     Score: {result.score:.3f}")
                    print(f"     Content: {result.content[:100]}...")
                    print()
        
        # Return results for analysis
        return {
            'query': query,
            'actual_route': route_decision.route.value,
            'expected_route': query_data.get('expected_route'),
            'confidence': route_decision.confidence,
            'processing_time': total_time,
            'results_count': len(response.results),
            'features': route_decision.features
        }
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        if show_details:
            print(f"❌ Query failed: {e}")
        return None

def show_routing_summary(routing_results: List[Dict], demo_queries: List[Dict]):
    """Show summary of routing performance."""
    
    print("\n" + "="*70)
    print("📈 ROUTING PERFORMANCE SUMMARY")
    print("="*70)
    
    # Calculate accuracy
    correct_routes = 0
    total_with_expected = 0
    
    route_distribution = {'vector': 0, 'graph': 0, 'hybrid': 0}
    total_time = 0
    total_results = 0
    
    for result in routing_results:
        if result is None:
            continue
            
        # Route distribution
        route_distribution[result['actual_route']] += 1
        
        # Performance metrics
        total_time += result['processing_time']
        total_results += result['results_count']
        
        # Accuracy
        if result.get('expected_route'):
            total_with_expected += 1
            if result['actual_route'] == result['expected_route']:
                correct_routes += 1
    
    # Display metrics
    if total_with_expected > 0:
        accuracy = (correct_routes / total_with_expected) * 100
        print(f"🎯 Router Accuracy: {accuracy:.1f}% ({correct_routes}/{total_with_expected})")
    
    print(f"⚡ Average Processing Time: {total_time/len(routing_results):.3f}s")
    print(f"📊 Average Results per Query: {total_results/len(routing_results):.1f}")
    
    print(f"\n🔄 Route Distribution:")
    total_queries = sum(route_distribution.values())
    for route, count in route_distribution.items():
        percentage = (count / total_queries) * 100 if total_queries > 0 else 0
        print(f"  {route.upper()}: {count} queries ({percentage:.1f}%)")
    
    # Confusion matrix
    print(f"\n🎯 Routing Decisions:")
    for result, query_data in zip(routing_results, demo_queries):
        if result is None:
            continue
        
        expected = query_data.get('expected_route', 'unknown')
        actual = result['actual_route']
        status = "✅" if expected == actual else "❌"
        
        print(f"  {status} Expected: {expected.upper():<6} → Actual: {actual.upper():<6} | {query_data['query'][:50]}...")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main entry point for the query router demo."""
    
    parser = argparse.ArgumentParser(
        description="GraphRAG Intelligent Query Router Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python query_router_system.py                    # Interactive demo
  python query_router_system.py --data-dir data/input   # Demo with specific data
  python query_router_system.py --batch            # Run all demo queries
        """
    )
    
    parser.add_argument(
        '--data-dir',
        help='Directory containing processed documents (optional)'
    )
    
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Run all demo queries in batch mode'
    )
    
    parser.add_argument(
        '--query',
        help='Run a single custom query'
    )
    
    args = parser.parse_args()
    
    print("🧠 GraphRAG Intelligent Query Router")
    print("====================================")
    print("Demonstrating intelligent routing between Vector, Graph, and Hybrid retrieval")
    print()
    
    try:
        if args.query:
            # Single query mode
            async def run_single():
                hybrid_retriever = HybridRetriever(config)
                query_data = {
                    'query': args.query,
                    'type': 'custom',
                    'description': 'Custom command-line query'
                }
                await demo_single_query(hybrid_retriever, query_data)
            
            asyncio.run(run_single())
        else:
            # Interactive or batch demo
            asyncio.run(run_router_demo(args.data_dir))
    
    except KeyboardInterrupt:
        print("\n❌ Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

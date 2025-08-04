# GraphRAG Document Processing System

## Overview

This project implements an advanced document processing system that automatically extracts knowledge from PDF files, CSV datasets, and text documents. The system builds a graph-based knowledge representation stored in Neo4j that enables intelligent information retrieval and analysis.

## What This System Does

The GraphRAG system takes a folder containing various document types and processes them to create a structured knowledge graph. During processing, the system identifies important entities such as people, organizations, dates, and locations within the documents. These entities are then stored in a Neo4j graph database along with their relationships.

The system is designed to work with any collection of documents provided at runtime, making it suitable for various use cases from financial document analysis to research paper processing. The lazy indexing approach ensures that only new or modified documents are processed, significantly reducing computational costs.

## Graph Database Visualization

After processing documents, you can visualize the extracted knowledge graph in Neo4j Browser using:

```cypher
MATCH (n)-[r]-(m) RETURN n,m,r 
```

This query will display a network of documents (blue nodes) connected to their extracted entities (pink nodes) through CONTAINS relationships. The visualization shows how information is structured and interconnected across your document collection.

![Neo4j Graph Visualization](cypher/neo4j-graph-example.png)

*Example: Document nodes (blue) connected to entity nodes (pink) showing extracted people, organizations, dates, and other concepts from processed documents.*

## Key Features

The document ingestion pipeline supports multiple file formats including PDF, CSV, TXT, and DOCX files. The system uses advanced natural language processing techniques to extract named entities and can handle complex document structures. All processing is incremental, meaning subsequent runs only process files that have changed since the last execution.

## Getting Started

### Prerequisites

You need access to a Neo4j database instance (the free Neo4j Aura tier works perfectly), Python 3.10 or higher, and the ability to install required dependencies.

### Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Configuration

Create a `.env` file in the project root with your database credentials:

```
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
```

### Processing Documents

Place your documents in any directory structure. The system will recursively process all supported file types:

```bash
python runtime_processor.py path/to/your/documents
```

The system will analyze the directory contents, extract entities from each document, and store the results in your Neo4j database. Processing reports are automatically generated with detailed metrics.

## Understanding the Output

After processing, the system generates comprehensive reports showing how many documents were processed, the number of entities extracted, and performance metrics. You can examine the extracted knowledge by connecting to your Neo4j database using the visualization query above.

The graph database will contain document nodes representing each processed file, entity nodes for extracted information, and CONTAINS relationships connecting documents to their entities.

## System Architecture

The core system consists of a data ingestion pipeline that handles document loading and preprocessing, an entity extraction engine that identifies and classifies important information, and a graph construction module that creates optimized database schemas.

The lazy processing engine ensures efficient resource utilization by tracking file changes and only processing modified content. This approach scales well for large document collections and reduces processing time for incremental updates.

## Testing and Validation

The project includes testing tools to validate system functionality. You can verify database connectivity and check entity extraction quality using the provided test scripts in the `tests/` directory.

## Development Status

This system currently provides the core document processing and graph construction functionality. Additional features including intelligent query routing and hybrid retrieval are under active development.

## License

This project is available under the MIT License.

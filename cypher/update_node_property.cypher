// Run this in Neo4j Browser to clean old data
MATCH (n) WHERE n.source_path CONTAINS "temp_PaulGrahamEssayDataset" 
SET n.source_path = 'temp_PaulGrahamEssayDataset/doc_0.txt'
RETURN n
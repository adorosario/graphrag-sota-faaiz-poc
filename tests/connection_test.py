from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

with driver.session() as session:
    result = session.run("MATCH (n) RETURN count(n) AS total_nodes")
    print(f"Total nodes: {result.single()['total_nodes']}")

    result = session.run("MATCH (d:Document) RETURN count(d) AS docs")
    print(f"Documents: {result.single()['docs']}")

    result = session.run("MATCH (e:Entity) RETURN count(e) AS entities")
    print(f"Entities: {result.single()['entities']}")

driver.close()

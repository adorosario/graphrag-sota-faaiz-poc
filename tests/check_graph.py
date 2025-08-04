# check_graph.py
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

with driver.session() as session:
    # Documents
    result = session.run("MATCH (d:Document) RETURN d.title, d.document_type")
    print("📄 Documents:")
    for record in result:
        print(f"  - {record['d.title']} ({record['d.document_type']})")

    # Entities
    result = session.run("MATCH (e:Entity) RETURN e.text, e.label LIMIT 10")
    print("\n🔍 Entities (top 10):")
    for record in result:
        print(f"  - {record['e.text']} ({record['e.label']})")

    # Relationships
    result = session.run(
        "MATCH ()-[r:RELATED]->() RETURN count(r) as rel_count")
    rel_count = result.single()['rel_count']
    print(f"\n🔗 Relationships: {rel_count}")

driver.close()

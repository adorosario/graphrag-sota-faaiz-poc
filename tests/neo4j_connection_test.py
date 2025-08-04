import os
from neo4j import GraphDatabase, basic_auth
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


def test_connection():
    try:
        driver = GraphDatabase.driver(
            NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            result = session.run("RETURN 1 AS result")
            value = result.single()["result"]
            print("Neo4j connection successful! Test query result:", value)
            result_nodes = session.run("MATCH (n) RETURN count(n) AS node_count")
            node_count = result_nodes.single()["node_count"]
            print(f"Total nodes in database: {node_count}")
    except Exception as e:
        print("Neo4j connection failed:", e)
    finally:
        driver.close()


if __name__ == "__main__":
    test_connection()

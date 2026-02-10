
import logging
from graph.neo4j_client import Neo4jClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_vector_indexes():
    client = Neo4jClient()
    
    # Define labels that have embeddings
    labels = ["Requirement", "Definition", "Exemption", "Process", "CrossReference"]
    
    for label in labels:
        index_name = f"{label.lower()}_vector_index"
        
        # Check if index exists already
        try:
            # Drop if exists (optional, but good for clean setup)
            # client.execute_query(f"DROP INDEX {index_name} IF EXISTS")
            
            # Neo4j 5.x syntax for vector index
            # dim=384 for all-MiniLM-L6-v2
            # cosine similarity is common for these models
            query = f"""
            CREATE VECTOR INDEX {index_name} IF NOT EXISTS
            FOR (n:{label})
            ON (n.embedding)
            OPTIONS {{
              indexConfig: {{
                `vector.dimensions`: 384,
                `vector.similarity_function`: 'cosine'
              }}
            }}
            """
            client.execute_query(query)
            logger.info(f"✓ Created vector index for {label}")
        except Exception as e:
            logger.error(f"Failed to create index for {label}: {e}")

    # Also create fulltext indexes for better keyword search
    try:
        client.execute_query("""
        CREATE FULLTEXT INDEX all_text_index IF NOT EXISTS
        FOR (n:Requirement|Definition|Exemption|Process)
        ON EACH [n.text, n.name, n.term]
        """)
        logger.info("✓ Created fulltext index")
    except Exception as e:
        logger.error(f"Failed to create fulltext index: {e}")

    client.close()

if __name__ == "__main__":
    setup_vector_indexes()

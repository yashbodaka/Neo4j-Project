"""
Neo4j Database Client
Handles all connections and operations with the Neo4j database.
"""

import logging
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Neo4jClient:
    """
    Neo4j database client with connection pooling and transaction management.
    """
    
    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
        """
        Initialize Neo4j connection.
        
        Args:
            uri: Neo4j connection URI (defaults to env variable)
            user: Neo4j username (defaults to env variable)
            password: Neo4j password (defaults to env variable)
        """
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "")
        
        self.driver = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=120
            )
            # Verify connectivity
            self.driver.verify_connectivity()
            logger.info(f"✓ Connected to Neo4j at {self.uri}")
        except AuthError:
            logger.error("Authentication failed. Check NEO4J_USER and NEO4J_PASSWORD.")
            raise
        except ServiceUnavailable:
            logger.error(f"Could not connect to Neo4j at {self.uri}. Is Neo4j running?")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def health_check(self) -> bool:
        """
        Check if the database connection is healthy.
        
        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS health")
                return result.single()["health"] == 1
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return results.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of result dictionaries
        """
        parameters = parameters or {}
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters)
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Parameters: {parameters}")
            raise
    
    def execute_write_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a write query within a transaction.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of result dictionaries
        """
        parameters = parameters or {}
        
        def _execute(tx):
            result = tx.run(query, parameters)
            return [record.data() for record in result]
        
        try:
            with self.driver.session() as session:
                return session.execute_write(_execute)
        except Exception as e:
            logger.error(f"Write query execution failed: {e}")
            logger.error(f"Query: {query}")
            raise
    
    def batch_write(self, query: str, data: List[Dict[str, Any]], batch_size: int = 1000):
        """
        Execute write query in batches for performance.
        
        Args:
            query: Cypher query with $batch parameter
            data: List of parameter dictionaries
            batch_size: Number of records per batch
        """
        total = len(data)
        for i in range(0, total, batch_size):
            batch = data[i:i + batch_size]
            try:
                self.execute_write_query(query, {"batch": batch})
                logger.info(f"Processed batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size}")
            except Exception as e:
                logger.error(f"Batch write failed at index {i}: {e}")
                raise
    
    def clear_database(self):
        """
        WARNING: Delete all nodes and relationships in the database.
        Use only for testing/development.
        """
        logger.warning("Clearing entire database...")
        self.execute_write_query("MATCH (n) DETACH DELETE n")
        logger.info("Database cleared")
    
    def get_database_stats(self) -> Dict[str, int]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with node and relationship counts
        """
        node_count_query = "MATCH (n) RETURN count(n) AS count"
        rel_count_query = "MATCH ()-[r]->() RETURN count(r) AS count"
        
        node_count = self.execute_query(node_count_query)[0]["count"]
        rel_count = self.execute_query(rel_count_query)[0]["count"]
        
        # Get counts by label
        label_query = """
        MATCH (n)
        RETURN labels(n)[0] AS label, count(*) AS count
        ORDER BY count DESC
        """
        label_counts = self.execute_query(label_query)
        
        # Get counts by relationship type
        rel_type_query = """
        MATCH ()-[r]->()
        RETURN type(r) AS type, count(*) AS count
        ORDER BY count DESC
        """
        rel_type_counts = self.execute_query(rel_type_query)
        
        return {
            "total_nodes": node_count,
            "total_relationships": rel_count,
            "nodes_by_label": {item["label"]: item["count"] for item in label_counts},
            "relationships_by_type": {item["type"]: item["count"] for item in rel_type_counts}
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


if __name__ == "__main__":
    # Test connection
    try:
        client = Neo4jClient()
        
        if client.health_check():
            print("✓ Neo4j connection healthy")
            
            stats = client.get_database_stats()
            print(f"\nDatabase Statistics:")
            print(f"  Total Nodes: {stats['total_nodes']}")
            print(f"  Total Relationships: {stats['total_relationships']}")
            
            if stats['nodes_by_label']:
                print(f"\nNodes by Label:")
                for label, count in stats['nodes_by_label'].items():
                    print(f"  {label}: {count}")
            
            if stats['relationships_by_type']:
                print(f"\nRelationships by Type:")
                for rel_type, count in stats['relationships_by_type'].items():
                    print(f"  {rel_type}: {count}")
        else:
            print("✗ Neo4j connection failed health check")
        
        client.close()
    except Exception as e:
        print(f"✗ Failed to connect to Neo4j: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure Neo4j is running locally")
        print("2. Check .env file has correct NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD")
        print("3. Verify credentials with: neo4j console")

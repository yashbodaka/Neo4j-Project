#!/usr/bin/env python3
"""
Export Neo4j Database
Creates a dump of the Neo4j database using Cypher queries
"""

from neo4j import GraphDatabase
from neo4j.time import DateTime
import json
from pathlib import Path
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

def convert_value(value):
    """Convert Neo4j types to JSON-serializable types"""
    if isinstance(value, DateTime):
        return value.iso_format()
    elif isinstance(value, (list, tuple)):
        return [convert_value(v) for v in value]
    elif isinstance(value, dict):
        return {k: convert_value(v) for k, v in value.items()}
    else:
        return value

def export_database():
    """Export the entire Neo4j database to JSON"""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    backup_dir = Path("backup")
    backup_dir.mkdir(exist_ok=True)
    
    with driver.session() as session:
        # Export all nodes
        print("Exporting nodes...")
        nodes_result = session.run("""
            MATCH (n)
            RETURN 
                id(n) as node_id,
                labels(n) as labels,
                properties(n) as properties
        """)
        
        nodes = []
        for record in nodes_result:
            nodes.append({
                "id": record["node_id"],
                "labels": record["labels"],
                "properties": convert_value(dict(record["properties"]))
            })
        
        print(f"✓ Exported {len(nodes)} nodes")
        
        # Export all relationships
        print("Exporting relationships...")
        rels_result = session.run("""
            MATCH (a)-[r]->(b)
            RETURN 
                id(r) as rel_id,
                id(a) as start_id,
                id(b) as end_id,
                type(r) as type,
                properties(r) as properties
        """)
        
        relationships = []
        for record in rels_result:
            relationships.append({
                "id": record["rel_id"],
                "start_id": record["start_id"],
                "end_id": record["end_id"],
                "type": record["type"],
                "properties": convert_value(dict(record["properties"]))
            })
        
        print(f"✓ Exported {len(relationships)} relationships")
        
        # Get database statistics
        stats = session.run("""
            MATCH (n) 
            WITH count(n) as node_count
            MATCH ()-[r]->()
            RETURN node_count, count(r) as rel_count
        """).single()
        
        # Create export data
        export_data = {
            "metadata": {
                "database": "neo4j",
                "export_time": "2026-02-10",
                "node_count": stats["node_count"],
                "relationship_count": stats["rel_count"]
            },
            "nodes": nodes,
            "relationships": relationships
        }
        
        # Save to file
        output_file = backup_dir / "neo4j_dump.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Database exported successfully to: {output_file}")
        print(f"  Nodes: {stats['node_count']}")
        print(f"  Relationships: {stats['rel_count']}")
        print(f"  File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    driver.close()

if __name__ == "__main__":
    export_database()

"""
Graph Traversal
Graph algorithms for finding paths, dependencies, and relationships.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass

from graph.neo4j_client import Neo4jClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GraphPath:
    """Represents a path through the knowledge graph."""
    nodes: List[Dict]
    relationships: List[str]
    length: int
    score: float = 0.0


class GraphTraversal:
    """Graph traversal algorithms for regulatory knowledge graph."""
    
    def __init__(self, neo4j_client: Optional[Neo4jClient] = None):
        """
        Initialize graph traversal.
        
        Args:
            neo4j_client: Neo4jClient instance
        """
        self.client = neo4j_client or Neo4jClient()
    
    def find_shortest_path(
        self,
        start_node_id: str,
        end_node_id: str,
        max_depth: int = 5
    ) -> Optional[GraphPath]:
        """
        Find shortest path between two nodes.
        
        Args:
            start_node_id: Starting node ID
            end_node_id: Ending node ID
            max_depth: Maximum path length
            
        Returns:
            GraphPath object or None
        """
        cypher = """
        MATCH (start {id: $start_id})
        MATCH (end {id: $end_id})
        MATCH path = shortestPath((start)-[*..{max_depth}]-(end))
        RETURN 
            [node IN nodes(path) | {
                id: node.id,
                type: labels(node)[0],
                text: node.text,
                name: COALESCE(node.name, node.term, node.text)
            }] AS nodes,
            [rel IN relationships(path) | type(rel)] AS relationships,
            length(path) AS length
        LIMIT 1
        """.replace('{max_depth}', str(max_depth))
        
        try:
            results = self.client.execute_query(cypher, {
                'start_id': start_node_id,
                'end_id': end_node_id
            })
            
            if results:
                record = results[0]
                return GraphPath(
                    nodes=record['nodes'],
                    relationships=record['relationships'],
                    length=record['length']
                )
            
        except Exception as e:
            logger.error(f"Shortest path query failed: {e}")
        
        return None
    
    def find_all_paths(
        self,
        start_node_id: str,
        end_node_id: str,
        max_depth: int = 4,
        limit: int = 10
    ) -> List[GraphPath]:
        """
        Find all paths between two nodes.
        
        Args:
            start_node_id: Starting node ID
            end_node_id: Ending node ID
            max_depth: Maximum path length
            limit: Maximum number of paths
            
        Returns:
            List of GraphPath objects
        """
        cypher = """
        MATCH (start {id: $start_id})
        MATCH (end {id: $end_id})
        MATCH path = (start)-[*..{max_depth}]-(end)
        WHERE length(path) <= {max_depth}
        RETURN 
            [node IN nodes(path) | {
                id: node.id,
                type: labels(node)[0],
                text: node.text,
                name: COALESCE(node.name, node.term, node.text)
            }] AS nodes,
            [rel IN relationships(path) | type(rel)] AS relationships,
            length(path) AS length
        ORDER BY length ASC
        LIMIT {limit}
        """.replace('{max_depth}', str(max_depth)).replace('{limit}', str(limit))
        
        try:
            results = self.client.execute_query(cypher, {
                'start_id': start_node_id,
                'end_id': end_node_id
            })
            
            paths = []
            for record in results:
                paths.append(GraphPath(
                    nodes=record['nodes'],
                    relationships=record['relationships'],
                    length=record['length']
                ))
            
            return paths
            
        except Exception as e:
            logger.error(f"All paths query failed: {e}")
            return []
    
    def find_dependencies(
        self,
        requirement_id: str,
        max_depth: int = 3
    ) -> List[GraphPath]:
        """
        Find all dependencies for a requirement.
        
        Args:
            requirement_id: Requirement node ID
            max_depth: Maximum dependency chain depth
            
        Returns:
            List of dependency paths
        """
        cypher = """
        MATCH (req:Requirement {id: $req_id})
        MATCH path = (req)-[:REQUIRES*1..{max_depth}]->(dep:Requirement)
        RETURN 
            [node IN nodes(path) | {
                id: node.id,
                type: labels(node)[0],
                text: node.text,
                citation: node.citation_text
            }] AS nodes,
            [rel IN relationships(path) | type(rel)] AS relationships,
            length(path) AS length
        ORDER BY length ASC
        LIMIT 20
        """.replace('{max_depth}', str(max_depth))
        
        try:
            results = self.client.execute_query(cypher, {
                'req_id': requirement_id
            })
            
            paths = []
            for record in results:
                paths.append(GraphPath(
                    nodes=record['nodes'],
                    relationships=record['relationships'],
                    length=record['length']
                ))
            
            return paths
            
        except Exception as e:
            logger.error(f"Dependencies query failed: {e}")
            return []
    
    def find_neighbors(
        self,
        node_id: str,
        relationship_types: Optional[List[str]] = None,
        max_neighbors: int = 20
    ) -> List[Dict]:
        """
        Find all neighboring nodes.
        
        Args:
            node_id: Node ID
            relationship_types: Filter by relationship types
            max_neighbors: Maximum neighbors to return
            
        Returns:
            List of neighbor node dictionaries
        """
        rel_filter = ""
        if relationship_types:
            rel_types = '|'.join(relationship_types)
            rel_filter = f":{rel_types}"
        
        cypher = f"""
        MATCH (node {{id: $node_id}})
        MATCH (node)-[rel{rel_filter}]-(neighbor)
        RETURN DISTINCT
            neighbor.id AS id,
            labels(neighbor)[0] AS type,
            neighbor.text AS text,
            COALESCE(neighbor.name, neighbor.term) AS name,
            type(rel) AS relationship
        LIMIT {max_neighbors}
        """
        
        try:
            results = self.client.execute_query(cypher, {
                'node_id': node_id
            })
            
            neighbors = []
            for record in results:
                neighbors.append({
                    'id': record['id'],
                    'type': record['type'],
                    'text': record['text'],
                    'name': record['name'],
                    'relationship': record['relationship']
                })
            
            return neighbors
            
        except Exception as e:
            logger.error(f"Neighbors query failed: {e}")
            return []
    
    def find_communities(
        self,
        min_size: int = 3,
        limit: int = 10
    ) -> List[Dict]:
        """
        Find clusters of related requirements.
        
        Args:
            min_size: Minimum community size
            limit: Maximum communities to return
            
        Returns:
            List of community dictionaries
        """
        cypher = """
        MATCH (req:Requirement)-[:APPLIES_TO]->(proc:Process)
        WITH proc, collect(DISTINCT req) AS requirements
        WHERE size(requirements) >= $min_size
        RETURN 
            proc.name AS process,
            size(requirements) AS requirement_count,
            [r IN requirements | {
                id: r.id,
                text: r.text,
                citation: r.citation_text
            }] AS requirements
        ORDER BY requirement_count DESC
        LIMIT $limit
        """
        
        try:
            results = self.client.execute_query(cypher, {
                'min_size': min_size,
                'limit': limit
            })
            
            communities = []
            for record in results:
                communities.append({
                    'process': record['process'],
                    'requirement_count': record['requirement_count'],
                    'requirements': record['requirements']
                })
            
            return communities
            
        except Exception as e:
            logger.error(f"Communities query failed: {e}")
            return []
    
    def trace_citation_chain(
        self,
        citation: str,
        max_depth: int = 3
    ) -> List[GraphPath]:
        """
        Trace how citations reference each other.
        
        Args:
            citation: Starting citation
            max_depth: Maximum chain depth
            
        Returns:
            List of citation paths
        """
        cypher = """
        MATCH (start)
        WHERE start.citation_text = $citation 
           OR start.source_doc = $citation
        MATCH path = (start)-[:REFERENCES|MENTIONS*1..{max_depth}]->(target)
        RETURN 
            [node IN nodes(path) | {
                type: labels(node)[0],
                citation: node.citation_text,
                text: node.text
            }] AS nodes,
            [rel IN relationships(path) | type(rel)] AS relationships,
            length(path) AS length
        ORDER BY length ASC
        LIMIT 20
        """.replace('{max_depth}', str(max_depth))
        
        try:
            results = self.client.execute_query(cypher, {
                'citation': citation
            })
            
            paths = []
            for record in results:
                paths.append(GraphPath(
                    nodes=record['nodes'],
                    relationships=record['relationships'],
                    length=record['length']
                ))
            
            return paths
            
        except Exception as e:
            logger.error(f"Citation chain query failed: {e}")
            return []
    
    def analyze_node_importance(
        self,
        node_type: str = 'Requirement',
        limit: int = 20
    ) -> List[Dict]:
        """
        Analyze node importance using degree centrality.
        
        Args:
            node_type: Type of nodes to analyze
            limit: Maximum nodes to return
            
        Returns:
            List of nodes with importance scores
        """
        cypher = f"""
        MATCH (node:{node_type})
        OPTIONAL MATCH (node)-[out_rel]->()
        OPTIONAL MATCH (node)<-[in_rel]-()
        WITH node, 
             count(DISTINCT out_rel) AS out_degree,
             count(DISTINCT in_rel) AS in_degree
        WITH node,
             out_degree + in_degree AS total_degree,
             out_degree,
             in_degree
        WHERE total_degree > 0
        RETURN 
            node.id AS id,
            COALESCE(node.name, node.term, node.text) AS name,
            node.citation_text AS citation,
            total_degree,
            out_degree,
            in_degree
        ORDER BY total_degree DESC
        LIMIT {limit}
        """
        
        try:
            results = self.client.execute_query(cypher)
            
            nodes = []
            for record in results:
                nodes.append({
                    'id': record['id'],
                    'name': record['name'],
                    'citation': record['citation'],
                    'importance_score': record['total_degree'],
                    'outgoing_connections': record['out_degree'],
                    'incoming_connections': record['in_degree']
                })
            
            return nodes
            
        except Exception as e:
            logger.error(f"Node importance query failed: {e}")
            return []


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add project root to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    traversal = GraphTraversal()
    
    # Test: Find important requirements
    print("\nMost Important Requirements:")
    important = traversal.analyze_node_importance(node_type='Requirement', limit=10)
    for node in important:
        print(f"  {node['citation']}: {node['importance_score']} connections")
    
    # Test: Find requirement communities
    print("\nRequirement Communities:")
    communities = traversal.find_communities(min_size=2, limit=5)
    for comm in communities:
        print(f"  Process: {comm['process']} ({comm['requirement_count']} requirements)")

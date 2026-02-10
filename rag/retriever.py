"""
Retriever
Hybrid retrieval combining vector similarity, graph traversal, and keyword matching.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer

from graph.neo4j_client import Neo4jClient
from rag.graph_traversal import GraphTraversal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RetrievedContext:
    """Represents retrieved context from the knowledge graph."""
    text: str
    node_id: str
    node_type: str
    source_doc: str
    section_id: str
    citation_text: str
    relevance_score: float
    retrieval_method: str  # 'vector', 'graph', 'keyword'
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class HybridRetriever:
    """Hybrid retrieval combining multiple strategies."""
    
    def __init__(
        self,
        neo4j_client: Optional[Neo4jClient] = None,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            neo4j_client: Neo4jClient instance
            embedding_model: SentenceTransformer model name
        """
        self.client = neo4j_client or Neo4jClient()
        self.traversal = GraphTraversal(self.client)
        self.embedding_model = SentenceTransformer(embedding_model)
        logger.info(f" HybridRetriever initialized with {embedding_model}")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        include_graph_context: bool = True,
        min_score: float = 0.2
    ) -> List[RetrievedContext]:
        """
        Retrieve relevant context using hybrid approach.
        
        Args:
            query: User query
            top_k: Number of results to return
            include_graph_context: Include graph neighbors
            min_score: Minimum relevance score
            
        Returns:
            List of retrieved contexts
        """
        all_contexts = []
        
        # 1. Vector similarity search (semantic match)
        vector_contexts = self._vector_search(query, limit=top_k)
        all_contexts.extend(vector_contexts)
        
        # 2. Keyword-based retrieval (fast, exact matches using fulltext index)
        keyword_contexts = self._keyword_search(query, limit=top_k)
        all_contexts.extend(keyword_contexts)
        
        # 3. Citation-based retrieval (if query mentions specific document)
        citation_contexts = self._citation_search(query, limit=5)
        all_contexts.extend(citation_contexts)
        
        # 4. Graph traversal (expand context with neighbors)
        if include_graph_context and all_contexts:
            graph_contexts = self._expand_with_graph(all_contexts, max_neighbors=3)
            all_contexts.extend(graph_contexts)
        
        # Deduplicate and rank
        unique_contexts = self._deduplicate_contexts(all_contexts)
        ranked_contexts = sorted(unique_contexts, key=lambda x: x.relevance_score, reverse=True)
        
        # Filter by minimum score
        filtered_contexts = [c for c in ranked_contexts if c.relevance_score >= min_score]
        
        return filtered_contexts[:top_k]

    def _vector_search(self, query: str, limit: int) -> List[RetrievedContext]:
        """Search using vector similarity across node labels."""
        try:
            query_embedding = self.embedding_model.encode(query).tolist()
            
            contexts = []
            labels = ["Requirement", "Definition", "Process", "Exemption"]
            
            for label in labels:
                index_name = f"{label.lower()}_vector_index"
                cypher = f"""
                CALL db.index.vector.queryNodes($index, $limit, $embedding)
                YIELD node, score
                RETURN 
                    node.id AS node_id,
                    labels(node)[0] AS node_type,
                    COALESCE(node.text, node.name, node.term) AS text,
                    node.source_doc AS source_doc,
                    node.section_id AS section_id,
                    node.citation_text AS citation_text,
                    score
                """
                
                results = self.client.execute_query(cypher, {
                    'index': index_name,
                    'limit': limit,
                    'embedding': query_embedding
                })
                
                for record in results:
                    contexts.append(RetrievedContext(
                        text=record['text'],
                        node_id=record['node_id'],
                        node_type=record['node_type'],
                        source_doc=record['source_doc'] or '',
                        section_id=record['section_id'] or '',
                        citation_text=record['citation_text'] or '',
                        relevance_score=float(record['score']),
                        retrieval_method='vector'
                    ))
            
            return contexts
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def _keyword_search(self, query: str, limit: int) -> List[RetrievedContext]:
        """Search using fulltext index matching."""
        # Pre-process query for fulltext search (basic term OR search)
        terms = [t for t in query.replace('?', '').split() if len(t) > 2]
        if not terms:
            return []
        
        search_query = " OR ".join([f"{t}~" for t in terms]) # Fuzziness added
        
        cypher = """
        CALL db.index.fulltext.queryNodes("all_text_index", $query)
        YIELD node, score
        RETURN 
            node.id AS node_id,
            labels(node)[0] AS node_type,
            COALESCE(node.text, node.name, node.term) AS text,
            node.source_doc AS source_doc,
            node.section_id AS section_id,
            node.citation_text AS citation_text,
            score
        LIMIT $limit
        """
        
        try:
            results = self.client.execute_query(cypher, {
                'query': search_query,
                'limit': limit
            })
            
            contexts = []
            for record in results:
                contexts.append(RetrievedContext(
                    text=record['text'],
                    node_id=record['node_id'],
                    node_type=record['node_type'],
                    source_doc=record['source_doc'] or '',
                    section_id=record['section_id'] or '',
                    citation_text=record['citation_text'] or '',
                    relevance_score=min(float(record['score']) / 10.0, 1.0), # Normalize score
                    retrieval_method='keyword'
                ))
            
            return contexts
            
        except Exception as e:
            logger.debug(f"Keyword search failed (probably index not ready): {e}")
            return []
    
    def _citation_search(self, query: str, limit: int) -> List[RetrievedContext]:
        """Search for specific citations mentioned in query."""
        # Extract document names
        doc_patterns = ['MHRA', 'EU GMP', 'UK GMP', 'ICH', 'FDA', 'EMA']
        mentioned_docs = [doc for doc in doc_patterns if doc.lower() in query.lower()]
        
        if not mentioned_docs:
            return []
        
        cypher = """
        MATCH (n)
        WHERE ANY(doc IN $docs WHERE toLower(n.source_doc) CONTAINS toLower(doc))
           OR ANY(doc IN $docs WHERE toLower(n.citation_text) CONTAINS toLower(doc))
        RETURN 
            n.id AS node_id,
            labels(n)[0] AS node_type,
            COALESCE(n.text, n.name, n.term) AS text,
            n.source_doc AS source_doc,
            n.section_id AS section_id,
            n.citation_text AS citation_text
        LIMIT $limit
        """
        
        try:
            results = self.client.execute_query(cypher, {
                'docs': mentioned_docs,
                'limit': limit
            })
            
            contexts = []
            for record in results:
                contexts.append(RetrievedContext(
                    text=record['text'],
                    node_id=record['node_id'],
                    node_type=record['node_type'],
                    source_doc=record['source_doc'] or '',
                    section_id=record['section_id'] or '',
                    citation_text=record['citation_text'] or '',
                    relevance_score=0.8,  # High score for citation matches
                    retrieval_method='citation'
                ))
            
            return contexts
            
        except Exception as e:
            logger.error(f"Citation search failed: {e}")
            return []
    
    def _expand_with_graph(
        self,
        base_contexts: List[RetrievedContext],
        max_neighbors: int
    ) -> List[RetrievedContext]:
        """Expand context using graph traversal."""
        expanded = []
        
        for context in base_contexts[:3]:  # Expand top 3 results
            neighbors = self.traversal.find_neighbors(
                context.node_id,
                max_neighbors=max_neighbors
            )
            
            for neighbor in neighbors:
                expanded.append(RetrievedContext(
                    text=neighbor['text'],
                    node_id=neighbor['id'],
                    node_type=neighbor['type'],
                    source_doc='',
                    section_id='',
                    citation_text='',
                    relevance_score=context.relevance_score * 0.7,  # Decay score
                    retrieval_method='graph',
                    metadata={'relationship': neighbor['relationship']}
                ))
        
        return expanded
    
    def _deduplicate_contexts(
        self,
        contexts: List[RetrievedContext]
    ) -> List[RetrievedContext]:
        """Remove duplicate contexts, keeping highest scored."""
        seen_ids = {}
        
        for context in contexts:
            if context.node_id not in seen_ids:
                seen_ids[context.node_id] = context
            else:
                # Keep higher score
                if context.relevance_score > seen_ids[context.node_id].relevance_score:
                    seen_ids[context.node_id] = context
        
        return list(seen_ids.values())
    
    def retrieve_by_process(self, process_name: str, limit: int = 10) -> List[RetrievedContext]:
        """Retrieve all requirements for a specific process."""
        cypher = """
        MATCH (proc:Process)-[:APPLIES_TO]-(req:Requirement)
        WHERE toLower(proc.name) CONTAINS toLower($process_name)
        RETURN 
            req.id AS node_id,
            'Requirement' AS node_type,
            req.text AS text,
            req.source_doc AS source_doc,
            req.section_id AS section_id,
            req.citation_text AS citation_text,
            req.category AS category,
            req.severity AS severity,
            proc.name AS process_name
        LIMIT $limit
        """
        
        try:
            results = self.client.execute_query(cypher, {
                'process_name': process_name,
                'limit': limit
            })
            
            contexts = []
            for record in results:
                contexts.append(RetrievedContext(
                    text=record['text'],
                    node_id=record['node_id'],
                    node_type=record['node_type'],
                    source_doc=record['source_doc'] or '',
                    section_id=record['section_id'] or '',
                    citation_text=record['citation_text'] or '',
                    relevance_score=0.9,
                    retrieval_method='process',
                    metadata={
                        'process': record['process_name'],
                        'category': record.get('category'),
                        'severity': record.get('severity')
                    }
                ))
            
            return contexts
            
        except Exception as e:
            logger.error(f"Process retrieval failed: {e}")
            return []
    
    def retrieve_definitions(self, term: str) -> List[RetrievedContext]:
        """Retrieve definitions for a term."""
        cypher = """
        MATCH (defn:Definition)
        WHERE toLower(defn.term) CONTAINS toLower($term)
        RETURN 
            defn.id AS node_id,
            defn.term AS term,
            defn.text AS text,
            defn.source_doc AS source_doc,
            defn.citation_text AS citation_text,
            defn.domain AS domain
        LIMIT 5
        """
        
        try:
            results = self.client.execute_query(cypher, {'term': term})
            
            contexts = []
            for record in results:
                contexts.append(RetrievedContext(
                    text=record['text'],
                    node_id=record['node_id'],
                    node_type='Definition',
                    source_doc=record['source_doc'] or '',
                    section_id='',
                    citation_text=record['citation_text'] or '',
                    relevance_score=0.95,
                    retrieval_method='definition',
                    metadata={
                        'term': record['term'],
                        'domain': record.get('domain')
                    }
                ))
            
            return contexts
            
        except Exception as e:
            logger.error(f"Definition retrieval failed: {e}")
            return []


if __name__ == "__main__":
    retriever = HybridRetriever()
    
    # Test queries
    test_queries = [
        "What are the requirements for aseptic filling?",
        "What is the definition of critical process parameter?",
        "MHRA requirements for sterilization"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        contexts = retriever.retrieve(query, top_k=5)
        print(f"Found {len(contexts)} contexts:")
        for i, ctx in enumerate(contexts, 1):
            print(f"  {i}. [{ctx.node_type}] {ctx.text[:100]}...")
            print(f"     Source: {ctx.citation_text} | Score: {ctx.relevance_score:.2f}")

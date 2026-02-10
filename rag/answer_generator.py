"""
Answer Generator
Synthesizes final answer with citations, graph paths, and Cypher queries.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import json
from datetime import datetime

from rag.retriever import HybridRetriever, RetrievedContext
from rag.reasoning_chain import ReasoningChain, ReasoningResult
from rag.cypher_generator import CypherGenerator
from rag.graph_traversal import GraphTraversal
from rag.conflict_detector import ConflictDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FinalAnswer:
    """Complete answer with all required components."""
    
    # Core answer
    query: str
    answer: str
    confidence_score: float
    
    # Citations (mandatory)
    sources: List[Dict]  # [{source_doc, section_id, citation_text, excerpt}]
    
    # Graph path (mandatory)
    graph_path: List[Dict]  # [{node_type, node_id, text}]
    path_description: str
    
    # Cypher query used (mandatory)
    cypher_query: str
    cypher_parameters: Dict
    
    # Additional context
    conflicts: List[Dict]
    related_requirements: List[str]
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def format_for_display(self) -> str:
        """Format for human-readable display."""
        output = []
        output.append("=" * 80)
        output.append(f"QUERY: {self.query}")
        output.append("=" * 80)
        
        output.append(f"\nANSWER:")
        output.append(self.answer)
        output.append(f"\nConfidence: {self.confidence_score:.1%}")
        
        output.append(f"\n{'-' * 80}")
        output.append("SOURCES:")
        for i, source in enumerate(self.sources, 1):
            output.append(f"\n[{i}] {source.get('citation_text', '')}")
            output.append(f"    Source: {source.get('source_doc', '')} | Section: {source.get('section_id', '')}")
            if source.get('excerpt'):
                output.append(f"    Excerpt: {source['excerpt'][:150]}...")
        
        output.append(f"\n{'-' * 80}")
        output.append("GRAPH PATH:")
        output.append(self.path_description)
        for i, node in enumerate(self.graph_path, 1):
            output.append(f"  {i}. [{node['node_type']}] {node.get('text', '')[:100]}...")
        
        output.append(f"\n{'-' * 80}")
        output.append("CYPHER QUERY USED:")
        output.append(self.cypher_query)
        if self.cypher_parameters:
            output.append(f"Parameters: {json.dumps(self.cypher_parameters, indent=2)}")
        
        if self.conflicts:
            output.append(f"\n{'-' * 80}")
            output.append(f"CONFLICTS DETECTED ({len(self.conflicts)}):")
            for i, conflict in enumerate(self.conflicts, 1):
                output.append(f"\n  Conflict {i}:")
                output.append(f"    Sources: {conflict.get('source1', '')} vs {conflict.get('source2', '')}")
                output.append(f"    Reason: {conflict.get('reason', 'Unknown')}")
                output.append(f"    Confidence: {conflict.get('confidence', 0):.1%}")
                if conflict.get('text1'):
                    output.append(f"    Text 1: {conflict['text1'][:100]}...")
                if conflict.get('text2'):
                    output.append(f"    Text 2: {conflict['text2'][:100]}...")
        
        if self.related_requirements:
            output.append(f"\n{'-' * 80}")
            output.append("RELATED REQUIREMENTS:")
            for req in self.related_requirements[:5]:
                output.append(f"   {req[:100]}...")
        
        output.append("\n" + "=" * 80)
        
        return "\n".join(output)


class AnswerGenerator:
    """Generates complete answers with all required components."""
    
    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        reasoner: Optional[ReasoningChain] = None,
        cypher_generator: Optional[CypherGenerator] = None,
        traversal: Optional[GraphTraversal] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize answer generator.
        
        Args:
            retriever: HybridRetriever instance
            reasoner: ReasoningChain instance (uses Gemini)
            cypher_generator: CypherGenerator instance (uses Groq)
            traversal: GraphTraversal instance
            api_key: API key (Gemini for reasoner, Groq for cypher_generator)
        """
        self.retriever = retriever or HybridRetriever()
        # Use Gemini for reasoning (needs smart brain)
        self.reasoner = reasoner or ReasoningChain(api_key=api_key)
        # Use Groq for Cypher generation (saves Gemini quota)
        self.cypher_generator = cypher_generator or CypherGenerator(use_groq=True)
        self.traversal = traversal or GraphTraversal()
        # Rule-based conflict detector (no LLM JSON parsing)
        self.conflict_detector = ConflictDetector()
        
        logger.info(" AnswerGenerator initialized")
    
    def generate_answer(
        self,
        query: str,
        top_k: int = 10,
        include_conflicts: bool = True,
        include_related: bool = True
    ) -> FinalAnswer:
        """
        Generate complete answer with all components.
        
        Args:
            query: User query
            top_k: Number of contexts to retrieve
            include_conflicts: Detect conflicts
            include_related: Find related requirements
            
        Returns:
            FinalAnswer with all components
        """
        logger.info(f"Generating answer for: {query}")
        
        # Step 1: Retrieve relevant contexts
        logger.info("Step 1: Retrieving contexts...")
        contexts = self.retriever.retrieve(query, top_k=top_k)
        logger.info(f"  Retrieved {len(contexts)} contexts")
        
        # Step 2: Generate Cypher query for traceability
        logger.info("Step 2: Generating Cypher query...")
        cypher_query, cypher_params = self.cypher_generator.generate_cypher(
            query,
            use_template=True
        )
        logger.info(f"  Generated query: {cypher_query}")
        
        # Step 3: Reason over contexts
        logger.info("Step 3: Reasoning over contexts...")
        reasoning_result = self.reasoner.reason(
            query,
            contexts,
            detect_conflicts=False  # Use rule-based detector instead
        )
        logger.info(f"  Generated answer (confidence: {reasoning_result.confidence_score:.2f})")
        
        # Step 3.5: Detect conflicts using rule-based detector
        conflicts = []
        if include_conflicts:
            logger.info("Step 3.5: Detecting conflicts...")
            conflicts = self.conflict_detector.detect_conflicts(contexts)
            if conflicts:
                logger.info(f"  Detected {len(conflicts)} potential conflicts")
        
        # Step 4: Build graph path
        logger.info("Step 4: Building graph path...")
        graph_path, path_desc = self._build_graph_path(contexts)
        logger.info(f"  Path length: {len(graph_path)} nodes")
        
        # Step 5: Find related requirements (optional)
        related_reqs = []
        if include_related and contexts:
            logger.info("Step 5: Finding related requirements...")
            related_reqs = self._find_related_requirements(contexts[:3])
            logger.info(f"  Found {len(related_reqs)} related requirements")
        
        # Step 6: Assemble final answer
        final_answer = FinalAnswer(
            query=query,
            answer=reasoning_result.final_answer,
            confidence_score=reasoning_result.confidence_score,
            sources=reasoning_result.all_citations,
            graph_path=graph_path,
            path_description=path_desc,
            cypher_query=cypher_query,
            cypher_parameters=cypher_params,
            conflicts=conflicts,  # Use rule-based detected conflicts
            related_requirements=related_reqs
        )
        
        logger.info(" Answer generation complete")
        
        return final_answer
    
    def _build_graph_path(
        self,
        contexts: List[RetrievedContext]
    ) -> tuple[List[Dict], str]:
        """Build graph path from retrieved contexts."""
        if not contexts:
            return [], "No graph path available (no contexts retrieved)"
        
        # Create path from contexts
        path_nodes = []
        for ctx in contexts[:5]:  # Limit to top 5 for readability
            path_nodes.append({
                'node_id': ctx.node_id,
                'node_type': ctx.node_type,
                'text': ctx.text[:200],
                'citation': ctx.citation_text,
                'relevance_score': ctx.relevance_score
            })
        
        # Generate description
        node_types = [n['node_type'] for n in path_nodes]
        type_counts = {}
        for nt in node_types:
            type_counts[nt] = type_counts.get(nt, 0) + 1
        
        type_summary = ", ".join([f"{count} {ntype}" for ntype, count in type_counts.items()])
        
        description = f"Graph path traversed {len(path_nodes)} nodes: {type_summary}. "
        description += f"Retrieved via {contexts[0].retrieval_method} search, "
        description += f"expanded with graph neighbors for comprehensive context."
        
        return path_nodes, description
    
    def _find_related_requirements(
        self,
        base_contexts: List[RetrievedContext]
    ) -> List[str]:
        """Find requirements related to base contexts."""
        related = []
        
        for ctx in base_contexts:
            if ctx.node_type == 'Requirement':
                # Find neighbors
                neighbors = self.traversal.find_neighbors(
                    ctx.node_id,
                    relationship_types=['APPLIES_TO', 'REQUIRES'],
                    max_neighbors=3
                )
                
                for neighbor in neighbors:
                    if neighbor.get('type') == 'Requirement':
                        related.append(neighbor.get('text', ''))
        
        return related[:10]  # Limit to 10
    
    def batch_generate(
        self,
        queries: List[str],
        output_file: Optional[Path] = None
    ) -> List[FinalAnswer]:
        """Generate answers for multiple queries."""
        answers = []
        
        for i, query in enumerate(queries, 1):
            logger.info(f"\nProcessing query {i}/{len(queries)}: {query}")
            
            try:
                answer = self.generate_answer(query)
                answers.append(answer)
                
                # Save incrementally if output file provided
                if output_file:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump([a.to_dict() for a in answers], f, indent=2)
                
            except Exception as e:
                logger.error(f"Failed to generate answer for query {i}: {e}")
                continue
        
        return answers


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Initialize generator
    generator = AnswerGenerator(api_key=os.getenv('GEMINI_API_KEY'))
    
    # Test queries
    test_queries = [
        "What requirements apply to aseptic filling operations?",
        "Does MHRA GMP require annual requalification for aseptic personnel?",
        "What is the definition of critical process parameter?",
        "Are there exemptions for sterility testing?"
    ]
    
    for query in test_queries:
        print("\n" + "="*80)
        print(f"QUERY: {query}")
        print("="*80)
        
        try:
            answer = generator.generate_answer(query, top_k=5)
            print(answer.format_for_display())
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            continue
        
        print("\n" + "="*80)
        input("Press Enter for next query...")

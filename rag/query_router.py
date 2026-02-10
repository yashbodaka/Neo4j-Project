"""
Query Router
Intelligently routes queries to optimized retrieval strategies based on query type.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import os
from typing import Dict, Tuple
from google import genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryRouter:
    """Routes queries to appropriate retrieval strategies."""
    
    QUERY_TYPES = {
        'definition': {
            'description': 'Asks for the meaning or definition of a term',
            'top_k': 3,
            'node_focus': ['Definition', 'Process'],
            'examples': ['What is aseptic filling?', 'Define critical process parameter']
        },
        'requirement': {
            'description': 'Asks about requirements, regulations, or compliance',
            'top_k': 7,
            'node_focus': ['Requirement'],
            'examples': ['What are the requirements for sterilization?', 'What documentation is required?']
        },
        'listing': {
            'description': 'Asks for a list or multiple items',
            'top_k': 10,
            'node_focus': ['Requirement', 'Process'],
            'examples': ['List all temperature monitoring requirements', 'What are the steps in batch release?']
        },
        'comparative': {
            'description': 'Compares two or more things',
            'top_k': 6,
            'node_focus': ['Definition', 'Requirement'],
            'examples': ['Difference between Grade A and Grade B cleanrooms?']
        },
        'cross_reference': {
            'description': 'Asks which regulations or sources mention something',
            'top_k': 8,
            'node_focus': ['Requirement', 'CrossReference'],
            'examples': ['Which regulations mention batch release?']
        },
        'conditional': {
            'description': 'Asks about when, where, or under what conditions',
            'top_k': 5,
            'node_focus': ['Requirement', 'Process'],
            'examples': ['When is requalification required?', 'Where must environmental monitoring occur?']
        },
        'procedural': {
            'description': 'Asks about steps, procedures, or how to do something',
            'top_k': 6,
            'node_focus': ['Process', 'Requirement'],
            'examples': ['What are the steps in validation?', 'How to perform batch release?']
        },
        'exemption': {
            'description': 'Asks about exemptions or exceptions',
            'top_k': 5,
            'node_focus': ['Exemption', 'Requirement'],
            'examples': ['Are there exemptions for investigational products?']
        },
        'complex': {
            'description': 'Multi-entity or relationship questions',
            'top_k': 10,
            'node_focus': ['Requirement', 'Process', 'Definition'],
            'examples': ['How do personnel qualifications relate to aseptic processing?']
        }
    }
    
    CLASSIFICATION_PROMPT = """You are a query classifier for a pharmaceutical regulatory knowledge system.

Query Types:
{type_descriptions}

User Query: "{query}"

Classify this query into ONE of the above types. Consider:
- Is it asking for a definition?
- Is it asking about requirements or compliance?
- Is it asking for a list?
- Is it comparing things?
- Is it asking about cross-references?
- Is it asking about conditions (when/where)?
- Is it asking about procedures (how/steps)?
- Is it asking about exemptions?
- Is it a complex multi-entity question?

Respond with ONLY the type name (e.g., "definition", "requirement", "listing", etc.)
"""
    
    def __init__(self):
        """Initialize query router with Gemini 3 Flash Preview."""
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.client = genai.Client(api_key=self.api_key)
        self.model = "gemini-3-flash-preview"
        logger.info(f"✓ QueryRouter initialized with {self.model}")
    
    def classify_query(self, query: str) -> str:
        """
        Classify query type using LLM.
        
        Args:
            query: User query
            
        Returns:
            Query type string
        """
        # Build type descriptions
        type_desc = "\n".join([
            f"- {name}: {info['description']}\n  Examples: {', '.join(info['examples'][:2])}"
            for name, info in self.QUERY_TYPES.items()
        ])
        
        prompt = self.CLASSIFICATION_PROMPT.format(
            type_descriptions=type_desc,
            query=query
        )
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    'temperature': 0.1,
                    'max_output_tokens': 50
                }
            )
            
            query_type = response.text.strip().lower()
            
            # Validate response
            if query_type not in self.QUERY_TYPES:
                logger.warning(f"Invalid query type '{query_type}', defaulting to 'complex'")
                query_type = 'complex'
            
            logger.info(f"Query classified as: {query_type}")
            return query_type
            
        except Exception as e:
            logger.error(f"Classification failed: {e}. Defaulting to 'complex'")
            return 'complex'
    
    def get_retrieval_config(self, query: str) -> Dict:
        """
        Get optimized retrieval configuration for query.
        
        Args:
            query: User query
            
        Returns:
            Config dict with top_k and node_focus
        """
        query_type = self.classify_query(query)
        config = self.QUERY_TYPES[query_type]
        
        return {
            'query_type': query_type,
            'top_k': config['top_k'],
            'node_focus': config['node_focus'],
            'description': config['description']
        }
    
    def route_query(self, query: str) -> Tuple[str, Dict]:
        """
        Route query and return type with config.
        
        Args:
            query: User query
            
        Returns:
            Tuple of (query_type, config)
        """
        config = self.get_retrieval_config(query)
        return config['query_type'], config


if __name__ == "__main__":
    # Test router
    router = QueryRouter()
    
    test_queries = [
        "What is aseptic filling?",
        "What are the requirements for sterilization?",
        "List all temperature monitoring requirements",
        "When is requalification required?",
        "Which regulations mention batch release?"
    ]
    
    print("\n" + "="*80)
    print("QUERY ROUTER TEST")
    print("="*80)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        query_type, config = router.route_query(query)
        print(f"  Type: {query_type}")
        print(f"  top_k: {config['top_k']}")
        print(f"  Node focus: {config['node_focus']}")

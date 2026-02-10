"""
Cypher Generator
LLM-powered Cypher query generation from natural language.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import os
from typing import Dict, List, Optional, Tuple
import json
import re
from google import genai

from rag.cypher_templates import CypherTemplates, QueryType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CypherGenerator:
    """Generates Cypher queries from natural language using Gemini."""
    
    SCHEMA_CONTEXT = """
    Neo4j Graph Schema:
    
    Node Types:
    - Regulation: {id, name, issued_by, effective_date, version, source_doc}
    - Requirement: {id, name, text, source_doc, section_id, citation_text, category, severity, scope, mandatory}
    - Definition: {id, term, text, source_doc, section_id, citation_text, domain}
    - Exemption: {id, text, source_doc, section_id, citation_text, condition, scope}
    - Process: {id, name, text, source_doc, section_id, citation_text, process_type}
    - CrossReference: {id, text, source_doc, target_doc, target_section, reference_type}
    
    Relationship Types:
    - (:Regulation)-[:CONTAINS]->(:Requirement)
    - (:Regulation)-[:SUPERSEDES]->(:Regulation)
    - (:Requirement)-[:APPLIES_TO]->(:Process)
    - (:Definition)-[:DEFINES]->(:Requirement)
    - (:Requirement)-[:REQUIRES]->(:Requirement)
    - (:Requirement)-[:REFERENCES]->(:CrossReference)
    - (:Requirement)-[:CONFLICTS_WITH]->(:Requirement)
    - (:CrossReference)-[:MENTIONS]->(:Requirement)
    - (:Exemption)-[:EXEMPT_FROM]->(:Requirement)
    
    Common Query Patterns:
    1. Find requirements for process: MATCH (req:Requirement)-[:APPLIES_TO]->(proc:Process)
    2. Find regulation contents: MATCH (reg:Regulation)-[:CONTAINS]->(req:Requirement)
    3. Check version history: MATCH (new:Regulation)-[:SUPERSEDES]->(old:Regulation)
    4. Check applicability: MATCH (req:Requirement) WHERE req.scope CONTAINS $keyword
    5. Find definitions: MATCH (defn:Definition) WHERE defn.term CONTAINS $term
    6. Detect conflicts: MATCH (req1:Requirement)-[:CONFLICTS_WITH]->(req2:Requirement)
    7. Trace dependencies: MATCH path = (req:Requirement)-[:REQUIRES*]->(dep:Requirement)
    """
    
    CYPHER_GENERATION_PROMPT = """You are a Neo4j Cypher query expert for regulatory compliance knowledge graphs.

Given a natural language question, generate a valid Cypher query that retrieves the required information.

Context:
{schema}

Question: {question}

Requirements:
1. Return ONLY valid Cypher syntax (no explanations)
2. Use parameterized queries with $ notation when possible
3. Include LIMIT clause (default 20)
4. Use OPTIONAL MATCH for relationships that may not exist
5. Return meaningful property names
6. Use case-insensitive matching with toLower()

Generate Cypher query:"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile", use_groq: bool = True):
        """
        Initialize Cypher generator with Groq (primary) or Gemini fallback.
        
        Args:
            api_key: API key (Groq or Gemini)
            model: Model name
            use_groq: Whether to use Groq API (default True)
        """
        self.use_groq = use_groq
        
        if use_groq:
            try:
                from groq import Groq
                self.groq_api_key = api_key or os.getenv("GROQ_API_KEY")
                self.groq_client = Groq(api_key=self.groq_api_key)
                self.model = model
                logger.info(f"✓ CypherGenerator initialized with Groq: {self.model}")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq: {e}. Falling back to Gemini.")
                self.use_groq = False
        
        if not self.use_groq:
            self.gemini_api_key = api_key or os.getenv("GEMINI_API_KEY")
            self.client = genai.Client(api_key=self.gemini_api_key)
            self.model_hierarchy = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
            logger.info(f"✓ CypherGenerator initialized with Gemini: {self.model_hierarchy}")
        
        self.templates = CypherTemplates()
    
    def generate_cypher(
        self, 
        question: str,
        parameters: Optional[Dict] = None,
        use_template: bool = True
    ) -> Tuple[str, Dict]:
        """
        Generate Cypher query from natural language.
        
        Args:
            question: Natural language question
            parameters: Optional query parameters
            use_template: Try template matching first
            
        Returns:
            Tuple of (cypher_query, parameters)
        """
        # Try template matching first
        if use_template:
            template_result = self._try_template_match(question, parameters)
            if template_result:
                return template_result
        
        # Fall back to LLM generation
        return self._generate_with_llm(question, parameters)
    
    def _try_template_match(
        self, 
        question: str,
        parameters: Optional[Dict]
    ) -> Optional[Tuple[str, Dict]]:
        """Try to match question to pre-defined template."""
        question_lower = question.lower()
        params = parameters or {}
        
        # Pattern matching for common queries
        if any(word in question_lower for word in ['requirement', 'apply', 'applicable']):
            if 'process' in question_lower:
                # Extract process name
                process_match = re.search(r'process[es]*\s+(?:of\s+)?([a-z\s]+)', question_lower)
                if process_match:
                    params['process_name'] = process_match.group(1).strip()
                    params['limit'] = params.get('limit', 20)
                    return (CypherTemplates.REQUIREMENTS_FOR_PROCESS, params)
        
        elif any(word in question_lower for word in ['definition', 'define', 'what is', 'what are']):
            # Extract term
            term_match = re.search(r'(?:what is|define|definition of)\s+([a-z\s]+)', question_lower)
            if term_match:
                params['term'] = term_match.group(1).strip()
                params['limit'] = params.get('limit', 10)
                return (CypherTemplates.TERM_DEFINITIONS, params)
        
        elif any(word in question_lower for word in ['cross-reference', 'reference', 'cite']):
            params['source_doc'] = params.get('source_doc', '')
            params['target_doc'] = params.get('target_doc', '')
            params['limit'] = params.get('limit', 20)
            return (CypherTemplates.CROSS_REFERENCES, params)
        
        elif any(word in question_lower for word in ['conflict', 'contradict', 'inconsistent']):
            params['keyword'] = params.get('keyword', '')
            params['limit'] = params.get('limit', 20)
            return (CypherTemplates.DETECT_CONFLICTS, params)
        
        elif any(word in question_lower for word in ['exemption', 'exception', 'exempt']):
            params['keyword'] = params.get('keyword', '')
            params['limit'] = params.get('limit', 20)
            return (CypherTemplates.FIND_EXEMPTIONS, params)
        
        return None
    
    def _generate_with_llm(
        self, 
        question: str,
        parameters: Optional[Dict]
    ) -> Tuple[str, Dict]:
        """Generate Cypher using LLM."""
        prompt = self.CYPHER_GENERATION_PROMPT.format(
            schema=self.SCHEMA_CONTEXT,
            question=question
        )
        
        try:
            if self.use_groq:
                # Use Groq for Cypher generation
                chat_completion = self.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model,
                    temperature=0.1,
                    max_tokens=1500
                )
                cypher = chat_completion.choices[0].message.content.strip()
                logger.info(f"✓ Generated Cypher with Groq ({len(cypher)} chars)")
            else:
                # Gemini fallback
                last_error = None
                for model_name in self.model_hierarchy:
                    try:
                        logger.info(f"Attempting Cypher generation with {model_name}...")
                        response = self.client.models.generate_content(
                            model=model_name,
                            contents=prompt,
                            config={
                                'temperature': 0.1,
                                'max_output_tokens': 1000
                            }
                        )
                        cypher = response.text.strip()
                        logger.info(f"✓ Generated Cypher with {model_name} ({len(cypher)} chars)")
                        break
                    except Exception as e:
                        last_error = e
                        logger.warning(f"Model {model_name} failed: {e}. Trying next...")
                        continue
                else:
                    raise last_error
            
            # Clean up response
            cypher = self._clean_cypher(cypher)
            
            # Extract parameters from question if not provided
            params = parameters or self._extract_parameters(question)
            params['limit'] = params.get('limit', 20)
            
            return (cypher, params)
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Fallback to generic search
            return self._fallback_query(question, parameters)
    
    def _clean_cypher(self, cypher: str) -> str:
        """Clean and validate Cypher query."""
        # Remove markdown code blocks and internal 'cypher' markers
        cypher = re.sub(r'```(?:cypher)?', '', cypher, flags=re.IGNORECASE)
        cypher = re.sub(r'```', '', cypher)
        
        # Remove leading 'cypher' word if it exists independently
        cypher = re.sub(r'^\s*cypher\s+', '', cypher, flags=re.IGNORECASE)
        
        # Remove trailing notes or explanations (often starts with "Note:" or "\n\n")
        cypher = cypher.split("\n\n")[0]
        cypher = cypher.split("Note:")[0]
        
        # Remove leading/trailing whitespace
        cypher = cypher.strip()
        
        # Ensure it starts with valid Cypher keyword
        valid_starts = ['MATCH', 'CREATE', 'MERGE', 'WITH', 'UNWIND', 'CALL', 'OPTIONAL']
        if not any(cypher.upper().startswith(keyword) for keyword in valid_starts):
            logger.warning(f"Generated query doesn't start with valid Cypher keyword: {cypher[:50]}...")
        
        return cypher
    
    def _extract_parameters(self, question: str) -> Dict:
        """Extract parameters from question text."""
        params = {}
        
        # Extract quoted strings as keywords
        quoted = re.findall(r'"([^"]+)"', question)
        if quoted:
            params['keyword'] = quoted[0]
        
        # Extract common parameter patterns
        doc_match = re.search(r'(MHRA|EU GMP|UK GMP|ICH)', question, re.IGNORECASE)
        if doc_match:
            params['source_doc'] = doc_match.group(1)
        
        return params
    
    def _fallback_query(
        self, 
        question: str,
        parameters: Optional[Dict]
    ) -> Tuple[str, Dict]:
        """Generate fallback query for general search."""
        params = parameters or {}
        params['search_term'] = question
        params['limit'] = params.get('limit', 20)
        
        return (CypherTemplates.SEARCH_REQUIREMENTS, params)
    
    def validate_cypher(self, cypher: str) -> bool:
        """
        Basic validation of Cypher syntax.
        
        Args:
            cypher: Cypher query string
            
        Returns:
            True if valid
        """
        # Check for balanced parentheses
        if cypher.count('(') != cypher.count(')'):
            return False
        
        # Check for balanced brackets
        if cypher.count('[') != cypher.count(']'):
            return False
        
        # Check for balanced braces
        if cypher.count('{') != cypher.count('}'):
            return False
        
        # Must contain MATCH or CREATE
        if not any(keyword in cypher.upper() for keyword in ['MATCH', 'CREATE', 'MERGE']):
            return False
        
        return True


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    generator = CypherGenerator(
        api_key=os.getenv('GEMINI_API_KEY'),
        model='gemini-3-flash-preview'
    )
    
    # Test queries
    test_questions = [
        "What requirements apply to sterilization process?",
        "Define the term 'critical process parameter'",
        "Find cross-references between MHRA and EU GMP",
        "Are there any conflicting requirements about temperature control?"
    ]
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        cypher, params = generator.generate_cypher(question)
        print(f"Cypher: {cypher[:200]}...")
        print(f"Params: {params}")

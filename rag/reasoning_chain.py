"""
Reasoning Chain
Multi-step reasoning over retrieved contexts with citation tracking.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from google import genai

from rag.retriever import RetrievedContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)





@dataclass
class ReasoningStep:
    """Represents a step in the reasoning chain."""
    question: str
    contexts_used: List[str]
    reasoning: str
    conclusion: str
    citations: List[str]
    confidence: float = 0.8


@dataclass
class ReasoningResult:
    """Final result of reasoning chain."""
    query: str
    steps: List[ReasoningStep]
    final_answer: str
    all_citations: List[Dict]
    conflicts_detected: List[Dict] = field(default_factory=list)
    confidence_score: float = 0.8


class ReasoningChain:
    """Multi-step reasoning over regulatory contexts."""
    
    REASONING_PROMPT = """You are an expert regulatory compliance analyst specializing in pharmaceutical GxP regulations.

User Question: {query}

Retrieved Context:
{contexts}

Your task is to:
1. Analyze the retrieved contexts carefully
2. Identify relevant requirements, definitions, and exemptions
3. Detect any conflicts or contradictions between sources
4. Provide a comprehensive answer with exact citations

Requirements for your response:
- Base your answer ONLY on the provided context
- Cite specific sources using format: [Source: <document> <section>]
- If information is insufficient, explicitly state what is missing
- Highlight any conflicts between different regulations
- Use clear, precise regulatory language

Provide your response in this JSON format:
{{
  "answer": "Your comprehensive answer here with [citations]",
  "key_points": [
    "Point 1 with [citation]",
    "Point 2 with [citation]"
  ],
  "conflicts": [
    {{"source1": "citation", "source2": "citation", "description": "conflict description"}}
  ],
  "confidence": 0.85,
  "missing_info": "What additional information would help if any"
}}
"""
    
    CONFLICT_DETECTION_PROMPT = """Analyze these regulatory requirements for conflicts or contradictions:

Requirements:
{requirements}

Identify any:
1. Direct contradictions (one requires X, another forbids X)
2. Incompatible conditions (mutually exclusive requirements)
3. Version conflicts (older vs newer regulations)
4. Scope conflicts (overlapping but different rules)

Return JSON:
{{
  "conflicts": [
    {{
      "type": "contradiction|incompatible|version|scope",
      "requirement1": {{"text": "...", "source": "..."}},
      "requirement2": {{"text": "...", "source": "..."}},
      "description": "Explain the conflict",
      "severity": "high|medium|low"
    }}
  ]
}}
"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = None):
        """
        Initialize reasoning chain with Gemini model hierarchy.
        
        Args:
            api_key: Gemini API key
            model: Primary model name (default: gemini-3-flash-preview)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        # Model hierarchy with automatic fallback
        # Keep only top 2 models to save quota
        self.model_hierarchy = [
            "gemini-3-flash-preview",
            "gemini-2.5-flash"
        ]
        
        self.client = genai.Client(api_key=self.api_key)
        logger.info(f"✓ ReasoningChain initialized with Gemini hierarchy: {self.model_hierarchy}")
    
    def reason(
        self,
        query: str,
        contexts: List[RetrievedContext],
        detect_conflicts: bool = False
    ) -> ReasoningResult:
        """
        Perform multi-step reasoning over contexts.
        
        Args:
            query: User query
            contexts: Retrieved contexts
            detect_conflicts: Whether to detect conflicts (disabled by default)
            
        Returns:
            ReasoningResult with answer and citations
        """
        # Format contexts for prompt
        context_text = self._format_contexts(contexts)
        
        # Main reasoning step
        reasoning_response = self._generate_reasoning(query, context_text)
        
        # Extract structured result
        result = self._parse_reasoning_response(reasoning_response, query, contexts)
        
        # Conflict detection disabled due to JSON parsing issues
        # if detect_conflicts and len(contexts) > 1:
        #     conflicts = self._detect_conflicts(contexts)
        #     result.conflicts_detected = conflicts
        
        return result
    
    def _format_contexts(self, contexts: List[RetrievedContext]) -> str:
        """Format contexts for prompt."""
        formatted = []
        
        for i, ctx in enumerate(contexts, 1):
            formatted.append(f"""
Context {i}:
Type: {ctx.node_type}
Source: {ctx.citation_text or ctx.source_doc}
Text: {ctx.text}
---""")
        
        return "\n".join(formatted)
    
    def _generate_reasoning(self, query: str, contexts: str) -> str:
        """Generate reasoning using LLM with automatic model fallback."""
        prompt = self.REASONING_PROMPT.format(
            query=query,
            contexts=contexts
        )
        
        last_error = None
        # Try each model in hierarchy
        for model_name in self.model_hierarchy:
            try:
                logger.info(f"Attempting reasoning with {model_name}...")
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={
                        'temperature': 0.2,
                        'max_output_tokens': 4096
                    }
                )
                logger.info(f"✓ Reasoning generated successfully with {model_name}")
                return response.text
            except Exception as e:
                last_error = e
                logger.warning(f"Model {model_name} failed: {e}. Trying next model...")
                continue
        
        # All models failed
        logger.error(f"All models failed. Last error: {last_error}")
        return "{}"
    
    def _parse_reasoning_response(
        self,
        response_text: str,
        query: str,
        contexts: List[RetrievedContext]
    ) -> ReasoningResult:
        """Parse JSON reasoning response."""
        import json
        import re
        
        try:
            # Extract JSON from markdown
            json_text = response_text.strip()
            if '```json' in json_text:
                json_text = json_text.split('```json')[1].split('```')[0].strip()
            elif '```' in json_text:
                json_text = json_text.split('```')[1].split('```')[0].strip()
            
            data = json.loads(json_text)
            
            # Extract citations from contexts
            all_citations = []
            for ctx in contexts:
                all_citations.append({
                    'source_doc': ctx.source_doc,
                    'section_id': getattr(ctx, 'section_id', ''),
                    'citation_text': ctx.citation_text,
                    'node_type': ctx.node_type,
                    'text_excerpt': ctx.text[:200]
                })
            
            # Create reasoning step
            step = ReasoningStep(
                question=query,
                contexts_used=[ctx.citation_text for ctx in contexts if ctx.citation_text],
                reasoning=data.get('answer', ''),
                conclusion=data.get('answer', ''),
                citations=data.get('key_points', []),
                confidence=data.get('confidence', 0.8)
            )
            
            return ReasoningResult(
                query=query,
                steps=[step],
                final_answer=data.get('answer', ''),
                all_citations=all_citations,
                conflicts_detected=data.get('conflicts', []),
                confidence_score=data.get('confidence', 0.8)
            )
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse reasoning JSON: {e}")
            # Fallback: use raw text
            return ReasoningResult(
                query=query,
                steps=[],
                final_answer=response_text,
                all_citations=[{
                    'source_doc': ctx.source_doc,
                    'section_id': getattr(ctx, 'section_id', ''),
                    'citation_text': ctx.citation_text,
                    'text_excerpt': ctx.text[:200]
                } for ctx in contexts],
                confidence_score=0.6
            )
        except Exception as e:
            logger.error(f"Failed to parse reasoning response: {e}")
            return ReasoningResult(
                query=query,
                steps=[],
                final_answer="Unable to generate answer from retrieved contexts.",
                all_citations=[],
                confidence_score=0.0
            )
    
    def _detect_conflicts(self, contexts: List[RetrievedContext]) -> List[Dict]:
        """Detect conflicts between requirements."""
        # Filter to only Requirements
        requirements = [ctx for ctx in contexts if ctx.node_type == 'Requirement']
        
        if len(requirements) < 2:
            return []
        
        # Format requirements for prompt
        req_text = "\n\n".join([
            f"Requirement {i+1}:\nSource: {req.citation_text}\nText: {req.text}"
            for i, req in enumerate(requirements)
        ])
        
        prompt = self.CONFLICT_DETECTION_PROMPT.format(requirements=req_text)
        
        # Try each model in hierarchy
        last_error = None
        for model_name in self.model_hierarchy:
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={'temperature': 0.1, 'max_output_tokens': 2048}
                )
                response_text = response.text
                
                # Parse conflicts
                import json
                json_text = response_text.strip()
                if '```json' in json_text:
                    json_text = json_text.split('```json')[1].split('```')[0].strip()
                
                data = json.loads(json_text)
                return data.get('conflicts', [])
                
            except Exception as e:
                last_error = e
                logger.warning(f"Model {model_name} failed for conflict detection: {e}. Trying next model...")
                continue
        
        # All models failed
        logger.warning(f"Conflict detection failed with all models. Last error: {last_error}")
        return []
    
    def explain_reasoning(self, result: ReasoningResult) -> str:
        """Generate human-readable explanation of reasoning."""
        explanation = f"Query: {result.query}\n\n"
        explanation += f"Answer: {result.final_answer}\n\n"
        
        if result.all_citations:
            explanation += "Sources Used:\n"
            for i, citation in enumerate(result.all_citations, 1):
                explanation += f"  {i}. {citation['citation_text']}\n"
        
        if result.conflicts_detected:
            explanation += f"\n  Conflicts Detected: {len(result.conflicts_detected)}\n"
            for conflict in result.conflicts_detected:
                explanation += f"  - {conflict.get('description', 'Unknown conflict')}\n"
        
        explanation += f"\nConfidence: {result.confidence_score:.1%}"
        
        return explanation


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from rag.retriever import HybridRetriever
    
    load_dotenv()
    
    # Test reasoning chain
    retriever = HybridRetriever()
    reasoner = ReasoningChain(
        api_key=os.getenv('GEMINI_API_KEY'),
        model='gemini-3-flash-preview'
    )
    
    query = "What are the requirements for aseptic filling operations?"
    
    print(f"Query: {query}\n")
    
    # Retrieve contexts
    contexts = retriever.retrieve(query, top_k=5)
    print(f"Retrieved {len(contexts)} contexts\n")
    
    # Reason over contexts
    result = reasoner.reason(query, contexts, detect_conflicts=True)
    
    # Print explanation
    print(reasoner.explain_reasoning(result))

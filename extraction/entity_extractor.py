"""
Entity Extractor
Extracts entities from document chunks using Gemini LLM.
"""

import sys
import logging
import json
import jsonlines
import os
import time
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Raised when API rate limit is hit after all retries."""
    pass


class EmptyExtractionError(Exception):
    """Raised when extraction returns 0 entities on substantial text."""
    pass


@dataclass
class ExtractedEntity:
    """Represents an extracted entity."""
    
    entity_type: str  # Requirement, Definition, Exemption, Process, CrossReference
    text: str
    source_chunk_id: str
    source_doc: str
    section_id: str
    section_heading: str
    properties: Dict
    confidence: float = 1.0
    extracted_at: Optional[str] = None
    
    def __post_init__(self):
        if self.extracted_at is None:
            self.extracted_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class GroqClient:
    """Client for Groq API with llama-3.1-8b-instant model."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.1-8b-instant"):
        """
        Initialize Groq client.
        
        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY env var)
            model: Model to use (default: llama-3.3-70b-versatile)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model_name = model
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Groq client."""
        if not self.api_key:
            logger.error("GROQ_API_KEY not found in environment variables")
            return
        
        try:
            from groq import Groq
            
            self.client = Groq(api_key=self.api_key)
            
            logger.info(f"✓ Groq client initialized with model: {self.model_name}")
            
        except ImportError:
            logger.error("groq not installed. Install with: pip install groq")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate(self, prompt: str, temperature: float = 0.1) -> str:
        """
        Generate response from Groq.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (lower = more deterministic)
            
        Returns:
            Generated text response
        """
        if not self.client:
            raise ValueError("Groq client not initialized")
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model_name,
                temperature=temperature,
                max_tokens=3072,  # Balanced: allows complete JSON while staying under 6K tokens/min with 3s delays
                top_p=0.95
            )
            
            return chat_completion.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            raise


class GeminiClient:
    """Client for Google Gemini API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
            model: Model to use (default: gemini-2.5-flash)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model
        self.client = None
        self.model = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Gemini client."""
        if not self.api_key:
            logger.error("GEMINI_API_KEY not found in environment variables")
            return
        
        try:
            # Use new google.genai package (google-generativeai is deprecated)
            from google import genai
            
            # Configure client
            self.client = genai.Client(api_key=self.api_key)
            self.model = self.model_name
            
            logger.info(f"✓ Gemini client initialized with model: {self.model_name}")
            
        except ImportError:
            logger.error("google-genai not installed. Install with: pip install google-genai")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate(self, prompt: str, temperature: float = 0.1) -> str:
        """
        Generate response from Gemini.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (lower = more deterministic)
            
        Returns:
            Generated text response
        """
        if not self.client or not self.model:
            raise ValueError("Gemini client not initialized")
        
        try:
            # Use new google.genai API
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "temperature": temperature,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 8192,
                }
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            raise


class EntityExtractor:
    """Extracts entities from document chunks using LLM."""
    
    def __init__(self, llm_client=None, use_groq: bool = True):
        """
        Initialize entity extractor.
        
        Args:
            llm_client: Optional LLM client instance (GroqClient or GeminiClient)
            use_groq: Use Groq if True, Gemini if False (default: True)
        """
        if llm_client:
            self.llm = llm_client
            self.fallback_llm = None
        else:
            if use_groq:
                # Primary: llama-3.1-8b-instant (14.4K requests/day)
                self.llm = GroqClient(model="llama-3.1-8b-instant")
                # Fallback: llama-3.3-70b-versatile (1K requests/day)
                self.fallback_llm = GroqClient(model="llama-3.3-70b-versatile")
            else:
                self.llm = GeminiClient()
                self.fallback_llm = None
        
        self.extraction_stats = {
            'total_chunks': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'total_entities': 0,
            'fallback_used': 0
        }
    
    def extract_from_chunk(self, chunk: Dict) -> List[ExtractedEntity]:
        """
        Extract entities from a single chunk using 2-cycle retry strategy.
        
        Each cycle: primary model (@retry 3x) → fallback model (@retry 3x)
        If both fail in cycle 1, wait 60s and run cycle 2.
        If both fail in cycle 2, log error and return empty.
        
        Also rotates to next model if current returns 0 entities on
        substantial text (>200 chars), preventing silent extraction failures.
        """
        from extraction.llm_prompts import get_entity_extraction_prompt
        
        prompt = get_entity_extraction_prompt(
            text=chunk['text'],
            source=chunk['source'],
            section_id=chunk['section_id'],
            section_heading=chunk['section_heading']
        )
        
        is_substantial = len(chunk.get('text', '')) > 200
        chunk_id = chunk.get('chunk_id', 'unknown')
        
        for cycle in range(1, 3):  # Cycle 1 and Cycle 2
            # Build ordered list of models to try
            models_to_try = [('primary', self.llm)]
            if self.fallback_llm:
                models_to_try.append(('fallback', self.fallback_llm))
            
            for model_label, llm_client in models_to_try:
                try:
                    response_text = llm_client.generate(prompt, temperature=0.1)
                    entities = self._parse_extraction_response(response_text, chunk)
                    
                    # Rotate if 0 entities on substantial text
                    # (model likely failed to follow JSON instructions)
                    if len(entities) == 0 and is_substantial:
                        model_name = getattr(llm_client, 'model_name', 'unknown')
                        logger.warning(
                            f"[Cycle {cycle}] {model_label} ({model_name}) returned "
                            f"0 entities on substantial text, rotating..."
                        )
                        continue
                    
                    # Success
                    self.extraction_stats['successful_extractions'] += 1
                    self.extraction_stats['total_entities'] += len(entities)
                    if model_label == 'fallback':
                        self.extraction_stats['fallback_used'] += 1
                    return entities
                    
                except Exception as e:
                    model_name = getattr(llm_client, 'model_name', 'unknown')
                    error_str = str(e)[:150]
                    logger.warning(
                        f"[Cycle {cycle}] {model_label} ({model_name}) failed for "
                        f"chunk {chunk_id}: {error_str}"
                    )
                    continue
            
            # All models in this cycle failed
            if cycle == 1:
                logger.warning(
                    f"[Cycle 1] All models exhausted for chunk {chunk_id}. "
                    f"Waiting 60s before cycle 2..."
                )
                time.sleep(60)
            else:
                logger.error(
                    f"[Cycle 2] All models exhausted for chunk {chunk_id}. "
                    f"Giving up after 2 full cycles (up to 12 API attempts)."
                )
        
        self.extraction_stats['failed_extractions'] += 1
        return []
    
    def _repair_truncated_json(self, json_text: str) -> str:
        """Repair truncated JSON by closing unclosed strings, arrays, and objects."""
        if not json_text:
            return "{}"
        
        # Count open braces, brackets, and quotes
        open_braces = json_text.count('{') - json_text.count('}')
        open_brackets = json_text.count('[') - json_text.count(']')
        
        # Check for unterminated string (odd number of unescaped quotes in last line)
        last_colon = json_text.rfind(':')
        after_colon = json_text[last_colon:] if last_colon != -1 else json_text
        
        # If likely unterminated string, close it
        if '"' in after_colon and after_colon.count('"') % 2 == 1:
            json_text += '"'
        
        # Remove trailing comma if present
        json_text = json_text.rstrip().rstrip(',')
        
        # Close arrays and objects
        json_text += ']' * open_brackets
        json_text += '}' * open_braces
        
        return json_text
    
    def _parse_extraction_response(self, response_text: str, chunk: Dict) -> List[ExtractedEntity]:
        """Parse JSON response from LLM."""
        entities = []
        
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_text = response_text.strip()
            if '```json' in json_text:
                json_text = json_text.split('```json')[1].split('```')[0].strip()
            elif '```' in json_text:
                json_text = json_text.split('```')[1].split('```')[0].strip()
            
            # Try to parse as-is first
            try:
                data = json.loads(json_text)
            except json.JSONDecodeError as e:
                # Attempt to repair and retry
                logger.debug(f"JSON parse failed, attempting repair: {str(e)[:100]}")
                json_text = self._repair_truncated_json(json_text)
                data = json.loads(json_text)
            
            # Process requirements
            for req in data.get('requirements', []):
                entities.append(ExtractedEntity(
                    entity_type='Requirement',
                    text=req.get('text', ''),
                    source_chunk_id=chunk['chunk_id'],
                    source_doc=chunk['source'],
                    section_id=chunk['section_id'],
                    section_heading=chunk['section_heading'],
                    properties={
                        'category': req.get('category'),
                        'severity': req.get('severity'),
                        'scope': req.get('scope'),
                        'mandatory': req.get('mandatory', True)
                    }
                ))
            
            # Process definitions
            for defn in data.get('definitions', []):
                entities.append(ExtractedEntity(
                    entity_type='Definition',
                    text=defn.get('definition', ''),
                    source_chunk_id=chunk['chunk_id'],
                    source_doc=chunk['source'],
                    section_id=chunk['section_id'],
                    section_heading=chunk['section_heading'],
                    properties={
                        'term': defn.get('term'),
                        'domain': defn.get('domain')
                    }
                ))
            
            # Process exemptions
            for exempt in data.get('exemptions', []):
                entities.append(ExtractedEntity(
                    entity_type='Exemption',
                    text=exempt.get('text', ''),
                    source_chunk_id=chunk['chunk_id'],
                    source_doc=chunk['source'],
                    section_id=chunk['section_id'],
                    section_heading=chunk['section_heading'],
                    properties={
                        'condition': exempt.get('condition'),
                        'scope': exempt.get('scope')
                    }
                ))
            
            # Process processes
            for proc in data.get('processes', []):
                entities.append(ExtractedEntity(
                    entity_type='Process',
                    text=proc.get('description', ''),
                    source_chunk_id=chunk['chunk_id'],
                    source_doc=chunk['source'],
                    section_id=chunk['section_id'],
                    section_heading=chunk['section_heading'],
                    properties={
                        'name': proc.get('name'),
                        'process_type': proc.get('process_type')
                    }
                ))
            
            # Process cross-references
            for ref in data.get('cross_references', []):
                entities.append(ExtractedEntity(
                    entity_type='CrossReference',
                    text=ref.get('reference_text', ''),
                    source_chunk_id=chunk['chunk_id'],
                    source_doc=chunk['source'],
                    section_id=chunk['section_id'],
                    section_heading=chunk['section_heading'],
                    properties={
                        'target_doc': ref.get('target_doc'),
                        'target_section': ref.get('target_section'),
                        'reference_type': ref.get('reference_type')
                    }
                ))
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response text: {response_text[:500]}")
        except Exception as e:
            logger.warning(f"Failed to process extraction response: {e}")
        
        return entities
    
    def extract_from_chunks_file(self, chunks_file: Path, output_file: Path, 
                                  max_chunks: Optional[int] = None,
                                  resume: bool = True) -> List[ExtractedEntity]:
        """
        Extract entities from all chunks in a JSONL file.
        
        Args:
            chunks_file: Path to chunks JSONL file
            output_file: Path to output JSONL file for entities
            max_chunks: Maximum number of chunks to process (for testing)
            resume: Resume from last checkpoint if True
            
        Returns:
            List of all extracted entities
        """
        logger.info(f"Starting entity extraction from: {chunks_file}")
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing entities and track processed chunks
        all_entities = []
        processed_chunk_ids = set()
        
        if resume and output_file.exists():
            logger.info("Resume mode enabled - loading existing entities...")
            with jsonlines.open(output_file) as reader:
                for entity_dict in reader:
                    all_entities.append(ExtractedEntity(**entity_dict))
                    processed_chunk_ids.add(entity_dict['source_chunk_id'])
            logger.info(f"  Loaded {len(all_entities)} existing entities from {len(processed_chunk_ids)} chunks")
        
        # Load all chunks
        with jsonlines.open(chunks_file) as reader:
            chunks = list(reader)
        
        if max_chunks:
            chunks = chunks[:max_chunks]
            logger.info(f"Processing first {max_chunks} chunks (testing mode)")
        
        # Filter out already processed chunks
        remaining_chunks = [c for c in chunks if c['chunk_id'] not in processed_chunk_ids]
        
        if not remaining_chunks:
            logger.info("✓ All chunks already processed!")
            self.extraction_stats['total_chunks'] = len(chunks)
            self.extraction_stats['successful_extractions'] = len(processed_chunk_ids)
            return all_entities
        
        logger.info(f"Resuming from chunk {len(processed_chunk_ids)+1}/{len(chunks)}")
        logger.info(f"Remaining chunks to process: {len(remaining_chunks)}")
        
        self.extraction_stats['total_chunks'] = len(chunks)
        self.extraction_stats['successful_extractions'] = len(processed_chunk_ids)
        
        chunks_processed = len(processed_chunk_ids)
        
        # Process remaining chunks with progress bar
        for chunk in tqdm(remaining_chunks, desc="Extracting entities", initial=chunks_processed, total=len(chunks)):
            entities = self.extract_from_chunk(chunk)
            all_entities.extend(entities)
            chunks_processed += 1
            
            # Save incrementally (every 5 chunks for safety)
            if chunks_processed % 5 == 0:
                with jsonlines.open(output_file, mode='w') as writer:
                    for entity in all_entities:
                        writer.write(entity.to_dict())
            
            # Rate limiting - wait 3 seconds to stay under quota
            time.sleep(3)
        
        # Final save
        with jsonlines.open(output_file, mode='w') as writer:
            for entity in all_entities:
                writer.write(entity.to_dict())
        
        logger.info(f"✓ Extraction complete")
        logger.info(f"  Chunks processed: {chunks_processed}")
        logger.info(f"  Entities extracted: {len(all_entities)}")
        logger.info(f"  Success rate: {self.extraction_stats['successful_extractions']}/{self.extraction_stats['total_chunks']}")
        
        # Save statistics
        self._save_statistics(output_file.parent / 'extraction_statistics.json')
        
        return all_entities
    
    def _save_statistics(self, output_path: Path):
        """Save extraction statistics."""
        from collections import Counter
        
        # Load all entities to compute stats
        entities_file = output_path.parent / 'entities.jsonl'
        if entities_file.exists():
            with jsonlines.open(entities_file) as reader:
                entities = list(reader)
            
            entity_types = Counter(e['entity_type'] for e in entities)
            sources = Counter(e['source_doc'] for e in entities)
            
            self.extraction_stats.update({
                'entities_by_type': dict(entity_types),
                'entities_by_source': dict(sources),
                'timestamp': datetime.now().isoformat()
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.extraction_stats, f, indent=2)
        
        logger.info(f"✓ Statistics saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract entities from document chunks')
    parser.add_argument(
        '--input',
        type=str,
        default='data/processed_docs/chunks.jsonl',
        help='Input JSONL file with chunks'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed_docs/entities.jsonl',
        help='Output JSONL file for entities'
    )
    parser.add_argument(
        '--max-chunks',
        type=int,
        default=None,
        help='Maximum chunks to process (for testing)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        default=True,
        help='Resume from last checkpoint (default: True)'
    )
    parser.add_argument(
        '--no-resume',
        action='store_false',
        dest='resume',
        help='Start from beginning, ignore existing entities'
    )
    
    args = parser.parse_args()
    
    # Check for API key (Groq for highest rate limits)
    if not os.getenv("GROQ_API_KEY"):
        logger.error("GROQ_API_KEY not found in environment")
        logger.error("Please set GROQ_API_KEY in .env file")
        exit(1)
    
    # Use Groq with fallback strategy
    # Primary: llama-3.1-8b-instant (14.4K requests/day, 6K req/min)
    # Fallback: llama-3.3-70b-versatile (1K requests/day, 12K req/min)
    use_groq = True
    logger.info("✓ Using Groq API (llama-3.3-70b-versatile) - 1000 requests/day")
    logger.info("✓ Groq client initialized with model: llama-3.1-8b-instant")
    
    extractor = EntityExtractor(use_groq=use_groq)
    
    logger.info("Starting entity extraction pipeline...")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Resume mode: {'ON' if args.resume else 'OFF'}")
    
    entities = extractor.extract_from_chunks_file(
        Path(args.input),
        Path(args.output),
        max_chunks=args.max_chunks,
        resume=args.resume
    )
    
    print(f"\n✓ Extracted {len(entities)} entities")
    print(f"✓ Results saved to: {args.output}")

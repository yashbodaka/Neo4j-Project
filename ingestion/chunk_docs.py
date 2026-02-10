"""
Document Chunking and Embedding
Chunks parsed documents semantically and generates vector embeddings.
"""

import logging
import json
import jsonlines
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of document text with metadata."""
    
    chunk_id: str
    doc_id: str
    source: str
    section_id: str
    section_heading: str
    text: str
    char_count: int
    token_estimate: int
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class SemanticChunker:
    """Chunks documents while preserving semantic structure."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize semantic chunker.
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_document(self, document: Dict) -> List[DocumentChunk]:
        """
        Chunk a parsed document into smaller pieces.
        
        Args:
            document: Parsed document dict with sections
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        doc_id = document['metadata'].get('filename', 'unknown').replace('.', '_')
        source = self._extract_source(document['metadata'].get('filepath', ''))
        
        # Process each section
        for section in document['sections']:
            section_chunks = self._chunk_section(
                section=section,
                doc_id=doc_id,
                source=source
            )
            chunks.extend(section_chunks)
        
        logger.info(f"✓ Created {len(chunks)} chunks from {doc_id}")
        return chunks
    
    def _chunk_section(self, section: Dict, doc_id: str, source: str) -> List[DocumentChunk]:
        """Chunk a single section."""
        section_id = section.get('section_id', 'unknown')
        heading = section.get('heading', 'No Heading')
        content = section.get('content', '')
        
        if not content or len(content.strip()) < 100:
            # Section too small, return as single chunk
            return [self._create_chunk(
                doc_id=doc_id,
                source=source,
                section_id=section_id,
                heading=heading,
                text=content,
                chunk_index=0
            )]
        
        # Split content into sentences
        sentences = self._split_into_sentences(content)
        
        chunks = []
        current_chunk_text = []
        current_length = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # Check if adding this sentence exceeds chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk_text:
                # Save current chunk
                chunk_text = ' '.join(current_chunk_text)
                chunks.append(self._create_chunk(
                    doc_id=doc_id,
                    source=source,
                    section_id=section_id,
                    heading=heading,
                    text=chunk_text,
                    chunk_index=chunk_index
                ))
                
                # Start new chunk with overlap
                # Keep last few sentences for context
                overlap_text = ' '.join(current_chunk_text[-2:]) if len(current_chunk_text) > 2 else ''
                current_chunk_text = [overlap_text, sentence] if overlap_text else [sentence]
                current_length = len(overlap_text) + sentence_length
                chunk_index += 1
            else:
                current_chunk_text.append(sentence)
                current_length += sentence_length
        
        # Save final chunk
        if current_chunk_text:
            chunk_text = ' '.join(current_chunk_text)
            chunks.append(self._create_chunk(
                doc_id=doc_id,
                source=source,
                section_id=section_id,
                heading=heading,
                text=chunk_text,
                chunk_index=chunk_index
            ))
        
        return chunks
    
    def _create_chunk(self, doc_id: str, source: str, section_id: str, 
                      heading: str, text: str, chunk_index: int) -> DocumentChunk:
        """Create a DocumentChunk object."""
        chunk_id = f"{doc_id}_{section_id}_{chunk_index}"
        
        return DocumentChunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            source=source,
            section_id=section_id,
            section_heading=heading,
            text=text.strip(),
            char_count=len(text),
            token_estimate=len(text.split()) * 1.3,  # Rough token estimate
            metadata={
                'chunk_index': chunk_index,
                'created_at': datetime.now().isoformat()
            }
        )
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting (can be improved with nltk or spacy)
        import re
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter out very short fragments
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        return sentences
    
    def _extract_source(self, filepath: str) -> str:
        """Extract source name from filepath."""
        filepath_lower = filepath.lower()
        
        if 'mhra' in filepath_lower:
            return 'MHRA'
        elif 'eu_gmp' in filepath_lower or 'eudralex' in filepath_lower:
            return 'EU GMP'
        elif 'uk_gmp' in filepath_lower:
            return 'UK GMP'
        elif 'ich' in filepath_lower:
            return 'ICH'
        else:
            return 'Unknown'


class EmbeddingGenerator:
    """Generates vector embeddings for text chunks."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Name of sentence-transformers model to use
        """
        self.model_name = model_name
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Load the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("✓ Embedding model loaded")
            
        except ImportError:
            logger.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.model = None
    
    def generate_embeddings(self, chunks: List[DocumentChunk], batch_size: int = 32) -> List[DocumentChunk]:
        """
        Generate embeddings for all chunks.
        
        Args:
            chunks: List of DocumentChunk objects
            batch_size: Batch size for embedding generation
            
        Returns:
            List of chunks with embeddings added
        """
        if not self.model:
            logger.warning("Embedding model not available. Skipping embedding generation.")
            return chunks
        
        texts = [chunk.text for chunk in chunks]
        
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        
        # Generate embeddings in batches
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False)
            embeddings.extend(batch_embeddings.tolist())
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        logger.info("✓ Embeddings generated")
        return chunks


class ChunkingPipeline:
    """Complete pipeline for chunking and embedding documents."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, 
                 embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        Initialize chunking pipeline.
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            embedding_model: Sentence-transformers model name
        """
        self.chunker = SemanticChunker(chunk_size, chunk_overlap)
        self.embedder = EmbeddingGenerator(embedding_model)
    
    def process_documents(self, input_dir: Path, output_file: Path) -> List[DocumentChunk]:
        """
        Process all parsed documents and generate chunks with embeddings.
        
        Args:
            input_dir: Directory containing parsed document JSONs
            output_file: Output JSONL file path for chunks
            
        Returns:
            List of all chunks
        """
        input_dir = Path(input_dir)
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Find all parsed JSON files
        json_files = list(input_dir.rglob('*_parsed.json'))
        
        logger.info(f"Found {len(json_files)} parsed documents")
        
        all_chunks = []
        
        # Process each document
        for json_file in json_files:
            logger.info(f"Processing: {json_file.name}")
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    document = json.load(f)
                
                # Chunk document
                chunks = self.chunker.chunk_document(document)
                all_chunks.extend(chunks)
                
            except Exception as e:
                logger.error(f"Failed to process {json_file}: {e}")
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        
        # Generate embeddings
        all_chunks = self.embedder.generate_embeddings(all_chunks)
        
        # Save chunks to JSONL
        logger.info(f"Saving chunks to: {output_file}")
        with jsonlines.open(output_file, mode='w') as writer:
            for chunk in all_chunks:
                writer.write(chunk.to_dict())
        
        logger.info(f"✓ Saved {len(all_chunks)} chunks with embeddings")
        
        # Save summary statistics
        self._save_statistics(all_chunks, output_file.parent / 'chunk_statistics.json')
        
        return all_chunks
    
    def _save_statistics(self, chunks: List[DocumentChunk], output_path: Path):
        """Save chunking statistics."""
        from collections import Counter
        
        stats = {
            'total_chunks': len(chunks),
            'chunks_by_source': dict(Counter(chunk.source for chunk in chunks)),
            'avg_chunk_size': sum(chunk.char_count for chunk in chunks) / len(chunks) if chunks else 0,
            'avg_token_estimate': sum(chunk.token_estimate for chunk in chunks) / len(chunks) if chunks else 0,
            'embeddings_generated': sum(1 for chunk in chunks if chunk.embedding is not None),
            'generated_at': datetime.now().isoformat()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"✓ Statistics saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Chunk and embed documents')
    parser.add_argument(
        '--input',
        type=str,
        default='data/processed_docs',
        help='Input directory with parsed documents'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed_docs/chunks.jsonl',
        help='Output JSONL file for chunks'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1000,
        help='Target chunk size in characters'
    )
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=200,
        help='Overlap between chunks in characters'
    )
    parser.add_argument(
        '--embedding-model',
        type=str,
        default='all-MiniLM-L6-v2',
        help='Sentence-transformers model name'
    )
    
    args = parser.parse_args()
    
    pipeline = ChunkingPipeline(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model
    )
    
    logger.info("Starting chunking and embedding pipeline...")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Chunk size: {args.chunk_size}")
    logger.info(f"Chunk overlap: {args.chunk_overlap}")
    
    chunks = pipeline.process_documents(Path(args.input), Path(args.output))
    
    print(f"\n✓ Created and embedded {len(chunks)} chunks")
    print(f"✓ Results saved to: {args.output}")

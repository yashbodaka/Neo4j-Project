"""
Document Parser
Parses PDF and HTML regulatory documents, extracting text and structure.
"""

import logging
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from bs4 import BeautifulSoup
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentSection:
    """Represents a section within a document."""
    
    def __init__(self, section_id: str, heading: str, content: str, level: int = 1, page_num: Optional[int] = None):
        """
        Initialize document section.
        
        Args:
            section_id: Unique identifier for the section
            heading: Section heading/title
            content: Full text content
            level: Hierarchical level (1=top, 2=sub, etc.)
            page_num: Page number (for PDFs)
        """
        self.section_id = section_id
        self.heading = heading
        self.content = content
        self.level = level
        self.page_num = page_num
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'section_id': self.section_id,
            'heading': self.heading,
            'content': self.content,
            'level': self.level,
            'page_num': self.page_num
        }


class PDFParser:
    """Parser for PDF documents."""
    
    def __init__(self):
        """Initialize PDF parser."""
        # PDF parsing patterns for section identification
        self.section_patterns = [
            r'^(Chapter|CHAPTER)\s+(\d+[\.\d]*):?\s*(.+)$',
            r'^(Section|SECTION)\s+(\d+[\.\d]*):?\s*(.+)$',
            r'^(Article|ARTICLE)\s+(\d+[\.\d]*):?\s*(.+)$',
            r'^(\d+[\.\d]+)\s+(.+)$',  # Numbered sections like "1.1 Title"
            r'^([A-Z][A-Z\s]{3,}?)$',  # ALL CAPS headings
        ]
    
    def parse(self, filepath: Path) -> Dict:
        """
        Parse PDF document.
        
        Args:
            filepath: Path to PDF file
            
        Returns:
            Structured document dict with metadata and sections
        """
        try:
            # Try to import pypdf
            try:
                from pypdf import PdfReader
            except ImportError:
                logger.warning("pypdf not installed. Install with: pip install pypdf")
                return self._create_placeholder(filepath)
            
            reader = PdfReader(str(filepath))
            
            # Extract metadata
            metadata = {
                'filename': filepath.name,
                'filepath': str(filepath),
                'num_pages': len(reader.pages),
                'title': self._extract_title(reader),
                'parsed_at': datetime.now().isoformat()
            }
            
            # Extract text by page
            pages_text = []
            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text()
                    pages_text.append({
                        'page': page_num,
                        'text': text
                    })
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {e}")
            
            # Combine all text
            full_text = "\n\n".join([p['text'] for p in pages_text])
            
            # Extract sections
            sections = self._extract_sections(full_text, pages_text)
            
            document = {
                'metadata': metadata,
                'full_text': full_text,
                'sections': [s.to_dict() for s in sections]
            }
            
            logger.info(f"✓ Parsed PDF: {filepath.name} ({len(sections)} sections)")
            return document
            
        except Exception as e:
            logger.error(f"✗ Failed to parse PDF {filepath}: {e}")
            return self._create_placeholder(filepath)
    
    def _extract_title(self, reader) -> str:
        """Extract document title from PDF metadata or first page."""
        # Try metadata first
        if reader.metadata and reader.metadata.title:
            return reader.metadata.title
        
        # Try first page
        if len(reader.pages) > 0:
            first_page_text = reader.pages[0].extract_text()
            lines = [l.strip() for l in first_page_text.split('\n') if l.strip()]
            if lines:
                return lines[0][:100]  # First line, max 100 chars
        
        return "Unknown Title"
    
    def _extract_sections(self, full_text: str, pages_text: List[Dict]) -> List[DocumentSection]:
        """Extract structured sections from text."""
        sections = []
        lines = full_text.split('\n')
        
        current_section = None
        current_content = []
        section_counter = 0
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Check if line is a section heading
            is_heading, section_id, heading, level = self._identify_heading(line_stripped)
            
            if is_heading:
                # Save previous section
                if current_section:
                    current_section.content = '\n'.join(current_content)
                    sections.append(current_section)
                
                # Start new section
                section_counter += 1
                if not section_id:
                    section_id = f"section_{section_counter}"
                
                current_section = DocumentSection(
                    section_id=section_id,
                    heading=heading,
                    content="",
                    level=level
                )
                current_content = []
            else:
                # Add to current section content
                current_content.append(line_stripped)
        
        # Save last section
        if current_section:
            current_section.content = '\n'.join(current_content)
            sections.append(current_section)
        
        # If no sections found, create one default section
        if not sections:
            sections.append(DocumentSection(
                section_id="section_1",
                heading="Full Document",
                content=full_text,
                level=1
            ))
        
        return sections
    
    def _identify_heading(self, line: str) -> Tuple[bool, Optional[str], str, int]:
        """
        Identify if line is a heading and extract details.
        
        Returns:
            (is_heading, section_id, heading_text, level)
        """
        for pattern in self.section_patterns:
            match = re.match(pattern, line)
            if match:
                groups = match.groups()
                
                if len(groups) >= 3:  # Chapter/Section/Article patterns
                    prefix = groups[0]
                    number = groups[1]
                    title = groups[2]
                    section_id = f"{prefix.lower()}_{number}"
                    level = self._calculate_level(number)
                    return True, section_id, f"{prefix} {number}: {title}", level
                
                elif len(groups) == 2:  # Numbered sections
                    number = groups[0]
                    title = groups[1]
                    section_id = f"section_{number}"
                    level = self._calculate_level(number)
                    return True, section_id, f"{number} {title}", level
                
                elif len(groups) == 1:  # ALL CAPS headings
                    if len(line) < 100:  # Reasonable heading length
                        return True, None, line, 1
        
        return False, None, line, 1
    
    def _calculate_level(self, number: str) -> int:
        """Calculate hierarchical level from section number (e.g., '1.2.3' -> level 3)."""
        return len(number.split('.'))
    
    def _create_placeholder(self, filepath: Path) -> Dict:
        """Create placeholder document structure when parsing fails."""
        return {
            'metadata': {
                'filename': filepath.name,
                'filepath': str(filepath),
                'parsed_at': datetime.now().isoformat(),
                'error': 'Failed to parse document'
            },
            'full_text': '',
            'sections': []
        }


class HTMLParser:
    """Parser for HTML documents."""
    
    def __init__(self):
        """Initialize HTML parser."""
        self.heading_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
    
    def parse(self, filepath: Path) -> Dict:
        """
        Parse HTML document.
        
        Args:
            filepath: Path to HTML file
            
        Returns:
            Structured document dict with metadata and sections
        """
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer']):
                element.decompose()
            
            # Extract metadata
            title_tag = soup.find('title')
            h1_tag = soup.find('h1')
            
            metadata = {
                'filename': filepath.name,
                'filepath': str(filepath),
                'title': title_tag.text.strip() if title_tag else (h1_tag.text.strip() if h1_tag else 'Unknown'),
                'parsed_at': datetime.now().isoformat()
            }
            
            # Extract text content
            full_text = soup.get_text(separator='\n', strip=True)
            
            # Extract sections based on headings
            sections = self._extract_sections_from_html(soup)
            
            document = {
                'metadata': metadata,
                'full_text': full_text,
                'sections': [s.to_dict() for s in sections]
            }
            
            logger.info(f"✓ Parsed HTML: {filepath.name} ({len(sections)} sections)")
            return document
            
        except Exception as e:
            logger.error(f"✗ Failed to parse HTML {filepath}: {e}")
            return self._create_placeholder(filepath)
    
    def _extract_sections_from_html(self, soup: BeautifulSoup) -> List[DocumentSection]:
        """Extract sections from HTML based on heading structure."""
        sections = []
        section_counter = 0
        
        # Find all heading elements
        headings = soup.find_all(self.heading_tags)
        
        for i, heading in enumerate(headings):
            section_counter += 1
            level = int(heading.name[1])  # Extract number from h1, h2, etc.
            heading_text = heading.get_text(strip=True)
            
            # Get content between this heading and next
            content_parts = []
            current = heading.next_sibling
            
            next_heading = headings[i + 1] if i + 1 < len(headings) else None
            
            while current and current != next_heading:
                if hasattr(current, 'get_text'):
                    text = current.get_text(strip=True)
                    if text:
                        content_parts.append(text)
                current = current.next_sibling
            
            content = '\n'.join(content_parts)
            
            section = DocumentSection(
                section_id=f"section_{section_counter}",
                heading=heading_text,
                content=content,
                level=level
            )
            
            sections.append(section)
        
        # If no headings found, create one section with all content
        if not sections:
            full_text = soup.get_text(separator='\n', strip=True)
            sections.append(DocumentSection(
                section_id="section_1",
                heading="Full Document",
                content=full_text,
                level=1
            ))
        
        return sections
    
    def _create_placeholder(self, filepath: Path) -> Dict:
        """Create placeholder document structure when parsing fails."""
        return {
            'metadata': {
                'filename': filepath.name,
                'filepath': str(filepath),
                'parsed_at': datetime.now().isoformat(),
                'error': 'Failed to parse document'
            },
            'full_text': '',
            'sections': []
        }


class DocumentParser:
    """Main document parser that handles multiple formats."""
    
    def __init__(self):
        """Initialize document parser."""
        self.pdf_parser = PDFParser()
        self.html_parser = HTMLParser()
    
    def parse_document(self, filepath: Path) -> Dict:
        """
        Parse document based on file extension.
        
        Args:
            filepath: Path to document file
            
        Returns:
            Structured document dict
        """
        suffix = filepath.suffix.lower()
        
        if suffix == '.pdf':
            return self.pdf_parser.parse(filepath)
        elif suffix in ['.html', '.htm']:
            return self.html_parser.parse(filepath)
        else:
            logger.warning(f"Unsupported file format: {suffix}")
            return {
                'metadata': {'filename': filepath.name, 'error': f'Unsupported format: {suffix}'},
                'full_text': '',
                'sections': []
            }
    
    def parse_directory(self, input_dir: Path, output_dir: Path) -> List[Dict]:
        """
        Parse all documents in a directory.
        
        Args:
            input_dir: Directory containing raw documents
            output_dir: Directory to save parsed documents
            
        Returns:
            List of parsed document dictionaries
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        parsed_documents = []
        
        # Find all PDF and HTML files
        files = list(input_dir.rglob('*.pdf')) + list(input_dir.rglob('*.html')) + list(input_dir.rglob('*.htm'))
        
        logger.info(f"Found {len(files)} documents to parse")
        
        for filepath in files:
            logger.info(f"Parsing: {filepath.name}")
            
            parsed_doc = self.parse_document(filepath)
            parsed_documents.append(parsed_doc)
            
            # Save parsed document
            output_filename = filepath.stem + '_parsed.json'
            output_path = output_dir / filepath.parent.relative_to(input_dir) / output_filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(parsed_doc, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✓ Saved parsed document: {output_path}")
        
        return parsed_documents


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Parse regulatory documents')
    parser.add_argument(
        '--input',
        type=str,
        default='data/raw_docs',
        help='Input directory with raw documents'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed_docs',
        help='Output directory for parsed documents'
    )
    
    args = parser.parse_args()
    
    doc_parser = DocumentParser()
    
    logger.info("Starting document parsing...")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    
    parsed_docs = doc_parser.parse_directory(Path(args.input), Path(args.output))
    
    print(f"\n✓ Parsed {len(parsed_docs)} documents")
    print(f"✓ Results saved to: {args.output}")

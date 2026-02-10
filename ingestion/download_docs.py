"""
Document Downloader
Downloads regulatory documents from various sources (MHRA, EU GMP, UK GMP, ICH).
"""

import logging
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseDownloader:
    """Base class for document downloaders."""
    
    def __init__(self, output_dir: str = "data/raw_docs"):
        """
        Initialize downloader.
        
        Args:
            output_dir: Directory to save downloaded documents
        """
        self.output_dir = Path(output_dir)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def download_file(self, url: str, filename: str, max_retries: int = 3) -> Optional[Path]:
        """
        Download a file with retry logic.
        
        Args:
            url: URL to download from
            filename: Local filename to save as
            max_retries: Maximum number of retry attempts
            
        Returns:
            Path to downloaded file or None if failed
        """
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=30, stream=True)
                response.raise_for_status()
                
                filepath = self.output_dir / filename
                filepath.parent.mkdir(parents=True, exist_ok=True)
                
                total_size = int(response.headers.get('content-length', 0))
                
                with open(filepath, 'wb') as f:
                    if total_size > 0:
                        with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    pbar.update(len(chunk))
                    else:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                
                logger.info(f"✓ Downloaded: {filename}")
                return filepath
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {url}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"✗ Failed to download {url}")
                    return None
    
    def save_metadata(self, metadata: Dict, filename: str):
        """Save document metadata as JSON."""
        filepath = self.output_dir / f"{filename}.meta.json"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Saved metadata: {filename}.meta.json")


class MHRADownloader(BaseDownloader):
    """Downloader for MHRA guidance documents."""
    
    def __init__(self, output_dir: str = "data/raw_docs/mhra"):
        super().__init__(output_dir)
        self.base_urls = [
            "https://www.gov.uk/guidance/good-manufacturing-practice-and-good-distribution-practice",
            "https://www.gov.uk/guidance/patient-information-leaflet-pil"
        ]
    
    def download(self) -> List[Dict]:
        """
        Download MHRA documents.
        
        Returns:
            List of metadata dictionaries for downloaded documents
        """
        logger.info("Starting MHRA document download...")
        documents = []
        
        for url in self.base_urls:
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract page title
                title_elem = soup.find('h1', class_='gem-c-title__text')
                title = title_elem.text.strip() if title_elem else "MHRA Guidance"
                
                # Save HTML content
                filename = f"mhra_{urlparse(url).path.split('/')[-1]}.html"
                html_path = self.output_dir / filename
                html_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                # Extract and download linked PDFs
                pdf_links = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if href.endswith('.pdf'):
                        pdf_url = urljoin(url, href)
                        pdf_filename = f"mhra_{Path(urlparse(href).path).name}"
                        pdf_path = self.download_file(pdf_url, pdf_filename)
                        if pdf_path:
                            pdf_links.append(str(pdf_path))
                
                # Save metadata
                metadata = {
                    'source': 'MHRA',
                    'title': title,
                    'url': url,
                    'type': 'guidance',
                    'date_accessed': datetime.now().isoformat(),
                    'html_file': str(html_path),
                    'pdf_files': pdf_links
                }
                
                self.save_metadata(metadata, filename.replace('.html', ''))
                documents.append(metadata)
                
                logger.info(f"✓ Downloaded MHRA guidance: {title}")
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"✗ Failed to download MHRA page {url}: {e}")
        
        return documents


class EUGMPDownloader(BaseDownloader):
    """Downloader for EU GMP guidelines."""
    
    def __init__(self, output_dir: str = "data/raw_docs/eu_gmp"):
        super().__init__(output_dir)
        self.base_url = "https://health.ec.europa.eu/medicinal-products/eudralex/eudralex-volume-4_en"
    
    def download(self) -> List[Dict]:
        """
        Download EU GMP documents.
        
        Returns:
            List of metadata dictionaries for downloaded documents
        """
        logger.info("Starting EU GMP document download...")
        documents = []
        
        try:
            response = self.session.get(self.base_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Save main page
            filename = "eu_gmp_main.html"
            html_path = self.output_dir / filename
            html_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            # Extract PDF links for GMP Annexes
            pdf_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if 'annex' in href.lower() and href.endswith('.pdf'):
                    pdf_url = urljoin(self.base_url, href)
                    pdf_filename = f"eu_gmp_{Path(urlparse(href).path).name}"
                    pdf_path = self.download_file(pdf_url, pdf_filename)
                    if pdf_path:
                        pdf_links.append(str(pdf_path))
            
            # Save metadata
            metadata = {
                'source': 'EU GMP',
                'title': 'EudraLex Volume 4 - GMP Guidelines',
                'url': self.base_url,
                'type': 'regulation',
                'date_accessed': datetime.now().isoformat(),
                'html_file': str(html_path),
                'pdf_files': pdf_links
            }
            
            self.save_metadata(metadata, filename.replace('.html', ''))
            documents.append(metadata)
            
            logger.info(f"✓ Downloaded EU GMP guidelines ({len(pdf_links)} PDFs)")
            
        except Exception as e:
            logger.error(f"✗ Failed to download EU GMP: {e}")
        
        return documents


class UKGMPDownloader(BaseDownloader):
    """Downloader for UK GMP guidelines."""
    
    def __init__(self, output_dir: str = "data/raw_docs/uk_gmp"):
        super().__init__(output_dir)
        self.base_url = "https://www.gov.uk/guidance/good-manufacturing-practice-and-good-distribution-practice"
    
    def download(self) -> List[Dict]:
        """
        Download UK GMP documents.
        
        Returns:
            List of metadata dictionaries for downloaded documents
        """
        logger.info("Starting UK GMP document download...")
        documents = []
        
        try:
            response = self.session.get(self.base_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_elem = soup.find('h1', class_='gem-c-title__text')
            title = title_elem.text.strip() if title_elem else "UK GMP Guidelines"
            
            # Save HTML content
            filename = "uk_gmp_main.html"
            html_path = self.output_dir / filename
            html_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            # Extract PDF links
            pdf_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.endswith('.pdf'):
                    pdf_url = urljoin(self.base_url, href)
                    pdf_filename = f"uk_gmp_{Path(urlparse(href).path).name}"
                    pdf_path = self.download_file(pdf_url, pdf_filename)
                    if pdf_path:
                        pdf_links.append(str(pdf_path))
            
            # Save metadata
            metadata = {
                'source': 'UK GMP',
                'title': title,
                'url': self.base_url,
                'type': 'guidance',
                'date_accessed': datetime.now().isoformat(),
                'html_file': str(html_path),
                'pdf_files': pdf_links
            }
            
            self.save_metadata(metadata, filename.replace('.html', ''))
            documents.append(metadata)
            
            logger.info(f"✓ Downloaded UK GMP guidelines ({len(pdf_links)} PDFs)")
            
        except Exception as e:
            logger.error(f"✗ Failed to download UK GMP: {e}")
        
        return documents


class ICHDownloader(BaseDownloader):
    """Downloader for ICH quality guidelines."""
    
    def __init__(self, output_dir: str = "data/raw_docs/ich"):
        super().__init__(output_dir)
        self.base_url = "https://www.ich.org/page/quality-guidelines"
    
    def download(self) -> List[Dict]:
        """
        Download ICH quality guidelines.
        
        Returns:
            List of metadata dictionaries for downloaded documents
        """
        logger.info("Starting ICH guidelines download...")
        documents = []
        
        try:
            response = self.session.get(self.base_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Save main page
            filename = "ich_quality_main.html"
            html_path = self.output_dir / filename
            html_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            # Extract PDF links
            pdf_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.endswith('.pdf') and 'quality' in href.lower():
                    pdf_url = urljoin(self.base_url, href)
                    pdf_filename = f"ich_{Path(urlparse(href).path).name}"
                    pdf_path = self.download_file(pdf_url, pdf_filename)
                    if pdf_path:
                        pdf_links.append(str(pdf_path))
            
            # Save metadata
            metadata = {
                'source': 'ICH',
                'title': 'ICH Quality Guidelines',
                'url': self.base_url,
                'type': 'guideline',
                'date_accessed': datetime.now().isoformat(),
                'html_file': str(html_path),
                'pdf_files': pdf_links
            }
            
            self.save_metadata(metadata, filename.replace('.html', ''))
            documents.append(metadata)
            
            logger.info(f"✓ Downloaded ICH guidelines ({len(pdf_links)} PDFs)")
            
        except Exception as e:
            logger.error(f"✗ Failed to download ICH guidelines: {e}")
        
        return documents


def download_all_sources(sources: Optional[List[str]] = None) -> Dict[str, List[Dict]]:
    """
    Download documents from all specified sources.
    
    Args:
        sources: List of source names to download. If None, downloads all.
                 Valid values: 'mhra', 'eu_gmp', 'uk_gmp', 'ich'
    
    Returns:
        Dictionary mapping source names to lists of document metadata
    """
    if sources is None:
        sources = ['mhra', 'eu_gmp', 'uk_gmp', 'ich']
    
    downloaders = {
        'mhra': MHRADownloader(),
        'eu_gmp': EUGMPDownloader(),
        'uk_gmp': UKGMPDownloader(),
        'ich': ICHDownloader()
    }
    
    results = {}
    
    for source in sources:
        if source.lower() in downloaders:
            logger.info(f"\n{'='*60}")
            logger.info(f"Downloading from: {source.upper()}")
            logger.info(f"{'='*60}")
            
            downloader = downloaders[source.lower()]
            documents = downloader.download()
            results[source] = documents
            
            logger.info(f"✓ Completed {source.upper()}: {len(documents)} documents")
        else:
            logger.warning(f"Unknown source: {source}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download regulatory documents')
    parser.add_argument(
        '--sources',
        nargs='+',
        choices=['mhra', 'eu_gmp', 'uk_gmp', 'ich', 'all'],
        default=['all'],
        help='Sources to download from'
    )
    
    args = parser.parse_args()
    
    sources = None if 'all' in args.sources else args.sources
    
    logger.info("Starting document download pipeline...")
    logger.info(f"Sources: {sources if sources else 'ALL'}")
    
    results = download_all_sources(sources)
    
    # Print summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    
    total_docs = 0
    for source, docs in results.items():
        print(f"{source.upper()}: {len(docs)} documents")
        total_docs += len(docs)
    
    print(f"\nTotal documents downloaded: {total_docs}")
    print("="*60)

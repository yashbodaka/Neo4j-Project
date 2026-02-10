"""
Graph Builder
Populates Neo4j knowledge graph from extracted entities.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import json
import jsonlines
import re
from typing import List, Dict, Optional, Set
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
import hashlib

from graph.neo4j_client import Neo4jClient
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NodeIDGenerator:
    """Generates unique, consistent node IDs."""
    
    @staticmethod
    def generate_id(entity_type: str, source_doc: str, text: str) -> str:
        """
        Generate unique node ID from entity content.
        
        Args:
            entity_type: Type of entity
            source_doc: Source document
            text: Entity text
            
        Returns:
            Unique ID string
        """
        # Handle None values
        source_doc = source_doc or 'unknown'
        text = text or 'unknown'
        # Create hash from text to ensure consistency
        text_hash = hashlib.md5(text.lower().strip().encode()).hexdigest()[:8]
        return f"{entity_type.lower()}_{source_doc.lower().replace(' ', '_')}_{text_hash}"


class GraphBuilder:
    """Builds Neo4j knowledge graph from extracted entities."""
    
    def __init__(self, neo4j_client: Optional[Neo4jClient] = None):
        """
        Initialize graph builder.
        
        Args:
            neo4j_client: Neo4jClient instance
        """
        self.client = neo4j_client or Neo4jClient()
        self.node_id_cache: Dict[str, str] = {}
        self.entity_map: Dict[str, Dict] = {}
        self.build_stats = {
            'total_entities': 0,
            'nodes_created': 0,
            'relationships_created': 0,
            'errors': 0
        }
        # Initialize embedding model
        logger.info("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✓ Embedding model loaded")
    
    def _extract_unique_source_docs(self, entities: List[Dict]) -> Set[str]:
        """Extract unique source document names from entities."""
        source_docs = set()
        for entity in entities:
            source_doc = entity.get('source_doc', '').strip()
            if source_doc:
                source_docs.add(source_doc)
        logger.info(f"Found {len(source_docs)} unique source documents: {sorted(source_docs)}")
        return source_docs
    
    def _auto_create_regulation_nodes(self, source_docs: Set[str]):
        """
        Automatically create Regulation nodes for each source document.
        This ensures every regulation has a proper node in the graph.
        """
        # Regulation metadata mapping
        regulation_metadata = {
            'EU GMP': {
                'name': 'EU GMP Volume 4 - Good Manufacturing Practice',
                'issued_by': 'European Commission',
                'effective_date': '2015-03-01',
                'version': 'Current Edition',
                'description': 'Guidelines on Good Manufacturing Practice for medicinal products for human and veterinary use'
            },
            'MHRA': {
                'name': 'MHRA Orange Guide - Rules and Guidance for Pharmaceutical Manufacturers',
                'issued_by': 'Medicines and Healthcare products Regulatory Agency (UK)',
                'effective_date': '',
                'version': 'Latest Edition',
                'description': 'UK guidance for pharmaceutical manufacturing quality'
            },
            'ICH': {
                'name': 'ICH Quality Guidelines',
                'issued_by': 'International Council for Harmonisation of Technical Requirements for Pharmaceuticals for Human Use',
                'effective_date': '',
                'version': 'Q7-Q14',
                'description': 'International harmonised technical requirements for pharmaceutical quality'
            },
            'UK GMP': {
                'name': 'UK Good Manufacturing Practice',
                'issued_by': 'Medicines and Healthcare products Regulatory Agency (UK)',
                'effective_date': '',
                'version': 'Current Edition',
                'description': 'UK-specific GMP requirements post-Brexit'
            }
        }
        
        logger.info(f"Creating Regulation nodes for {len(source_docs)} source documents...")
        
        for source_doc in source_docs:
            # Get metadata or use defaults
            metadata = regulation_metadata.get(source_doc, {
                'name': source_doc,
                'issued_by': 'Unknown Regulatory Authority',
                'effective_date': '',
                'version': 'Unknown',
                'description': f'Regulatory document: {source_doc}'
            })
            
            # Generate embedding for regulation
            embedding_text = f"{metadata['name']} {metadata['description']}"
            embedding = self._generate_embedding(embedding_text)
            
            cypher = """
            MERGE (reg:Regulation {id: $id})
            SET reg.name = $name,
                reg.issued_by = $issued_by,
                reg.effective_date = $effective_date,
                reg.version = $version,
                reg.source_doc = $source_doc,
                reg.description = $description,
                reg.embedding = $embedding,
                reg.created_at = datetime(),
                reg.updated_at = datetime()
            RETURN reg
            """
            
            try:
                self.client.execute_write_query(cypher, {
                    'id': f"regulation_{source_doc.lower().replace(' ', '_')}",
                    'name': metadata['name'],
                    'issued_by': metadata['issued_by'],
                    'effective_date': metadata['effective_date'],
                    'version': metadata['version'],
                    'source_doc': source_doc,
                    'description': metadata['description'],
                    'embedding': embedding
                })
                self.build_stats['nodes_created'] += 1
                logger.info(f"  ✓ Created: {metadata['name']}")
            except Exception as e:
                logger.error(f"  ✗ Failed to create Regulation for {source_doc}: {e}")
                self.build_stats['errors'] += 1
    
    def build_from_entities_file(self, entities_file: Path) -> Dict:
        """
        Build graph from entities JSONL file.
        
        Args:
            entities_file: Path to entities.jsonl
            
        Returns:
            Build statistics
        """
        logger.info(f"Starting graph build from: {entities_file}")
        
        # Load all entities
        entities = []
        with jsonlines.open(entities_file) as reader:
            entities = list(reader)
        
        self.build_stats['total_entities'] = len(entities)
        logger.info(f"Loaded {len(entities)} entities")
        
        # ✅ PERMANENT FIX: Auto-create Regulation nodes from source documents
        logger.info("Auto-creating Regulation nodes from source documents...")
        source_docs = self._extract_unique_source_docs(entities)
        self._auto_create_regulation_nodes(source_docs)
        
        # Index entities by type
        entities_by_type = defaultdict(list)
        for entity in entities:
            entities_by_type[entity['entity_type']].append(entity)
        
        # Create all nodes
        logger.info("Creating nodes...")
        for entity_type, entity_list in entities_by_type.items():
            self._create_nodes_by_type(entity_type, entity_list)
        
        logger.info(f" Nodes created: {self.build_stats['nodes_created']}")
        
        # Create relationships
        logger.info("Creating relationships...")
        self._create_relationships(entities)
        
        logger.info(f" Relationships created: {self.build_stats['relationships_created']}")
        
        # Save statistics
        self._save_build_stats(entities_file.parent / 'graph_build_statistics.json')
        
        return self.build_stats
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        try:
            embedding = self.embedding_model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return []
    
    def _create_nodes_by_type(self, entity_type: str, entities: List[Dict]):
        """Create nodes for a specific entity type."""
        
        if entity_type == 'Requirement':
            self._create_requirement_nodes(entities)
        elif entity_type == 'Definition':
            self._create_definition_nodes(entities)
        elif entity_type == 'Exemption':
            self._create_exemption_nodes(entities)
        elif entity_type == 'Process':
            self._create_process_nodes(entities)
        elif entity_type == 'Regulation':
            self._create_regulation_nodes(entities)
        elif entity_type == 'CrossReference':
            self._create_crossreference_nodes(entities)
    
    def _create_requirement_nodes(self, entities: List[Dict]):
        """Create Requirement nodes."""
        logger.info(f"Creating {len(entities)} Requirement nodes...")
        
        # Deduplication
        seen = set()
        unique_entities = []
        
        for entity in entities:
            text_key = entity['text'].lower().strip()
            if text_key not in seen:
                seen.add(text_key)
                unique_entities.append(entity)
        
        for entity in tqdm(unique_entities, desc="Requirement Nodes"):
            node_id = NodeIDGenerator.generate_id('Requirement', entity['source_doc'], entity['text'])
            
            props = entity.get('properties', {})
            
            # Generate embedding
            embedding = self._generate_embedding(entity['text'])
            
            cypher = """
            MERGE (req:Requirement {id: $id})
            SET req.name = $name,
                req.text = $text,
                req.source_doc = $source_doc,
                req.source_url = COALESCE(req.source_url, ''),
                req.section_id = $section_id,
                req.citation_text = $citation_text,
                req.category = $category,
                req.severity = $severity,
                req.scope = $scope,
                req.mandatory = $mandatory,
                req.embedding = $embedding,
                req.created_at = datetime()
            RETURN req
            """
            
            try:
                self.client.execute_write_query(cypher, {
                    'id': node_id,
                    'name': entity['text'][:200],
                    'text': entity['text'],
                    'source_doc': entity['source_doc'],
                    'section_id': entity['section_id'],
                    'citation_text': f"{entity['source_doc']} {entity['section_id']}",
                    'category': props.get('category', ''),
                    'severity': props.get('severity', ''),
                    'scope': props.get('scope', ''),
                    'mandatory': props.get('mandatory', True),
                    'embedding': embedding
                })
                
                self.entity_map[node_id] = entity
                self.build_stats['nodes_created'] += 1
                
            except Exception as e:
                logger.error(f"Failed to create Requirement node: {e}")
                self.build_stats['errors'] += 1
    
    def _create_definition_nodes(self, entities: List[Dict]):
        """Create Definition nodes."""
        logger.info(f"Creating {len(entities)} Definition nodes...")
        
        seen = set()
        unique_entities = []
        for entity in entities:
            term = entity.get('properties', {}).get('term', '')
            if term and term not in seen:
                seen.add(term)
                unique_entities.append(entity)
        
        for entity in tqdm(unique_entities, desc="Definition Nodes"):
            props = entity.get('properties', {})
            term = props.get('term', 'Unknown')
            node_id = NodeIDGenerator.generate_id('Definition', entity['source_doc'], term)
            
            # Generate embedding
            embedding = self._generate_embedding(f"{term}: {entity['text']}")
            
            cypher = """
            MERGE (defn:Definition {id: $id})
            SET defn.term = $term,
                defn.text = $text,
                defn.source_doc = $source_doc,
                defn.section_id = $section_id,
                defn.citation_text = $citation_text,
                defn.domain = $domain,
                defn.embedding = $embedding,
                defn.created_at = datetime()
            RETURN defn
            """
            
            try:
                self.client.execute_write_query(cypher, {
                    'id': node_id,
                    'term': term,
                    'text': entity['text'],
                    'source_doc': entity['source_doc'],
                    'section_id': entity['section_id'],
                    'citation_text': f"{entity['source_doc']} {entity['section_id']}",
                    'domain': props.get('domain', ''),
                    'embedding': embedding
                })
                
                self.entity_map[node_id] = entity
                self.build_stats['nodes_created'] += 1
                
            except Exception as e:
                logger.error(f"Failed to create Definition node: {e}")
                self.build_stats['errors'] += 1
    
    def _create_exemption_nodes(self, entities: List[Dict]):
        """Create Exemption nodes."""
        logger.info(f"Creating {len(entities)} Exemption nodes...")
        
        for entity in tqdm(entities, desc="Exemption Nodes"):
            node_id = NodeIDGenerator.generate_id('Exemption', entity['source_doc'], entity['text'])
            props = entity.get('properties', {})
            
            # Generate embedding
            embedding = self._generate_embedding(entity['text'])
            
            cypher = """
            MERGE (exempt:Exemption {id: $id})
            SET exempt.text = $text,
                exempt.source_doc = $source_doc,
                exempt.section_id = $section_id,
                exempt.citation_text = $citation_text,
                exempt.condition = $condition,
                exempt.scope = $scope,
                exempt.embedding = $embedding,
                exempt.created_at = datetime()
            RETURN exempt
            """
            
            try:
                self.client.execute_write_query(cypher, {
                    'id': node_id,
                    'text': entity['text'],
                    'source_doc': entity['source_doc'],
                    'section_id': entity['section_id'],
                    'citation_text': f"{entity['source_doc']} {entity['section_id']}",
                    'condition': props.get('condition', ''),
                    'scope': props.get('scope', ''),
                    'embedding': embedding
                })
                
                self.entity_map[node_id] = entity
                self.build_stats['nodes_created'] += 1
                
            except Exception as e:
                logger.error(f"Failed to create Exemption node: {e}")
                self.build_stats['errors'] += 1
    
    def _create_process_nodes(self, entities: List[Dict]):
        """Create Process nodes."""
        logger.info(f"Creating {len(entities)} Process nodes...")
        
        seen = set()
        unique_entities = []
        for entity in entities:
            proc_name = entity.get('properties', {}).get('name', '')
            if proc_name and proc_name not in seen:
                seen.add(proc_name)
                unique_entities.append(entity)
        
        for entity in tqdm(unique_entities, desc="Process Nodes"):
            props = entity.get('properties', {})
            proc_name = props.get('name', 'Unknown')
            node_id = NodeIDGenerator.generate_id('Process', entity['source_doc'], proc_name)
            
            # Generate embedding
            embedding = self._generate_embedding(f"{proc_name}: {entity['text']}")
            
            cypher = """
            MERGE (proc:Process {id: $id})
            SET proc.name = $name,
                proc.text = $text,
                proc.source_doc = $source_doc,
                proc.section_id = $section_id,
                proc.citation_text = $citation_text,
                proc.process_type = $process_type,
                proc.industry = 'Pharmaceutical',
                proc.embedding = $embedding,
                proc.created_at = datetime()
            RETURN proc
            """
            
            try:
                self.client.execute_write_query(cypher, {
                    'id': node_id,
                    'name': proc_name,
                    'text': entity['text'],
                    'source_doc': entity['source_doc'],
                    'section_id': entity['section_id'],
                    'citation_text': f"{entity['source_doc']} {entity['section_id']}",
                    'process_type': props.get('process_type', ''),
                    'embedding': embedding
                })
                
                self.entity_map[node_id] = entity
                self.build_stats['nodes_created'] += 1
                
            except Exception as e:
                logger.error(f"Failed to create Process node: {e}")
                self.build_stats['errors'] += 1
    
    def _create_crossreference_nodes(self, entities: List[Dict]):
        """Create CrossReference nodes."""
        logger.info(f"Creating {len(entities)} CrossReference nodes...")
        
        for entity in tqdm(entities, desc="CrossReference Nodes"):
            props = entity.get('properties', {})
            target_doc = props.get('target_doc', 'Unknown') or 'Unknown'
            source_doc = entity.get('source_doc', 'Unknown') or 'Unknown'
            
            node_id = NodeIDGenerator.generate_id('CrossReference', source_doc, target_doc)
            
            # Generate embedding
            embedding = self._generate_embedding(entity.get('text', ''))
            
            cypher = """
            MERGE (cref:CrossReference {id: $id})
            SET cref.text = $text,
                cref.source_doc = $source_doc,
                cref.target_doc = $target_doc,
                cref.target_section = $target_section,
                cref.reference_type = $reference_type,
                cref.embedding = $embedding,
                cref.created_at = datetime()
            RETURN cref
            """
            
            try:
                self.client.execute_write_query(cypher, {
                    'id': node_id,
                    'text': entity.get('text', ''),
                    'source_doc': source_doc,
                    'target_doc': target_doc,
                    'target_section': props.get('target_section', ''),
                    'reference_type': props.get('reference_type', 'references'),
                    'embedding': embedding
                })
                
                self.entity_map[node_id] = entity
                self.build_stats['nodes_created'] += 1
                
            except Exception as e:
                logger.error(f"Failed to create CrossReference node: {e}")
                self.build_stats['errors'] += 1
    
    def _create_regulation_nodes(self, entities: List[Dict]):
        """Create Regulation nodes."""
        logger.info(f"Creating {len(entities)} Regulation nodes...")
        
        seen = set()
        unique_entities = []
        for entity in entities:
            reg_name = entity.get('properties', {}).get('name', '')
            if reg_name and reg_name not in seen:
                seen.add(reg_name)
                unique_entities.append(entity)
        
        for entity in tqdm(unique_entities, desc="Regulation Nodes"):
            props = entity.get('properties', {})
            reg_name = props.get('name', 'Unknown')
            node_id = NodeIDGenerator.generate_id('Regulation', entity['source_doc'], reg_name)
            
            # Generate embedding
            embedding = self._generate_embedding(f"{reg_name}: {entity.get('text', '')}")
            
            cypher = """
            MERGE (reg:Regulation {id: $id})
            SET reg.name = $name,
                reg.issued_by = $issued_by,
                reg.effective_date = $effective_date,
                reg.version = $version,
                reg.source_doc = $source_doc,
                reg.embedding = $embedding,
                reg.created_at = datetime()
            RETURN reg
            """
            
            try:
                self.client.execute_write_query(cypher, {
                    'id': node_id,
                    'name': reg_name,
                    'issued_by': props.get('issued_by', 'Unknown'),
                    'effective_date': props.get('effective_date', ''),
                    'version': props.get('version', ''),
                    'source_doc': entity['source_doc'],
                    'embedding': embedding
                })
                
                self.entity_map[node_id] = entity
                self.build_stats['nodes_created'] += 1
                
            except Exception as e:
                logger.error(f"Failed to create Regulation node: {e}")
                self.build_stats['errors'] += 1
    
    def _create_relationships(self, entities: List[Dict]):
        """Create relationships between entities."""
        logger.info("Inferring and creating relationships...")
        
        # Build indexes
        requirements = [e for e in entities if e['entity_type'] == 'Requirement']
        processes = [e for e in entities if e['entity_type'] == 'Process']
        definitions = [e for e in entities if e['entity_type'] == 'Definition']
        exemptions = [e for e in entities if e['entity_type'] == 'Exemption']
        
        # ✅ PERMANENT FIX: CONTAINS - Link Regulations to their Requirements
        logger.info("Creating CONTAINS relationships (Regulation → Requirements)...")
        contains_count = self._create_contains_relationships()
        logger.info(f"  ✓ Created {contains_count} CONTAINS relationships")
        
        # ✅ PERMANENT FIX: SUPERSEDES - Detect supersession from requirement text
        logger.info("Creating SUPERSEDES relationships...")
        supersedes_count = self._create_supersedes_relationships(requirements)
        logger.info(f"  ✓ Created {supersedes_count} SUPERSEDES relationships")
        
        # ✅ NEW: CONFLICTS_WITH - Detect conflicts between requirements
        logger.info("Creating CONFLICTS_WITH relationships...")
        conflicts_count = self._create_conflicts_relationships(requirements)
        logger.info(f"  ✓ Created {conflicts_count} CONFLICTS_WITH relationships")
        
        # 1. APPLIES_TO: Requirements to Processes
        logger.info("Creating APPLIES_TO relationships...")
        for req in tqdm(requirements, desc="APPLIES_TO"):
            req_text = req['text'].lower()
            req_scope = req.get('properties', {}).get('scope', '').lower()
            
            for proc in processes:
                proc_name = proc.get('properties', {}).get('name', '').lower()
                
                if proc_name and (proc_name in req_text or proc_name in req_scope):
                    req_id = NodeIDGenerator.generate_id('Requirement', req['source_doc'], req['text'])
                    proc_id = NodeIDGenerator.generate_id('Process', 'general', proc['properties'].get('name', ''))
                    
                    self.client.execute_write_query("""
                        MATCH (req:Requirement {id: $req_id})
                        MATCH (proc:Process {id: $proc_id})
                        MERGE (req)-[:APPLIES_TO]->(proc)
                    """, {'req_id': req_id, 'proc_id': proc_id})
                    self.build_stats['relationships_created'] += 1

        # 2. DEFINES: Definitions to Requirements
        logger.info("Creating DEFINES relationships...")
        for defn in tqdm(definitions, desc="DEFINES"):
            term = defn.get('properties', {}).get('term', '').lower()
            if not term: continue
            
            for req in requirements:
                if term in req['text'].lower():
                    defn_id = NodeIDGenerator.generate_id('Definition', defn['source_doc'], term)
                    req_id = NodeIDGenerator.generate_id('Requirement', req['source_doc'], req['text'])
                    
                    self.client.execute_write_query("""
                        MATCH (defn:Definition {id: $defn_id})
                        MATCH (req:Requirement {id: $req_id})
                        MERGE (defn)-[:DEFINES]->(req)
                    """, {'defn_id': defn_id, 'req_id': req_id})
                    self.build_stats['relationships_created'] += 1

        # 3. REQUIRES: Requirement Dependencies (Inferred)
        logger.info("Creating REQUIRES relationships...")
        for req1 in tqdm(requirements, desc="REQUIRES"):
            req1_text = req1['text'].lower()
            for req2 in requirements:
                if req1 == req2: continue
                # Very basic inference: if req1 mentions text from req2 (long enough)
                req2_snippet = req2['text'][:100].lower()
                if len(req2_snippet) > 30 and req2_snippet in req1_text:
                    id1 = NodeIDGenerator.generate_id('Requirement', req1['source_doc'], req1['text'])
                    id2 = NodeIDGenerator.generate_id('Requirement', req2['source_doc'], req2['text'])
                    self.client.execute_write_query("""
                        MATCH (r1:Requirement {id: $id1})
                        MATCH (r2:Requirement {id: $id2})
                        MERGE (r1)-[:REQUIRES]->(r2)
                    """, {'id1': id1, 'id2': id2})
                    self.build_stats['relationships_created'] += 1

        # 4. EXEMPT_FROM: Exemptions to Requirements
        logger.info("Creating EXEMPT_FROM relationships...")
        for ex in tqdm(exemptions, desc="EXEMPT_FROM"):
            ex_text = ex['text'].lower()
            for req in requirements:
                req_text = req['text'].lower()
                # If exemption text mentions requirement context or vice versa
                if any(word in ex_text and word in req_text for word in ['aseptic', 'filling', 'sterile', 'monitoring']):
                    ex_id = NodeIDGenerator.generate_id('Exemption', ex['source_doc'], ex['text'])
                    req_id = NodeIDGenerator.generate_id('Requirement', req['source_doc'], req['text'])
                    self.client.execute_write_query("""
                        MATCH (ex:Exemption {id: $ex_id})
                        MATCH (req:Requirement {id: $req_id})
                        MERGE (ex)-[:EXEMPT_FROM]->(req)
                    """, {'ex_id': ex_id, 'req_id': req_id})
                    self.build_stats['relationships_created'] += 1

        logger.info(f" Relationships created: {self.build_stats['relationships_created']}")        
        # 5. CONTAINS: Regulations to Requirements
        logger.info("Creating CONTAINS relationships...")
        regulations = [e for e in entities if e['entity_type'] == 'Regulation']
        for reg in tqdm(regulations, desc="CONTAINS"):
            reg_name = reg.get('properties', {}).get('name', '').lower()
            reg_source = reg['source_doc'].lower()
            
            for req in requirements:
                req_source = req['source_doc'].lower()
                # Match if requirement comes from same source document
                if reg_source in req_source or req_source in reg_name:
                    reg_id = NodeIDGenerator.generate_id('Regulation', reg['source_doc'], reg['properties']['name'])
                    req_id = NodeIDGenerator.generate_id('Requirement', req['source_doc'], req['text'])
                    
                    try:
                        self.client.execute_write_query("""
                            MATCH (reg:Regulation {id: $reg_id})
                            MATCH (req:Requirement {id: $req_id})
                            MERGE (reg)-[:CONTAINS]->(req)
                        """, {'reg_id': reg_id, 'req_id': req_id})
                        self.build_stats['relationships_created'] += 1
                    except Exception as e:
                        logger.debug(f"Could not create CONTAINS relationship: {e}")

        # 6. SUPERSEDES: Regulation version control
        logger.info("Creating SUPERSEDES relationships...")
        cross_refs = [e for e in entities if e['entity_type'] == 'CrossReference']
        for cref in tqdm(cross_refs, desc="SUPERSEDES"):
            ref_type = cref.get('properties', {}).get('reference_type', '')
            if 'supersede' in ref_type.lower():
                source_doc = cref['source_doc']
                target_doc = cref.get('properties', {}).get('target_doc', '')
                
                # Find matching regulations
                for reg1 in regulations:
                    if source_doc in reg1.get('properties', {}).get('name', ''):
                        for reg2 in regulations:
                            if target_doc in reg2.get('properties', {}).get('name', ''):
                                reg1_id = NodeIDGenerator.generate_id('Regulation', reg1['source_doc'], reg1['properties']['name'])
                                reg2_id = NodeIDGenerator.generate_id('Regulation', reg2['source_doc'], reg2['properties']['name'])
                                
                                try:
                                    self.client.execute_write_query("""
                                        MATCH (new:Regulation {id: $reg1_id})
                                        MATCH (old:Regulation {id: $reg2_id})
                                        MERGE (new)-[:SUPERSEDES]->(old)
                                    """, {'reg1_id': reg1_id, 'reg2_id': reg2_id})
                                    self.build_stats['relationships_created'] += 1
                                except Exception as e:
                                    logger.debug(f"Could not create SUPERSEDES relationship: {e}")
        
        logger.info(f"✓ Total relationships created: {self.build_stats['relationships_created']}")
    
    def _create_contains_relationships(self) -> int:
        """
        Create CONTAINS relationships from Regulations to Requirements.
        Uses source_doc to link them.
        """
        cypher = """
        MATCH (req:Requirement)
        WHERE req.source_doc IS NOT NULL
        MATCH (reg:Regulation {source_doc: req.source_doc})
        MERGE (reg)-[:CONTAINS]->(req)
        RETURN count(*) as count
        """
        
        try:
            result = self.client.execute_query(cypher)
            count = result[0]['count'] if result else 0
            return count
        except Exception as e:
            logger.error(f"Failed to create CONTAINS relationships: {e}")
            return 0
    
    def _create_supersedes_relationships(self, requirements: List[Dict]) -> int:
        """
        Detect SUPERSEDES relationships from requirement text patterns.
        Looks for "X replaced by Y", "X repealed by Y", etc.
        """
        supersedes_patterns = [
            r'([A-Z][a-z]+\s+(?:Directive|Regulation)\s+(?:\([A-Z]+\)\s+)?[0-9]{2,4}/[0-9]+/[A-Z]+)\s+(?:is\s+)?(repealed|replaced|superseded)\s+by\s+([A-Z][a-z]+\s+(?:Directive|Regulation)\s+(?:\([A-Z]+\)\s+)?[0-9]{2,4}/[0-9]+/[A-Z]+)',
            r'([0-9]{2,4}/[0-9]+/[A-Z]+)\s+(?:is\s+)?(repealed|replaced|superseded)\s+by\s+([0-9]{2,4}/[0-9]+/[A-Z]+)',
        ]
        
        supersedes_found = []
        
        for req in requirements:
            text = req['text']
            
            for pattern in supersedes_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    if len(match.groups()) >= 3:
                        old_reg = match.group(1)
                        new_reg = match.group(3)
                        supersedes_found.append({
                            'new': new_reg,
                            'old': old_reg,
                            'source_doc': req['source_doc'],
                            'evidence_text': text[:200]
                        })
        
        created_count = 0
        for item in supersedes_found:
            # Create nodes for specific directives/regulations
            self._ensure_specific_regulation_node(item['new'], item['source_doc'])
            self._ensure_specific_regulation_node(item['old'], item['source_doc'])
            
            # Create SUPERSEDES relationship
            cypher = """
            MATCH (new_reg:Regulation)
            WHERE new_reg.name CONTAINS $new_code OR new_reg.directive_code = $new_code
            MATCH (old_reg:Regulation)
            WHERE old_reg.name CONTAINS $old_code OR old_reg.directive_code = $old_code
            MERGE (new_reg)-[r:SUPERSEDES]->(old_reg)
            SET r.evidence = $evidence,
                r.detected_date = datetime()
            RETURN new_reg, old_reg
            """
            
            try:
                result = self.client.execute_write_query(cypher, {
                    'new_code': item['new'],
                    'old_code': item['old'],
                    'evidence': item['evidence_text']
                })
                if result:
                    created_count += 1
                    logger.info(f"    {item['new']} SUPERSEDES {item['old']}")
            except Exception as e:
                logger.debug(f"Could not create SUPERSEDES: {e}")
        
        return created_count
    
    def _ensure_specific_regulation_node(self, regulation_code: str, source_doc: str):
        """
        Ensure a Regulation node exists for a specific directive/regulation code.
        Example: "Commission Directive 2017/1572/EU"
        """
        # Extract just the code (e.g., "2017/1572/EU")
        code_match = re.search(r'([0-9]{2,4}/[0-9]+/[A-Z]+)', regulation_code)
        if not code_match:
            return
        
        code = code_match.group(1)
        
        # Determine issuing body
        issued_by = 'European Commission'
        if 'EU' in code:
            issued_by = 'European Union'
        elif 'EC' in code:
            issued_by = 'European Commission'
        
        embedding = self._generate_embedding(regulation_code)
        
        cypher = """
        MERGE (reg:Regulation {directive_code: $code})
        ON CREATE SET
            reg.id = $id,
            reg.name = $name,
            reg.issued_by = $issued_by,
            reg.source_doc = $source_doc,
            reg.embedding = $embedding,
            reg.created_at = datetime()
        RETURN reg
        """
        
        try:
            self.client.execute_write_query(cypher, {
                'code': code,
                'id': f"regulation_directive_{code.replace('/', '_')}",
                'name': regulation_code,
                'issued_by': issued_by,
                'source_doc': source_doc,
                'embedding': embedding
            })
        except Exception as e:
            logger.debug(f"Directive node may already exist for {code}: {e}")
    
    def _save_build_stats(self, output_path: Path):
        """Save graph build statistics."""
        stats = {
            'total_entities_processed': self.build_stats['total_entities'],
            'nodes_created': self.build_stats['nodes_created'],
            'relationships_created': self.build_stats['relationships_created'],
            'errors': self.build_stats['errors'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Get database stats
        try:
            db_stats = self.client.get_database_stats()
            stats.update({
                'total_nodes_in_db': db_stats['total_nodes'],
                'total_relationships_in_db': db_stats['total_relationships'],
                'nodes_by_label': db_stats['nodes_by_label'],
                'relationships_by_type': db_stats['relationships_by_type']
            })
        except Exception as e:
            logger.warning(f"Failed to get database stats: {e}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f" Build statistics saved to: {output_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("GRAPH BUILD SUMMARY")
        print("="*60)
        print(f"Entities Processed: {self.build_stats['total_entities']}")
        print(f"Nodes Created: {self.build_stats['nodes_created']}")
        print(f"Relationships Created: {self.build_stats['relationships_created']}")
        print(f"Errors: {self.build_stats['errors']}")
        if 'total_nodes_in_db' in stats:
            print(f"\nDatabase Status:")
            print(f"  Total Nodes: {stats['total_nodes_in_db']}")
            print(f"  Total Relationships: {stats['total_relationships_in_db']}")
        print("="*60)


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path
    
    # Add project root to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    parser = argparse.ArgumentParser(description='Build Neo4j knowledge graph from entities')
    parser.add_argument(
        '--input',
        type=str,
        default='data/processed_docs/entities.jsonl',
        help='Input JSONL file with entities'
    )
    
    args = parser.parse_args()
    
    try:
        builder = GraphBuilder()
        
        logger.info("Starting graph build from extracted entities...")
        stats = builder.build_from_entities_file(Path(args.input))
        
        logger.info(" Graph build complete!")
        
    except Exception as e:
        logger.error(f"Graph build failed: {e}")
        exit(1)

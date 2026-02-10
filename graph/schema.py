"""
Neo4j Graph Schema Definition
Defines all node types, relationships, properties, constraints, and indexes.
"""

import logging
from typing import List, Dict
from graph.neo4j_client import Neo4jClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphSchema:
    """
    Defines and manages the knowledge graph schema.
    """
    
    # Node Types
    NODE_TYPES = {
        "Regulation": "A regulatory document or standard (e.g., MHRA Blue Guide, EU GMP Annex 1)",
        "Section": "A section or chapter within a regulation",
        "Requirement": "A specific requirement or obligation from regulations",
        "Definition": "A defined term or concept",
        "Exemption": "An exception or exemption to a requirement",
        "Process": "A manufacturing or operational process",
        "Topic": "A thematic area or subject",
        "CrossReference": "A reference to another regulation or section"
    }
    
    # Relationship Types
    RELATIONSHIP_TYPES = {
        "CONTAINS": "A regulation/section contains another section/requirement",
        "REQUIRES": "A requirement mandates a specific action or condition",
        "REFERENCES": "Cross-reference to another regulation or section",
        "DEFINES": "Defines a term or concept",
        "APPLIES_TO": "A requirement applies to a specific process or topic",
        "SUPERSEDES": "A newer regulation replaces an older one",
        "CONFLICTS_WITH": "Two requirements contradict each other",
        "MENTIONS": "General mention or discussion of a topic"
    }
    
    # Node Properties (common to all nodes)
    COMMON_PROPERTIES = [
        "id",              # Unique identifier
        "name",            # Display name
        "text",            # Full text content
        "source_doc",      # Source document name
        "source_url",      # URL to source document
        "section_id",      # Section identifier (e.g., "Article 5", "Section 2.3")
        "citation_text",   # Text for citation display
        "created_at",      # Timestamp of node creation
        "embedding"        # Vector embedding for similarity search
    ]
    
    # Specific properties for each node type
    SPECIFIC_PROPERTIES = {
        "Regulation": ["jurisdiction", "effective_date", "version", "regulation_type"],
        "Section": ["level", "parent_section", "page_number"],
        "Requirement": ["category", "severity", "scope", "mandatory"],
        "Definition": ["term", "domain"],
        "Exemption": ["condition", "scope"],
        "Process": ["process_type", "industry"],
        "Topic": ["category", "keywords"],
        "CrossReference": ["target_doc", "target_section", "reference_type"]
    }
    
    def __init__(self, client: Neo4jClient):
        """
        Initialize schema manager.
        
        Args:
            client: Neo4jClient instance
        """
        self.client = client
    
    def create_constraints(self):
        """
        Create uniqueness constraints and indexes for optimal query performance.
        """
        logger.info("Creating database constraints and indexes...")
        
        constraints = [
            # Uniqueness constraints
            "CREATE CONSTRAINT regulation_id IF NOT EXISTS FOR (r:Regulation) REQUIRE r.id IS UNIQUE",
            "CREATE CONSTRAINT section_id IF NOT EXISTS FOR (s:Section) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT requirement_id IF NOT EXISTS FOR (req:Requirement) REQUIRE req.id IS UNIQUE",
            "CREATE CONSTRAINT definition_term IF NOT EXISTS FOR (d:Definition) REQUIRE d.term IS UNIQUE",
            "CREATE CONSTRAINT exemption_id IF NOT EXISTS FOR (e:Exemption) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT process_id IF NOT EXISTS FOR (p:Process) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT topic_id IF NOT EXISTS FOR (t:Topic) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT crossref_id IF NOT EXISTS FOR (cr:CrossReference) REQUIRE cr.id IS UNIQUE",
        ]
        
        for constraint in constraints:
            try:
                self.client.execute_write_query(constraint)
                logger.info(f"✓ Created constraint: {constraint.split('FOR')[1].split('REQUIRE')[0].strip()}")
            except Exception as e:
                # Constraint might already exist
                if "already exists" not in str(e).lower():
                    logger.warning(f"Constraint creation issue: {e}")
        
        # Text indexes for full-text search
        text_indexes = [
            "CREATE FULLTEXT INDEX regulation_text IF NOT EXISTS FOR (r:Regulation) ON EACH [r.name, r.text, r.citation_text]",
            "CREATE FULLTEXT INDEX requirement_text IF NOT EXISTS FOR (req:Requirement) ON EACH [req.text, req.citation_text]",
            "CREATE FULLTEXT INDEX definition_text IF NOT EXISTS FOR (d:Definition) ON EACH [d.term, d.text]",
        ]
        
        for index in text_indexes:
            try:
                self.client.execute_write_query(index)
                logger.info(f"✓ Created fulltext index")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Index creation issue: {e}")
        
        # Property indexes for common queries
        property_indexes = [
            "CREATE INDEX regulation_name IF NOT EXISTS FOR (r:Regulation) ON (r.name)",
            "CREATE INDEX requirement_category IF NOT EXISTS FOR (req:Requirement) ON (req.category)",
            "CREATE INDEX process_type IF NOT EXISTS FOR (p:Process) ON (p.process_type)",
            "CREATE INDEX section_source IF NOT EXISTS FOR (s:Section) ON (s.source_doc)",
        ]
        
        for index in property_indexes:
            try:
                self.client.execute_write_query(index)
                logger.info(f"✓ Created property index")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Index creation issue: {e}")
        
        logger.info("✓ Schema constraints and indexes created successfully")
    
    def verify_schema(self) -> Dict[str, List[str]]:
        """
        Verify that schema constraints and indexes are in place.
        
        Returns:
            Dictionary with constraints and indexes
        """
        # Get all constraints
        constraints_query = "SHOW CONSTRAINTS"
        constraints = self.client.execute_query(constraints_query)
        
        # Get all indexes
        indexes_query = "SHOW INDEXES"
        indexes = self.client.execute_query(indexes_query)
        
        return {
            "constraints": [c.get("name") for c in constraints],
            "indexes": [i.get("name") for i in indexes]
        }
    
    def create_sample_graph(self):
        """
        Create a sample subgraph for testing.
        Creates: 1 Regulation -> 2 Sections -> 3 Requirements -> 1 Process
        """
        logger.info("Creating sample graph for testing...")
        
        # Create sample regulation
        self.client.execute_write_query("""
        MERGE (r:Regulation {id: 'sample-reg-001'})
        SET r.name = 'MHRA GMP Guidelines',
            r.text = 'Good Manufacturing Practice guidelines for pharmaceutical manufacturing',
            r.source_doc = 'MHRA GMP',
            r.source_url = 'https://www.gov.uk/guidance/good-manufacturing-practice',
            r.citation_text = 'MHRA GMP Guidelines (2024)',
            r.jurisdiction = 'UK',
            r.effective_date = '2024-01-01',
            r.version = '1.0',
            r.regulation_type = 'GMP',
            r.created_at = datetime()
        """)
        
        # Create sample sections
        self.client.execute_write_query("""
        MATCH (r:Regulation {id: 'sample-reg-001'})
        MERGE (s1:Section {id: 'sample-sec-001'})
        SET s1.name = 'Section 5: Aseptic Processing',
            s1.text = 'Requirements for aseptic processing in sterile manufacturing',
            s1.source_doc = 'MHRA GMP',
            s1.source_url = 'https://www.gov.uk/guidance/good-manufacturing-practice',
            s1.section_id = 'Section 5',
            s1.citation_text = 'MHRA GMP Section 5',
            s1.level = 1,
            s1.created_at = datetime()
        MERGE (s2:Section {id: 'sample-sec-002'})
        SET s2.name = 'Section 5.1: Environmental Control',
            s2.text = 'Requirements for environmental monitoring in cleanrooms',
            s2.source_doc = 'MHRA GMP',
            s2.source_url = 'https://www.gov.uk/guidance/good-manufacturing-practice',
            s2.section_id = 'Section 5.1',
            s2.citation_text = 'MHRA GMP Section 5.1',
            s2.level = 2,
            s2.created_at = datetime()
        MERGE (r)-[:CONTAINS]->(s1)
        MERGE (s1)-[:CONTAINS]->(s2)
        """)
        
        # Create sample requirements
        self.client.execute_write_query("""
        MATCH (s:Section {id: 'sample-sec-002'})
        MERGE (req1:Requirement {id: 'sample-req-001'})
        SET req1.name = 'Grade A Cleanroom Requirement',
            req1.text = 'Aseptic filling operations must be performed in Grade A cleanroom environments with continuous particle monitoring',
            req1.source_doc = 'MHRA GMP',
            req1.source_url = 'https://www.gov.uk/guidance/good-manufacturing-practice',
            req1.section_id = 'Section 5.1',
            req1.citation_text = 'MHRA GMP Section 5.1, Para 1',
            req1.category = 'Environmental Control',
            req1.severity = 'Critical',
            req1.scope = 'Aseptic Processing',
            req1.mandatory = true,
            req1.created_at = datetime()
        MERGE (req2:Requirement {id: 'sample-req-002'})
        SET req2.name = 'Personnel Qualification Requirement',
            req2.text = 'Personnel performing aseptic operations must undergo aseptic technique qualification and annual requalification',
            req2.source_doc = 'MHRA GMP',
            req2.source_url = 'https://www.gov.uk/guidance/good-manufacturing-practice',
            req2.section_id = 'Section 5.1',
            req2.citation_text = 'MHRA GMP Section 5.1, Para 3',
            req2.category = 'Personnel',
            req2.severity = 'Critical',
            req2.scope = 'Aseptic Processing',
            req2.mandatory = true,
            req2.created_at = datetime()
        MERGE (req3:Requirement {id: 'sample-req-003'})
        SET req3.name = 'Media Fill Validation',
            req3.text = 'Aseptic filling processes must be validated using media fill simulation with defined acceptance criteria',
            req3.source_doc = 'MHRA GMP',
            req3.source_url = 'https://www.gov.uk/guidance/good-manufacturing-practice',
            req3.section_id = 'Section 5.2',
            req3.citation_text = 'MHRA GMP Section 5.2, Para 1',
            req3.category = 'Validation',
            req3.severity = 'Critical',
            req3.scope = 'Aseptic Processing',
            req3.mandatory = true,
            req3.created_at = datetime()
        MERGE (s)-[:CONTAINS]->(req1)
        MERGE (s)-[:CONTAINS]->(req2)
        MERGE (s)-[:CONTAINS]->(req3)
        """)
        
        # Create sample process and relationships
        self.client.execute_write_query("""
        MATCH (req1:Requirement {id: 'sample-req-001'})
        MATCH (req2:Requirement {id: 'sample-req-002'})
        MATCH (req3:Requirement {id: 'sample-req-003'})
        MERGE (p:Process {id: 'sample-proc-001'})
        SET p.name = 'Aseptic Filling',
            p.text = 'Process of filling sterile products into containers under aseptic conditions',
            p.process_type = 'Manufacturing',
            p.industry = 'Pharmaceutical',
            p.created_at = datetime()
        MERGE (req1)-[:APPLIES_TO]->(p)
        MERGE (req2)-[:APPLIES_TO]->(p)
        MERGE (req3)-[:APPLIES_TO]->(p)
        """)
        
        logger.info("✓ Sample graph created successfully")
        
        # Verify creation
        stats = self.client.get_database_stats()
        logger.info(f"  Nodes: {stats['total_nodes']}")
        logger.info(f"  Relationships: {stats['total_relationships']}")
    
    def get_schema_documentation(self) -> str:
        """
        Generate human-readable schema documentation.
        
        Returns:
            Markdown-formatted schema documentation
        """
        doc = "# Knowledge Graph Schema\n\n"
        
        doc += "## Node Types\n\n"
        for node_type, description in self.NODE_TYPES.items():
            doc += f"### {node_type}\n"
            doc += f"{description}\n\n"
            doc += "**Properties:**\n"
            doc += "- " + "\n- ".join(self.COMMON_PROPERTIES) + "\n"
            if node_type in self.SPECIFIC_PROPERTIES:
                doc += "- " + "\n- ".join(self.SPECIFIC_PROPERTIES[node_type]) + "\n"
            doc += "\n"
        
        doc += "## Relationship Types\n\n"
        for rel_type, description in self.RELATIONSHIP_TYPES.items():
            doc += f"### {rel_type}\n"
            doc += f"{description}\n\n"
        
        return doc


if __name__ == "__main__":
    # Test schema creation
    from graph.neo4j_client import Neo4jClient
    
    try:
        client = Neo4jClient()
        schema = GraphSchema(client)
        
        # Create schema
        schema.create_constraints()
        
        # Verify schema
        verification = schema.verify_schema()
        print(f"\n✓ Schema verified:")
        print(f"  Constraints: {len(verification['constraints'])}")
        print(f"  Indexes: {len(verification['indexes'])}")
        
        # Create sample graph
        schema.create_sample_graph()
        
        # Test query: Find requirements for aseptic filling
        print("\n--- Test Query: Requirements for Aseptic Filling ---")
        query = """
        MATCH (p:Process {name: 'Aseptic Filling'})<-[:APPLIES_TO]-(req:Requirement)
        RETURN req.name AS requirement, req.citation_text AS citation, req.severity AS severity
        """
        results = client.execute_query(query)
        for result in results:
            print(f"  • {result['requirement']}")
            print(f"    Citation: {result['citation']}")
            print(f"    Severity: {result['severity']}\n")
        
        client.close()
        print("✓ Schema test completed successfully")
        
    except Exception as e:
        print(f"✗ Schema test failed: {e}")

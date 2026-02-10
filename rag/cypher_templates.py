"""
Cypher Templates
Pre-defined Cypher query templates for common regulatory queries.
"""

from typing import Dict, List, Optional
from enum import Enum


class QueryType(Enum):
    """Types of regulatory queries."""
    APPLICABILITY = "applicability"
    REQUIREMENTS = "requirements"
    CROSS_REFERENCE = "cross_reference"
    CONFLICT = "conflict"
    DEPENDENCY = "dependency"
    DEFINITION = "definition"
    EXEMPTION = "exemption"


class CypherTemplates:
    """Pre-defined Cypher query templates."""
    
    # Query 1: Find requirements that apply to a specific process
    REQUIREMENTS_FOR_PROCESS = """
    MATCH (req:Requirement)-[:APPLIES_TO]->(proc:Process)
    WHERE toLower(proc.name) CONTAINS toLower($process_name)
    RETURN req.text AS requirement,
           req.citation_text AS source,
           req.category AS category,
           req.severity AS severity,
           req.mandatory AS mandatory,
           proc.name AS process
    ORDER BY req.severity DESC
    LIMIT $limit
    """
    
    # Query 2: Check if regulation applies to specific context
    REGULATION_APPLICABILITY = """
    MATCH (req:Requirement)
    WHERE toLower(req.text) CONTAINS toLower($keyword)
       OR toLower(req.scope) CONTAINS toLower($keyword)
    OPTIONAL MATCH (req)-[:APPLIES_TO]->(proc:Process)
    RETURN req.text AS requirement,
           req.citation_text AS source,
           req.scope AS scope,
           collect(DISTINCT proc.name) AS applicable_processes
    LIMIT $limit
    """
    
    # Query 3: Find cross-references between documents
    CROSS_REFERENCES = """
    MATCH (cref:CrossReference)
    WHERE toLower(cref.source_doc) CONTAINS toLower($source_doc)
       OR toLower(cref.target_doc) CONTAINS toLower($target_doc)
    RETURN cref.source_doc AS from_document,
           cref.target_doc AS to_document,
           cref.target_section AS section,
           cref.reference_type AS type,
           cref.text AS context
    LIMIT $limit
    """
    
    # Query 4: Find definitions of specific terms
    TERM_DEFINITIONS = """
    MATCH (defn:Definition)
    WHERE toLower(defn.term) CONTAINS toLower($term)
    OPTIONAL MATCH (defn)-[:DEFINES]->(req:Requirement)
    RETURN defn.term AS term,
           defn.text AS definition,
           defn.citation_text AS source,
           defn.domain AS domain,
           collect(DISTINCT req.text)[0..3] AS related_requirements
    LIMIT $limit
    """
    
    # Query 5: Find conflicting requirements
    DETECT_CONFLICTS = """
    MATCH (req1:Requirement)-[:CONFLICTS_WITH]->(req2:Requirement)
    WHERE toLower(req1.text) CONTAINS toLower($keyword)
       OR toLower(req2.text) CONTAINS toLower($keyword)
    RETURN req1.text AS requirement1,
           req1.citation_text AS source1,
           req2.text AS requirement2,
           req2.citation_text AS source2
    LIMIT $limit
    """
    
    # Query 6: Find exemptions for specific scenarios
    FIND_EXEMPTIONS = """
    MATCH (exempt:Exemption)
    WHERE toLower(exempt.text) CONTAINS toLower($keyword)
       OR toLower(exempt.condition) CONTAINS toLower($keyword)
       OR toLower(exempt.scope) CONTAINS toLower($keyword)
    RETURN exempt.text AS exemption,
           exempt.citation_text AS source,
           exempt.condition AS condition,
           exempt.scope AS scope
    LIMIT $limit
    """
    
    # Query 7: Trace dependency chains
    DEPENDENCY_CHAIN = """
    MATCH path = (req:Requirement)-[:REQUIRES*1..3]->(dep:Requirement)
    WHERE toLower(req.text) CONTAINS toLower($keyword)
    RETURN [node IN nodes(path) | node.text] AS dependency_chain,
           [node IN nodes(path) | node.citation_text] AS sources,
           length(path) AS chain_length
    ORDER BY chain_length
    LIMIT $limit
    """
    
    # Query 8: Find related requirements via graph traversal
    RELATED_REQUIREMENTS = """
    MATCH (req:Requirement)
    WHERE id(req) = $node_id
    OPTIONAL MATCH path1 = (req)-[:APPLIES_TO]->(proc:Process)<-[:APPLIES_TO]-(related:Requirement)
    OPTIONAL MATCH path2 = (req)<-[:DEFINES]-(defn:Definition)-[:DEFINES]->(related2:Requirement)
    WITH req, 
         collect(DISTINCT related) + collect(DISTINCT related2) AS related_reqs
    UNWIND related_reqs AS related_req
    RETURN DISTINCT related_req.text AS requirement,
           related_req.citation_text AS source,
           related_req.category AS category
    LIMIT $limit
    """
    
    # Query 9: Full-text search across all requirements
    SEARCH_REQUIREMENTS = """
    MATCH (req:Requirement)
    WHERE toLower(req.text) CONTAINS toLower($search_term)
    OPTIONAL MATCH (req)-[:APPLIES_TO]->(proc:Process)
    RETURN req.text AS requirement,
           req.citation_text AS source,
           req.category AS category,
           req.severity AS severity,
           collect(DISTINCT proc.name) AS processes
    ORDER BY req.severity DESC
    LIMIT $limit
    """
    
    # Query 10: Get context for a specific citation
    CITATION_CONTEXT = """
    MATCH (node)
    WHERE node.citation_text = $citation
       OR node.source_doc = $citation
    OPTIONAL MATCH (node)-[rel]->(related)
    RETURN labels(node)[0] AS entity_type,
           node.text AS content,
           node.citation_text AS source,
           collect(DISTINCT {
               type: type(rel),
               target: labels(related)[0],
               text: related.text
           }) AS relationships
    LIMIT $limit
    """
    
    @classmethod
    def get_template(cls, query_type: QueryType) -> str:
        """
        Get template for specific query type.
        
        Args:
            query_type: Type of query
            
        Returns:
            Cypher query template
        """
        mapping = {
            QueryType.APPLICABILITY: cls.REGULATION_APPLICABILITY,
            QueryType.REQUIREMENTS: cls.REQUIREMENTS_FOR_PROCESS,
            QueryType.CROSS_REFERENCE: cls.CROSS_REFERENCES,
            QueryType.CONFLICT: cls.DETECT_CONFLICTS,
            QueryType.DEPENDENCY: cls.DEPENDENCY_CHAIN,
            QueryType.DEFINITION: cls.TERM_DEFINITIONS,
            QueryType.EXEMPTION: cls.FIND_EXEMPTIONS
        }
        return mapping.get(query_type, cls.SEARCH_REQUIREMENTS)
    
    @classmethod
    def get_all_templates(cls) -> Dict[str, str]:
        """Get all available query templates."""
        return {
            'requirements_for_process': cls.REQUIREMENTS_FOR_PROCESS,
            'regulation_applicability': cls.REGULATION_APPLICABILITY,
            'cross_references': cls.CROSS_REFERENCES,
            'term_definitions': cls.TERM_DEFINITIONS,
            'detect_conflicts': cls.DETECT_CONFLICTS,
            'find_exemptions': cls.FIND_EXEMPTIONS,
            'dependency_chain': cls.DEPENDENCY_CHAIN,
            'related_requirements': cls.RELATED_REQUIREMENTS,
            'search_requirements': cls.SEARCH_REQUIREMENTS,
            'citation_context': cls.CITATION_CONTEXT
        }


# Example usage
if __name__ == "__main__":
    templates = CypherTemplates.get_all_templates()
    print(f"Available templates: {len(templates)}")
    for name, template in templates.items():
        print(f"\n{name}:")
        print(template[:200] + "...")

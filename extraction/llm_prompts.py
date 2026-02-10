"""
LLM Prompts for Entity and Relationship Extraction
Carefully engineered prompts for accurate extraction from regulatory documents.
"""

# Entity Extraction Prompt
ENTITY_EXTRACTION_PROMPT = """You are an expert in pharmaceutical regulatory compliance. Extract structured entities from the following regulatory document text.

Extract the following entity types:

1. **Requirements**: Specific obligations, mandates, or specifications that must be met
   - Include the full requirement text
   - Classify category (e.g., "Environmental Control", "Personnel", "Validation", "Documentation")
   - Assess severity (Critical, Major, Minor)
   - Determine scope (what it applies to)

2. **Definitions**: Terms that are explicitly defined in the text
   - Extract the term being defined
   - Extract the full definition
   - Note the domain/category

3. **Exemptions**: Exceptions or conditions where requirements don't apply
   - Extract the exemption text
   - Note conditions and scope

4. **Processes**: Manufacturing or operational processes mentioned
   - Process name
   - Process type
   - Industry category

5. **Regulations**: Regulatory documents or standards themselves
   - Regulation name (e.g., "EU GMP Annex 1", "MHRA Blue Guide", "ICH Q7")
   - Issued by (regulatory authority like "European Commission", "MHRA", "ICH")
   - Effective date (if mentioned in text)
   - Version or revision number

6. **Cross-References**: References to other regulations, sections, or documents
   - Source reference
   - Target document/section
   - Reference type (cites, supersedes, conflicts with, etc.)

**Document Context:**
Source: {source}
Section: {section_id} - {section_heading}

**Text to Analyze:**
{text}

**Output Format (JSON):**
```json
{{
  "requirements": [
    {{
      "text": "full requirement text",
      "category": "category name",
      "severity": "Critical|Major|Minor",
      "scope": "what it applies to",
      "mandatory": true|false
    }}
  ],
  "definitions": [
    {{
      "term": "term being defined",
      "definition": "full definition text",
      "domain": "subject area"
    }}
  ],
  "exemptions": [
    {{
      "text": "exemption text",
      "condition": "conditions that apply",
      "scope": "what is exempted"
    }}
  ],
  "processes": [
    {{
      "name": "process name",
      "process_type": "Manufacturing|Quality Control|Documentation|Other",
      "description": "brief description"
    }}
  ],
  "regulations": [
    {{
      "name": "regulation name",
      "issued_by": "regulatory authority",
      "effective_date": "YYYY-MM-DD or empty string",
      "version": "version number or empty string"
    }}
  ],
  "cross_references": [
    {{
      "reference_text": "text of the reference",
      "target_doc": "referenced document name",
      "target_section": "section if specified",
      "reference_type": "cites|supersedes|refers_to"
    }}
  ]
}}
```

**CRITICAL Instructions:**
- Only extract entities explicitly mentioned in the text
- Be CONCISE: Keep text fields under 200 characters when possible
- OMIT empty arrays - only include entity types that have actual content
- For requirements, accurately assess severity based on regulatory language (shall=Critical, should=Major, may=Minor)
- Return VALID, COMPLETE JSON only - ensure all strings are properly closed
- If text is very long, prioritize the most important entities to fit within token limits
"""


# Relationship Extraction Prompt
RELATIONSHIP_EXTRACTION_PROMPT = """You are an expert in pharmaceutical regulatory compliance. Analyze the relationships between entities in regulatory documents.

Given these extracted entities from document "{source}", section "{section_heading}":

**Requirements:**
{requirements}

**Processes:**
{processes}

**Definitions:**
{definitions}

**Context Text:**
{text}

**Task:** Identify relationships between entities:

1. **APPLIES_TO**: Requirements that apply to specific processes
   - Which requirements govern which processes?

2. **REQUIRES**: Requirements that mandate other requirements
   - Which requirements depend on others?

3. **DEFINES**: Definitions that clarify terms used in requirements
   - Which definitions explain terms in which requirements?

4. **REFERENCES**: Cross-references between sections/documents
   - What other regulations/sections are cited?

**Output Format (JSON):**
```json
{{
  "relationships": [
    {{
      "source_entity": "entity ID or text",
      "relationship_type": "APPLIES_TO|REQUIRES|DEFINES|REFERENCES",
      "target_entity": "entity ID or text",
      "confidence": 0.0-1.0,
      "evidence": "text evidence for this relationship"
    }}
  ]
}}
```

**Instructions:**
- Only extract relationships explicitly supported by the text
- Provide confidence scores (1.0 = explicit, 0.7-0.9 = strong inference, <0.7 = weak inference)
- Include evidence text that supports each relationship
- Return valid JSON only
"""


# Conflict Detection Prompt
CONFLICT_DETECTION_PROMPT = """You are an expert in pharmaceutical regulatory compliance. Analyze potential conflicts between requirements.

Given these requirements from different regulatory sources:

**Requirement 1:**
Source: {source1}
Section: {section1}
Text: {text1}

**Requirement 2:**
Source: {source2}
Section: {section2}
Text: {text2}

**Task:** Determine if these requirements conflict, contradict, or are inconsistent.

Consider:
- Do they specify different values/thresholds for the same thing?
- Do they mandate contradictory actions?
- Are the scopes overlapping but requirements different?
- Do newer regulations supersede older ones?

**Output Format (JSON):**
```json
{{
  "conflict_detected": true|false,
  "conflict_type": "contradiction|inconsistency|superseded|scope_overlap|none",
  "explanation": "detailed explanation of the conflict",
  "severity": "High|Medium|Low",
  "resolution_suggestion": "how to resolve this conflict"
}}
```

**Instructions:**
- Be conservative - only flag clear conflicts
- Consider temporal aspects (newer regulations may supersede older)
- Distinguish between contradictions (impossible to comply with both) and inconsistencies (different but compatible)
- Return valid JSON only
"""


# Cypher Query Generation Prompt
CYPHER_GENERATION_PROMPT = """You are an expert in Neo4j Cypher queries for knowledge graphs of regulatory documents.

**Graph Schema:**
Nodes: Regulation, Section, Requirement, Definition, Exemption, Process, Topic, CrossReference
Relationships: CONTAINS, REQUIRES, REFERENCES, DEFINES, APPLIES_TO, SUPERSEDES, CONFLICTS_WITH, MENTIONS

**Node Properties:**
- All nodes have: id, name, text, source_doc, source_url, section_id, citation_text
- Requirement: category, severity, scope, mandatory
- Definition: term, domain
- Process: process_type, industry

**User Question:**
{question}

**Task:** Generate a Cypher query to answer this question.

**Requirements:**
1. Use MATCH clauses to traverse the graph
2. Include relevant WHERE clauses for filtering
3. RETURN nodes, relationships, and paths as appropriate
4. Limit results to top 10-20 most relevant
5. Order by relevance where applicable

**Output Format (JSON):**
```json
{{
  "cypher_query": "MATCH ... WHERE ... RETURN ... LIMIT ...",
  "explanation": "what this query does in simple terms",
  "expected_result_type": "nodes|relationships|paths|counts"
}}
```

**Example Queries:**

<function_calls>User: "What requirements apply to aseptic filling?"
Cypher: 
```cypher
MATCH (p:Process {{name: 'Aseptic Filling'}})<-[:APPLIES_TO]-(req:Requirement)
RETURN req.name, req.text, req.citation_text, req.severity
ORDER BY req.severity DESC
LIMIT 20
```

User: "Find cross-references between MHRA and EU GMP"
Cypher:
```cypher
MATCH (r1:Regulation)-[:REFERENCES]->(r2:Regulation)
WHERE r1.source_doc CONTAINS 'MHRA' AND r2.source_doc CONTAINS 'EU GMP'
RETURN r1.name, r2.name, r1.citation_text, r2.citation_text
LIMIT 20
```

User: "Trace dependency chain for requirement X"
Cypher:
```cypher
MATCH path = (req:Requirement {{name: 'X'}})-[:REQUIRES*1..3]->(dep:Requirement)
RETURN path, length(path) AS depth
ORDER BY depth
LIMIT 20
```

**Instructions:**
- Generate syntactically correct Cypher
- Use parameterized queries where appropriate
- Consider using OPTIONAL MATCH for flexible queries
- Return valid JSON only
"""


# Few-shot Examples for Entity Extraction
ENTITY_EXTRACTION_EXAMPLES = """
**Example 1:**

Input Text: "Aseptic processing areas shall be Grade A with a Grade B background. Particle monitoring shall be continuous using a system with an immediate alarm threshold."

Output:
```json
{{
  "requirements": [
    {{
      "text": "Aseptic processing areas shall be Grade A with a Grade B background",
      "category": "Environmental Control",
      "severity": "Critical",
      "scope": "Aseptic Processing",
      "mandatory": true
    }},
    {{
      "text": "Particle monitoring shall be continuous using a system with an immediate alarm threshold",
      "category": "Environmental Monitoring",
      "severity": "Critical",
      "scope": "Cleanroom Monitoring",
      "mandatory": true
    }}
  ],
  "processes": [
    {{
      "name": "Aseptic Processing",
      "process_type": "Manufacturing",
      "description": "Sterile product processing in controlled environment"
    }}
  ]
}}
```

**Example 2:**

Input Text: "For the purposes of this Annex, 'validation' means documented evidence that a process consistently produces a result meeting its predetermined specifications."

Output:
```json
{{
  "definitions": [
    {{
      "term": "validation",
      "definition": "documented evidence that a process consistently produces a result meeting its predetermined specifications",
      "domain": "Quality Assurance"
    }}
  ]
}}
```

**Example 3:**

Input Text: "As referenced in EU GMP Annex 1, Section 4.12, personnel qualification requirements align with those specified in MHRA Blue Guide Chapter 2."

Output:
```json
{{
  "cross_references": [
    {{
      "reference_text": "As referenced in EU GMP Annex 1, Section 4.12",
      "target_doc": "EU GMP Annex 1",
      "target_section": "Section 4.12",
      "reference_type": "cites"
    }},
    {{
      "reference_text": "MHRA Blue Guide Chapter 2",
      "target_doc": "MHRA Blue Guide",
      "target_section": "Chapter 2",
      "reference_type": "cites"
    }}
  ]
}}
```
"""


def get_entity_extraction_prompt(text: str, source: str, section_id: str, section_heading: str) -> str:
    """
    Get entity extraction prompt with context.
    
    Args:
        text: Text to analyze
        source: Source document name
        section_id: Section identifier
        section_heading: Section heading
        
    Returns:
        Formatted prompt string
    """
    return ENTITY_EXTRACTION_PROMPT.format(
        text=text,
        source=source,
        section_id=section_id,
        section_heading=section_heading
    )


def get_relationship_extraction_prompt(text: str, source: str, section_heading: str,
                                       requirements: str, processes: str, definitions: str) -> str:
    """
    Get relationship extraction prompt with context.
    
    Args:
        text: Context text
        source: Source document name
        section_heading: Section heading
        requirements: JSON string of requirements
        processes: JSON string of processes
        definitions: JSON string of definitions
        
    Returns:
        Formatted prompt string
    """
    return RELATIONSHIP_EXTRACTION_PROMPT.format(
        text=text,
        source=source,
        section_heading=section_heading,
        requirements=requirements,
        processes=processes,
        definitions=definitions
    )


def get_conflict_detection_prompt(source1: str, section1: str, text1: str,
                                  source2: str, section2: str, text2: str) -> str:
    """
    Get conflict detection prompt.
    
    Args:
        source1: First source document
        section1: First section
        text1: First requirement text
        source2: Second source document
        section2: Second section
        text2: Second requirement text
        
    Returns:
        Formatted prompt string
    """
    return CONFLICT_DETECTION_PROMPT.format(
        source1=source1,
        section1=section1,
        text1=text1,
        source2=source2,
        section2=section2,
        text2=text2
    )


def get_cypher_generation_prompt(question: str) -> str:
    """
    Get Cypher query generation prompt.
    
    Args:
        question: User's natural language question
        
    Returns:
        Formatted prompt string
    """
    return CYPHER_GENERATION_PROMPT.format(question=question)

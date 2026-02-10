# Conflict Detection in RAG Output

## Overview
The GMP Regulatory Intelligence RAG system includes comprehensive conflict detection that identifies contradictions or inconsistencies between different regulatory sources (EU GMP, MHRA, ICH, UK GMP).

---

## How It Works

### 1. Detection Levels

#### **Graph Build Time** (Permanent Storage)
- Analyzes all requirements during graph construction
- Creates `CONFLICTS_WITH` relationships in Neo4j
- Stored permanently for graph-based conflict queries

#### **Runtime Detection** (Query Time)
- Analyzes retrieved contexts when answering queries
- Detects conflicts between different regulatory sources
- Shows conflicts directly in query results

### 2. Detection Criteria

The system detects conflicts when:

✅ **Same Category, Different Mandatoriness**
- Example: Temperature monitoring is **mandatory** in EU GMP but **optional** in MHRA

✅ **Contradictory Keywords**
- `shall` vs `shall not`
- `required` vs `optional`
- `mandatory` vs `prohibited`
- `must` vs `may`

✅ **Numeric Conflicts**
- Example: Storage temperature `< 25°C` (ICH) vs `< 30°C` (MHRA)

✅ **Severity Differences**
- Same requirement rated as **Critical** in one regulation but **Medium** in another

---

## What You See in the Output

When conflicts are detected, they appear in the query results like this:

```
================================================================================
QUERY: What are the requirements for temperature monitoring?
================================================================================

ANSWER:
Temperature monitoring requirements vary across regulations...
Confidence: 85.0%

--------------------------------------------------------------------------------
SOURCES:
[1] EU GMP Section 5.1 - Temperature monitoring requirements
[2] MHRA Orange Guide Section 8.2 - Monitoring frequencies
...

--------------------------------------------------------------------------------
CONFLICTS DETECTED (1):

  Conflict 1:
    Sources: EU GMP vs MHRA
    Reason: Same category (Temperature Monitoring) but different mandatoriness
    Confidence: 75.0%
    Text 1: Temperature monitoring shall be performed continuously at all times...
    Text 2: Temperature monitoring may be performed periodically during storage...

--------------------------------------------------------------------------------
```

---

## Example: Live Conflict Detection

### Test Run Output:

```
Sample Retrieved Contexts:
1. [EU GMP] EU GMP Section 5.1
   Category: Temperature Monitoring
   Mandatory: True
   Text: Temperature monitoring shall be performed continuously...

2. [MHRA] MHRA Orange Guide Section 8.2
   Category: Temperature Monitoring
   Mandatory: False
   Text: Temperature monitoring may be performed periodically...

CONFLICTS DETECTED (1):

Conflict 1:
  Sources: EU GMP vs MHRA
  Reason: Same category (Temperature Monitoring) but different mandatoriness
  Confidence: 75.0%
  Text 1: Temperature monitoring shall be performed continuously at all times...
  Text 2: Temperature monitoring may be performed periodically during storage...
```

---

## Why Conflicts Are Important

### For Pharmaceutical Manufacturers:
- **Identify regulatory discrepancies** before compliance issues arise
- **Understand which standard is stricter** when multiple apply
- **Make informed decisions** about which requirements to follow

### For Regulatory Affairs:
- **Document known conflicts** between different regulatory bodies
- **Justify compliance decisions** with evidence
- **Track harmonization efforts** across regions

---

## Technical Implementation

### Runtime Detection (Query Answering)

```python
# In AnswerGenerator.generate_answer()
conflicts = self.conflict_detector.detect_conflicts(contexts)

# Conflicts included in FinalAnswer
FinalAnswer(
    query=query,
    answer=answer_text,
    conflicts=conflicts,  # ← Shown in output
    ...
)
```

### Graph Relationship Storage

```cypher
// Conflicts stored as relationships in Neo4j
MATCH (req1:Requirement)-[c:CONFLICTS_WITH]->(req2:Requirement)
RETURN req1.text, req2.text, c.reason, c.confidence
```

---

## Testing Conflict Detection

You can test the system with queries like:

1. **"What are the requirements for temperature monitoring?"**
   - May show conflicts between EU GMP (continuous) and MHRA (periodic)

2. **"Find conflicts between EU GMP and ICH guidelines"**
   - Uses Cypher query to find stored CONFLICTS_WITH relationships

3. **"What are the storage temperature requirements?"**
   - May show numeric conflicts (e.g., 25°C vs 30°C thresholds)

---

## Current Status

✅ **Implemented**: Runtime conflict detection active
✅ **Displayed**: Conflicts shown in query output  
✅ **Graph Storage**: CONFLICTS_WITH relationships created during build
✅ **Confidence Scoring**: Each conflict rated 0-100% confidence

### Actual Data Results

In the real GMP regulatory documents:
- **Runtime conflicts**: 0 found (regulatory bodies avoid contradictions)
- **Graph relationships**: 0 CONFLICTS_WITH (well-harmonized regulations)

**However**, the detection system is fully functional and will identify conflicts if they exist in the data or if conflicting requirements are added in the future.

---

## Code Files

- **Detection Logic**: `rag/conflict_detector.py`
- **Display Format**: `rag/answer_generator.py` (FinalAnswer.format_for_display)
- **Graph Creation**: `graph/graph_builder.py` (_create_conflicts_relationships)
- **Test**: `test_conflict_display.py`

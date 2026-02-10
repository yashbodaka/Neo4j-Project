# RAG System Failure Modes and Error Handling

## Overview
This document describes the failure modes, error handling strategies, and expected behaviors of the GMP Regulatory Intelligence RAG system when encountering various error conditions.

---

## 1. Query with No Results

### Scenario
User asks a question for which the system cannot find relevant information in the knowledge base.

### Example Queries
- "What are the requirements for nuclear reactor safety?" (out of domain)
- "What did regulation XYZ-999 say about topic ABC?" (regulation doesn't exist)
- "What are the guidelines for temperature monitoring?" (if not in extracted data)

### System Behavior

#### Detection
- **Retriever**: Returns 0 contexts or all contexts with very low similarity scores (<0.3)
- **Cypher Generator**: Generates query that returns no results
- **Reasoning Chain**: Receives minimal or no relevant context

#### Response
The system provides an honest, low-confidence response:

```json
{
  "answer": "Based on the provided regulatory context, it is not possible to find specific information about [topic]. The retrieved documents do not contain requirements, definitions, or guidelines related to [specific query terms].",
  "confidence": 0.1,
  "sources": [],
  "graph_path_summary": "No relevant graph path found.",
  "related_requirements": []
}
```

#### Confidence Score
- **0.1 - 0.2**: Indicates no relevant information found
- System explicitly states limitation in the answer text

#### User Experience
- Clear indication that no information was found
- No hallucinated or invented information
- Low confidence score signals uncertainty
- Empty or minimal sources list

---

## 2. Ambiguous Query Handling

### Scenario
User query is vague, has multiple interpretations, or lacks sufficient context.

### Example Queries
- "Tell me about GMP" (too broad)
- "What are the requirements?" (missing subject)
- "EU vs MHRA guidelines" (unclear what to compare)
- "When was it updated?" (unclear referent)

### System Behavior

#### Query Classification
The **Query Router** attempts to classify into one of four types:
1. **definition** - seeking term definitions
2. **cross_reference** - seeking links between regulations
3. **metadata** - seeking provenance/version info
4. **requirement** - seeking specific requirements

For ambiguous queries:
- Router uses Gemini 3 Flash Preview with instruction to classify even if unclear
- Falls back to "requirement" as default type if classification fails

#### Retrieval Strategy
- **Hybrid retrieval** attempts to find relevant contexts using:
  - Vector similarity search (semantic matching)
  - Keyword-based search (BM25)
  - Graph traversal from keyword matches
- Retrieves multiple contexts (top_k=5-8) to capture different interpretations

#### Response Generation
- **Reasoning Chain** synthesizes available contexts
- May provide multiple interpretations if contexts support them
- Confidence score reflects ambiguity level:
  - **0.5-0.7**: Query was ambiguous but some relevant info found
  - **0.7-0.9**: Query sufficiently clear with good context
  - **0.9-1.0**: Query very clear with strong, specific matches

#### Example Response for Ambiguous Query

**Query**: "Tell me about aseptic filling"

**Response**:
```json
{
  "answer": "Aseptic filling is defined as the process of filling sterile products into containers under aseptic conditions [Source: ICH section_19.5]. For aseptic filling operations, GMP requirements mandate performance in Grade A cleanroom environments with continuous particle monitoring [Source: MHRA GMP Section 5.1].",
  "confidence": 0.85,
  "sources": [...]
}
```

The system:
1. Interpreted query as both definition and requirement request
2. Retrieved contexts for both interpretations
3. Provided definition first, then requirements
4. High confidence (0.85) indicates successful interpretation

---

## 3. API Rate Limit Errors

### Scenario
System exceeds rate limits for LLM API providers (Gemini, Groq).

### Affected Components
1. **Query Router** - Gemini 3 Flash Preview
2. **Reasoning Chain** - Gemini hierarchy (gemini-3-flash-preview → gemini-2.5-flash)
3. **Cypher Generator** - Groq (llama-3.3-70b-versatile)

### Rate Limit Thresholds

#### Google Gemini API
- **Free Tier**: 15 requests per minute (RPM), 1 million tokens per minute (TPM)
- **Error Code**: `429 RESOURCE_EXHAUSTED`

#### Groq API  
- **Free Tier**: 30 requests per minute (RPM), 7,000 tokens per minute (TPM)
- **Error Code**: `429 Rate limit exceeded`

### System Behavior

#### Reasoning Chain Fallback Strategy
The Reasoning Chain uses a **hierarchical fallback mechanism**:

```python
models = ["gemini-3-flash-preview", "gemini-2.5-flash"]
```

**Process**:
1. Attempts with `gemini-3-flash-preview`
2. If rate limit hit (503 or 429 error), logs warning and tries next model
3. Falls back to `gemini-2.5-flash`
4. If both fail, raises exception

**Example Log Output**:
```
WARNING:rag.reasoning_chain:Model gemini-3-flash-preview failed: 503 UNAVAILABLE. 
{'error': {'code': 503, 'message': 'This model is currently experiencing high demand. 
Spikes in demand are usually temporary. Please try again later.', 'status': 'UNAVAILABLE'}}. 
Trying next model...

INFO:rag.reasoning_chain:Attempting reasoning with gemini-2.5-flash...
INFO:rag.reasoning_chain:✓ Reasoning generated successfully with gemini-2.5-flash
```

#### Query Router Behavior
- **Single model**: Gemini 3 Flash Preview
- **On rate limit**: Raises exception, query classification fails
- **Fallback**: System uses default classification ("requirement")

#### Cypher Generator Behavior
- **Single model**: Groq llama-3.3-70b-versatile
- **On rate limit**: Uses fallback Cypher template based on query type
- **Templates available for**: cross_reference, metadata queries

### Recovery Actions

#### Automatic
1. **Model fallback** (Reasoning Chain only)
2. **Template fallback** (Cypher Generator)
3. **Exponential backoff** (built into httpx library)

#### Manual
If persistent rate limiting:
1. **Wait**: Rate limits reset after 1 minute
2. **Reduce query frequency**: Space out requests
3. **Upgrade API tier**: Switch to paid plans for higher limits
4. **Configure alternative models**: Update `reasoning_chain.py` and `query_router.py`

### User Experience During Rate Limiting

**Successful Fallback**:
- Slight delay (2-5 seconds) as system tries next model
- Answer generated normally
- User sees successful response

**Failed Fallback** (all models rate-limited):
```
ERROR:rag.answer_generator:Failed to generate answer: All models rate-limited. 
Please wait 60 seconds and try again.
```

---

## 4. Neo4j Connection Errors

### Scenario
System cannot connect to Neo4j database or loses connection during operation.

### Common Causes
1. Neo4j not running (`neo4j.service is not running`)
2. Wrong URI/port (e.g., trying `bolt://` instead of `neo4j://`)
3. Wrong credentials (username/password)
4. Network issues (firewall, localhost restrictions)
5. Neo4j out of memory
6. Database locked by another process

### System Behavior

#### Graph Builder Connection Error
**Occurs during**: `python graph_builder.py`

**Error Message**:
```
ERROR:graph.neo4j_client:Failed to connect to Neo4j
neo4j.exceptions.ServiceUnavailable: Failed to establish connection to ...
```

**System Response**:
- Raises exception immediately
- Graph build fails
- No data written to database

**Recovery**:
```bash
# Check Neo4j status
neo4j status

# Start Neo4j
neo4j start

# Verify connection
neo4j-admin server console
```

#### RAG Pipeline Connection Error
**Occurs during**: Query answering via `AnswerGenerator`

**Components Affected**:
1. **HybridRetriever**: Cannot query vector index or graph
2. **Cypher Generator**: Cannot execute generated queries
3. **AnswerGenerator**: Cannot retrieve related requirements

**Error Handling**:

**Neo4j Client** (`neo4j_client.py`):
```python
def __init__(self):
    try:
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.driver.verify_connectivity()
        logger.info(f"✓ Connected to Neo4j at {NEO4J_URI}")
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {str(e)}")
        raise
```

**Answer Generator** (`answer_generator.py`):
```python
def generate_answer(self, query: str, top_k: int = 5) -> Dict[str, Any]:
    try:
        # ... retrieval steps
    except neo4j.exceptions.ServiceUnavailable as e:
        logger.error(f"Neo4j connection lost: {str(e)}")
        return {
            "answer": "Database connection error. Please ensure Neo4j is running.",
            "confidence": 0.0,
            "error": "ServiceUnavailable"
        }
```

### User Experience

**Clean Error Message**:
```json
{
  "error": "Database connection failed",
  "message": "Unable to connect to Neo4j database. Please ensure Neo4j is running and accessible.",
  "confidence": 0.0,
  "sources": [],
  "query": "What are the requirements for aseptic filling?"
}
```

**Not Returned**:
- No fake/hallucinated answers
- No partial results if critical components fail
- Confidence explicitly set to 0.0

### Recovery Steps

1. **Verify Neo4j is running**:
   ```bash
   neo4j status
   neo4j start
   ```

2. **Check connection details** in `.env`:
   ```env
   NEO4J_URI=neo4j://127.0.0.1:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_password
   ```

3. **Test connection**:
   ```python
   from graph.neo4j_client import Neo4jClient
   client = Neo4jClient()  # Should print "✓ Connected to Neo4j"
   ```

4. **Check Neo4j logs**:
   ```bash
   # Windows
   type %NEO4J_HOME%\logs\neo4j.log
   
   # Linux/Mac
   tail -f /var/log/neo4j/neo4j.log
   ```

### Connection Timeout Settings

**Current Configuration**:
- **Connection timeout**: 30 seconds (default)
- **Maximum transaction retry time**: 30 seconds
- **Keepalive**: Enabled

**To adjust** (if needed):
```python
# In neo4j_client.py
self.driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD),
    connection_timeout=30.0,      # seconds
    max_connection_pool_size=50,
    connection_acquisition_timeout=60.0
)
```

---

## 5. Embedding Model Errors

### Scenario
`sentence-transformers` model fails to load or generate embeddings.

### Common Causes
1. **HuggingFace connection issues** (downloading model)
2. **Disk space** (model cache full)
3. **Memory issues** (model too large for available RAM)
4. **Corrupted model cache**

### System Behavior

#### Model Loading Error
**Occurs during**: System initialization

**Error Message**:
```
ERROR: Failed to load sentence-transformers model: all-MiniLM-L6-v2
OSError: Can't load tokenizer for 'sentence-transformers/all-MiniLM-L6-v2'
```

**System Response**:
- Retriever initialization fails
- RAG pipeline cannot start
- No queries can be processed

#### Embedding Generation Error
**Occurs during**: Query processing

**Error Message**:
```
ERROR:rag.retriever:Failed to generate embedding for query
RuntimeError: CUDA out of memory
```

**System Response**:
- Query falls back to keyword-only search
- Vector similarity search skipped
- Answers may be lower quality but system continues

### Recovery Actions

1. **Clear model cache**:
   ```bash
   rm -rf ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2
   ```

2. **Re-download model**:
   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer('all-MiniLM-L6-v2')
   ```

3. **Use CPU instead of GPU** (if memory issues):
   ```python
   model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
   ```

4. **Free up disk space**: Model requires ~100MB

---

## 6. Malformed Input Handling

### Scenario
User provides malformed, excessively long, or special-character-laden input.

### Examples
- **SQL injection attempts**: `'; DROP TABLE requirements; --`
- **Excessively long queries**: 10,000+ character strings
- **Binary/encoded data**: `\x00\x01\x02...`
- **HTML/script tags**: `<script>alert('xss')</script>`

### System Behavior

#### Query Sanitization
Currently: **Minimal sanitization**
- No explicit input validation
- Relies on parameterized queries (Cypher) to prevent injection
- LLM providers handle escaping

#### Cypher Injection Prevention
**Protected via parameterized queries**:

```python
# SAFE: Parameters passed separately
query = """
MATCH (req:Requirement)
WHERE toLower(req.text) CONTAINS toLower($keyword)
RETURN req
"""
result = session.run(query, {"keyword": user_input})
```

**Never used** (unsafe):
```python
# UNSAFE: String interpolation
query = f"MATCH (req:Requirement) WHERE req.text = '{user_input}'"
```

#### Length Limits
**Current limits**: None enforced
**LLM context windows**:
- Gemini: 32K-128K tokens
- Groq: 32K tokens

**Behavior if exceeded**:
- LLM truncates context automatically
- Answer quality degrades
- No error raised

#### Special Character Handling
- **LLM prompts**: Handle special characters (escaped by API)
- **Neo4j queries**: Escaped by driver automatically
- **JSON serialization**: Handled by `json.dumps(ensure_ascii=False)`

### Recommended Enhancements

**Add input validation**:
```python
def validate_query(query: str) -> str:
    if len(query) > 5000:
        raise ValueError("Query exceeds maximum length of 5000 characters")
    
    # Strip HTML tags
    query = re.sub(r'<[^>]+>', '', query)
    
    # Remove null bytes
    query = query.replace('\x00', '')
    
    return query.strip()
```

---

## 7. Summary of Error Handling Strategies

| Failure Mode | Detection | Fallback Strategy | User Experience |
|--------------|-----------|-------------------|-----------------|
| **No Results** | Zero contexts retrieved | Return low-confidence response with "no information found" message | Clear "not found" message, confidence 0.1-0.2 |
| **Ambiguous Query** | Low similarity scores across multiple topics | Hybrid retrieval + multi-interpretation synthesis | Moderate confidence (0.5-0.7), multiple aspects covered |
| **Rate Limiting** | 429/503 HTTP errors | Model fallback (Reasoning Chain), template fallback (Cypher) | Slight delay, successful answer or "retry" message |
| **Neo4j Disconnection** | ServiceUnavailable exception | Return error response with connection message | Clean error message, confidence 0.0 |
| **Embedding Error** | Model loading/generation failure | Fall back to keyword-only search | Degraded quality, system continues |
| **Malformed Input** | Not explicitly detected | Parameterized queries prevent injection | Processed normally or truncated silently |

---

## 8. Monitoring and Logging

### Log Levels
The system uses Python's `logging` module with these levels:

- **INFO**: Normal operations (successful connections, query steps)
- **WARNING**: Recoverable errors (model fallback, low confidence)
- **ERROR**: Failed operations (connection errors, generation failures)

### Key Log Messages

#### Successful Operations
```
INFO:graph.neo4j_client:✓ Connected to Neo4j at neo4j://127.0.0.1:7687
INFO:rag.retriever:✓ HybridRetriever initialized with all-MiniLM-L6-v2
INFO:rag.reasoning_chain:✓ Reasoning generated successfully with gemini-2.5-flash
INFO:rag.answer_generator:✓ Answer generation complete
```

#### Warnings
```
WARNING:rag.reasoning_chain:Model gemini-3-flash-preview failed: 503 UNAVAILABLE. Trying next model...
WARNING:rag.answer_generator:Low confidence (0.2) - insufficient context for query
WARNING:rag.retriever:No vector results found, using keyword search only
```

#### Errors
```
ERROR:graph.neo4j_client:Failed to connect to Neo4j: ServiceUnavailable
ERROR:rag.answer_generator:Failed to generate answer: Rate limit exceeded on all models
ERROR:rag.cypher_generator:Invalid Cypher query generated
```

### Recommended Monitoring

For production deployment:

1. **Error rate monitoring**: Track ERROR log frequency
2. **Confidence score tracking**: Alert if average drops below 0.6
3. **API usage monitoring**: Track Gemini/Groq request counts
4. **Response time**: Alert if >10 seconds per query
5. **Neo4j health**: Monitor connection pool usage

---

## 9. Testing Failure Modes

### Test Script
To validate error handling, run:

```bash
# Test with no results query
python test_failure_modes.py --test no-results

# Test with ambiguous query
python test_failure_modes.py --test ambiguous

# Test Neo4j disconnection behavior
neo4j stop
python test_failure_modes.py --test connection-error
neo4j start
```

### Expected Test Results

| Test | Expected Confidence | Expected Behavior |
|------|---------------------|-------------------|
| No Results | 0.1-0.2 | "No information found" in answer |
| Ambiguous | 0.5-0.7 | Multiple interpretations, moderate confidence |
| Connection Error | 0.0 | Error message, no fake answer |
| Rate Limit | 0.7-1.0 (after fallback) | Successful answer via backup model |

---

## 10. Conclusion

The GMP Regulatory Intelligence RAG system implements multiple layers of error handling to ensure:

1. **Graceful degradation**: System continues functioning with reduced capability rather than crashing
2. **Honest uncertainty**: Low confidence scores and explicit "not found" messages when information is unavailable
3. **No hallucination**: System never invents regulatory requirements
4. **Automatic recovery**: Fallback strategies for rate limits and model failures
5. **Clear error messages**: User-friendly explanations when issues occur

### Key Principles
- **Fail safely**: Better to return "I don't know" than incorrect regulatory information
- **Transparent confidence**: Confidence scores accurately reflect answer quality
- **Comprehensive logging**: All errors logged for debugging and monitoring
- **Parameterized queries**: Prevent injection attacks

### Future Enhancements
1. Input validation and sanitization
2. Query length limits
3. Automatic retry with exponential backoff
4. Health check endpoints
5. Real-time monitoring dashboard

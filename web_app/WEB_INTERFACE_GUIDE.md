# Web Interface - Quick Reference

## Access URLs
- **Local**: http://localhost:5000
- **Network**: http://192.168.1.70:5000

## Starting the Application

```powershell
pipenv run python web_app/app.py
```

## Complex Test Queries

### 1. Aseptic Filling Requirements (95% confidence)
```
What GMP requirements apply to aseptic filling?
```

### 2. Cross-References Detection (100% confidence)
```
Find cross-references between MHRA and EU GMP Annex 1
```

### 3. Dependency Tracing (70% confidence)
```
What are the dependencies for sterility testing requirements?
```

### 4. Conflict Detection (10% confidence - appropriate, no conflicts)
```
Are there conflicts between EU GMP and ICH guidelines on temperature monitoring?
```

### 5. Temperature Requirements
```
What are the temperature requirements for storage of sterile products?
```

### 6. Cross-Reference Lookup
```
Show me all requirements that reference MHRA Orange Guide
```

### 7. Validation Requirements
```
What are the validation requirements for aseptic processing?
```

### 8. Superseded Requirements
```
Find all requirements related to cleanroom classification that were superseded
```

### 9. Documentation Requirements
```
What documentation is required for contamination control strategy?
```

### 10. Exemptions
```
List all requirements exemptions for investigational medicinal products
```

### 11. Training Requirements
```
What are the personnel training requirements for sterile manufacturing?
```

### 12. Environmental Monitoring
```
What are the environmental monitoring requirements for Grade A cleanrooms?
```

## Features

- **Query Input**: Natural language queries with auto-resize textarea
- **Example Queries**: Click to populate quick test queries
- **Results Configuration**: Adjust top_k (5/7/10/15 results)
- **Confidence Display**: Animated confidence meter with color coding
  - Green (80-100%): High confidence
  - Cyan-Yellow (50-79%): Medium confidence
  - Yellow-Red (0-49%): Low confidence
- **Source Documents**: Expandable cards with hover effects
- **Citations**: Complete source tracking with requirement IDs
- **Responsive Design**: Works on desktop, tablet, and mobile

## API Endpoints

### POST /api/query
Query the RAG system

**Request:**
```json
{
  "query": "What GMP requirements apply to aseptic filling?",
  "top_k": 7
}
```

**Response:**
```json
{
  "success": true,
  "answer": {
    "text": "...",
    "confidence": 0.95,
    "query_type": "requirement",
    "sources": [...],
    "metadata": {...}
  }
}
```

### GET /api/health
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "service": "GMP Regulatory Intelligence",
  "rag_system": "operational"
}
```

## Design Features

- **Typography**: IBM Plex Mono + Outfit (distinctive, professional)
- **Color Palette**: Deep slate with cyan accents (technical precision)
- **Animations**: Smooth transitions, animated confidence meter, staggered source card reveals
- **Background**: Animated gradient particles for depth
- **Interactions**: Hover effects, auto-resize textarea, keyboard shortcuts

## Keyboard Shortcuts

- **Enter**: Submit query
- **Shift+Enter**: New line in query
- **Click example queries**: Auto-populate query input

## Technology Stack

- **Backend**: Flask 3.0+
- **Frontend**: Vanilla JavaScript (no framework dependencies)
- **Styling**: Custom CSS with CSS variables
- **Fonts**: Google Fonts (IBM Plex Mono, Outfit)
- **Icons**: Inline SVG (no icon library needed)

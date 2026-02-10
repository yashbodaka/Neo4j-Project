"""
Rule-Based Conflict Detector
Detects contradictions without relying on LLM JSON parsing.
"""

import logging
from typing import List, Dict, Optional
import re

# Import needed for graph building
try:
    from graph.graph_builder import NodeIDGenerator
except ImportError:
    # Define locally if not available
    import hashlib
    class NodeIDGenerator:
        @staticmethod
        def generate_id(entity_type: str, source_doc: str, text: str) -> str:
            source_doc = source_doc or 'unknown'
            text = text or 'unknown'
            text_hash = hashlib.md5(text.lower().strip().encode()).hexdigest()[:8]
            return f"{entity_type.lower()}_{source_doc.lower().replace(' ', '_')}_{text_hash}"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConflictDetector:
    """Detect conflicts between requirements using rule-based logic."""
    
    # Contradiction keywords
    CONTRADICTIONS = [
        ('shall', 'shall not'),
        ('must', 'must not'),
        ('required', 'not required'),
        ('required', 'optional'),
        ('mandatory', 'optional'),
        ('always', 'never'),
        ('permitted', 'prohibited'),
        ('allowed', 'forbidden'),
        ('mandatory', 'prohibited'),
        ('required', 'forbidden'),
        ('shall', 'may'),
        ('must', 'may')
    ]
    
    # Keywords indicating different standards for same topic
    COMPARISON_KEYWORDS = [
        'minimum', 'maximum', 'at least', 'no more than',
        'exceed', 'below', 'above', 'less than', 'greater than'
    ]
    
    def detect_conflicts(self, contexts: List[Dict]) -> List[Dict]:
        """
        Detect conflicts between retrieved contexts.
        
        Args:
            contexts: List of retrieved requirement contexts (RetrievedContext objects or dicts)
            
        Returns:
            List of detected conflicts
        """
        conflicts = []
        
        for i, ctx1 in enumerate(contexts):
            for ctx2 in contexts[i+1:]:
                conflict = self._are_conflicting(ctx1, ctx2)
                if conflict:
                    # Handle both RetrievedContext objects and dicts
                    source1 = getattr(ctx1, 'source_doc', None) or (ctx1.get('source_doc') if isinstance(ctx1, dict) else 'Unknown')
                    source2 = getattr(ctx2, 'source_doc', None) or (ctx2.get('source_doc') if isinstance(ctx2, dict) else 'Unknown')
                    text1 = getattr(ctx1, 'text', None) or (ctx1.get('text', '') if isinstance(ctx1, dict) else '')
                    text2 = getattr(ctx2, 'text', None) or (ctx2.get('text', '') if isinstance(ctx2, dict) else '')
                    
                    conflicts.append({
                        'source1': source1,
                        'source2': source2,
                        'text1': text1[:200],
                        'text2': text2[:200],
                        'reason': conflict['reason'],
                        'confidence': conflict['confidence']
                    })
        
        logger.info(f"Detected {len(conflicts)} potential conflicts")
        return conflicts
    
    def _are_conflicting(self, ctx1, ctx2) -> Optional[Dict]:
        """Check if two contexts conflict. Returns conflict details or None."""
        # Handle both RetrievedContext objects and dicts
        text1 = (getattr(ctx1, 'text', None) or (ctx1.get('text', '') if isinstance(ctx1, dict) else '')).lower()
        text2 = (getattr(ctx2, 'text', None) or (ctx2.get('text', '') if isinstance(ctx2, dict) else '')).lower()
        metadata1 = getattr(ctx1, 'metadata', None) or (ctx1.get('metadata', {}) if isinstance(ctx1, dict) else {})
        metadata2 = getattr(ctx2, 'metadata', None) or (ctx2.get('metadata', {}) if isinstance(ctx2, dict) else {})
        
        # Don't compare if from same source (likely same regulation)
        source1 = getattr(ctx1, 'source_doc', None) or (ctx1.get('source_doc', '') if isinstance(ctx1, dict) else '')
        source2 = getattr(ctx2, 'source_doc', None) or (ctx2.get('source_doc', '') if isinstance(ctx2, dict) else '')
        
        # Check if texts are similar enough to be about the same topic
        if not self._have_overlapping_keywords(text1, text2, threshold=3):
            return None
        
        # Check 1: Same category but conflicting mandatoriness
        category1 = metadata1.get('category')
        category2 = metadata2.get('category')
        mandatory1 = metadata1.get('mandatory')
        mandatory2 = metadata2.get('mandatory')
        
        if (category1 and category1 == category2 and 
            source1 != source2 and
            mandatory1 is not None and mandatory2 is not None and 
            mandatory1 != mandatory2):
            return {
                'reason': f'Same category ({category1}) but different mandatoriness in {source1} vs {source2}',
                'confidence': 0.75
            }
        
        # Check 2: Contradiction keywords
        contradiction = self._has_contradiction_keywords(text1, text2)
        if contradiction:
            return contradiction
        
        # Check 3: Numeric contradictions (e.g., "< 5" vs "> 10")
        numeric_conflict = self._has_numeric_contradiction(text1, text2)
        if numeric_conflict:
            return numeric_conflict
        
        # Check 4: Different severity for same category across regulations
        severity1 = metadata1.get('severity')
        severity2 = metadata2.get('severity')
        if (category1 and category1 == category2 and
            severity1 and severity2 and severity1 != severity2 and
            source1 != source2):
            # Only flag if severity difference is significant
            severity_levels = ['Low', 'Medium', 'High', 'Critical']
            if severity1 in severity_levels and severity2 in severity_levels:
                idx1 = severity_levels.index(severity1)
                idx2 = severity_levels.index(severity2)
                if abs(idx1 - idx2) >= 2:  # At least 2 levels apart
                    return {
                        'reason': f'Different severity levels for {category1}: {severity1} ({source1}) vs {severity2} ({source2})',
                        'confidence': 0.6
                    }
        
        return None
    
    def _has_contradiction_keywords(self, text1: str, text2: str) -> Optional[Dict]:
        """Check for contradictory keywords. Returns conflict details or None."""
        for pos, neg in self.CONTRADICTIONS:
            # Check both directions
            if (pos in text1 and neg in text2) or (neg in text1 and pos in text2):
                # Extract context around the keywords
                context1 = self._extract_context_around_keyword(text1, pos if pos in text1 else neg)
                context2 = self._extract_context_around_keyword(text2, neg if neg in text2 else pos)
                
                return {
                    'reason': f"Contradictory terms: '{pos}' vs '{neg}' - {context1[:50]}... vs {context2[:50]}...",
                    'confidence': 0.85
                }
        return None
    
    def _extract_context_around_keyword(self, text: str, keyword: str, window: int = 30) -> str:
        """Extract text context around a keyword."""
        idx = text.find(keyword)
        if idx == -1:
            return text[:50]
        start = max(0, idx - window)
        end = min(len(text), idx + len(keyword) + window)
        return text[start:end].strip()
    
    def _have_overlapping_keywords(self, text1: str, text2: str, threshold: int = 3) -> bool:
        """Check if texts share significant keywords."""
        # Extract meaningful words (> 4 chars)
        words1 = set(w for w in text1.split() if len(w) > 4 and w.isalnum())
        words2 = set(w for w in text2.split() if len(w) > 4 and w.isalnum())
        
        overlap = words1 & words2
        return len(overlap) >= threshold
    
    def _has_numeric_contradiction(self, text1: str, text2: str) -> Optional[Dict]:
        """Detect numeric contradictions (e.g., different temperature ranges)."""
        # Extract numbers with comparison operators and units
        pattern = r'([<>≤≥]=?)\s*(\d+(?:\.\d+)?)\s*([°C°FK%]*)'
        matches1 = re.findall(pattern, text1)
        matches2 = re.findall(pattern, text2)
        
        if matches1 and matches2:
            # Check for opposite operators with overlapping values
            for op1, val1, unit1 in matches1:
                for op2, val2, unit2 in matches2:
                    # Same unit or both unitless
                    if unit1 == unit2 or (not unit1 and not unit2):
                        v1 = float(val1)
                        v2 = float(val2)
                        
                        # Conflicting ranges (e.g., "< 5" and "> 10")
                        if '<' in op1 and '>' in op2 and v1 < v2:
                            return {
                                'reason': f'Conflicting numeric ranges: {op1}{val1}{unit1} vs {op2}{val2}{unit2}',
                                'confidence': 0.8
                            }
                        elif '>' in op1 and '<' in op2 and v1 > v2:
                            return {
                                'reason': f'Conflicting numeric ranges: {op1}{val1}{unit1} vs {op2}{val2}{unit2}',
                                'confidence': 0.8
                            }
        
        return None
    
    @staticmethod
    def detect_conflicts_in_entities(entities: List[Dict]) -> List[Dict]:
        """Static method to detect conflicts during graph building."""
        conflicts = []
        requirements = [e for e in entities if e.get('entity_type') == 'Requirement']
        
        logger.info(f"Analyzing {len(requirements)} requirements for conflicts...")
        
        for i, req1 in enumerate(requirements):
            for req2 in requirements[i+1:]:
                conflict = ConflictDetector._check_entity_conflict(req1, req2)
                if conflict:
                    conflicts.append(conflict)
        
        logger.info(f"Found {len(conflicts)} conflicts during graph building")
        return conflicts
    
    @staticmethod
    def _check_entity_conflict(req1: Dict, req2: Dict) -> Optional[Dict]:
        """Check if two requirement entities conflict."""
        text1 = req1.get('text', '').lower()
        text2 = req2.get('text', '').lower()
        props1 = req1.get('properties', {})
        props2 = req2.get('properties', {})
        source1 = req1.get('source_doc', '')
        source2 = req2.get('source_doc', '')
        
        # Skip if same source (same regulation)
        if source1 == source2:
            return None
        
        # Must have overlapping topic
        category1 = props1.get('category', '')
        category2 = props2.get('category', '')
        
        if not category1 or not category2:
            return None
        
        # Same category but conflicting mandatoriness
        if category1 == category2:
            mandatory1 = props1.get('mandatory')
            mandatory2 = props2.get('mandatory')
            
            if mandatory1 is not None and mandatory2 is not None and mandatory1 != mandatory2:
                # Check if they're about similar topics
                words1 = set(w for w in text1.split() if len(w) > 4)
                words2 = set(w for w in text2.split() if len(w) > 4)
                overlap = len(words1 & words2)
                
                if overlap >= 3:
                    return {
                        'req1_id': NodeIDGenerator.generate_id('Requirement', source1, req1['text']),
                        'req2_id': NodeIDGenerator.generate_id('Requirement', source2, req2['text']),
                        'source1': source1,
                        'source2': source2,
                        'category': category1,
                        'reason': f'{category1}: mandatory in {source1} but optional in {source2}',
                        'confidence': 0.7
                    }
        
        return None
    
# Import needed at the top for graph building
try:
    from graph.graph_builder import NodeIDGenerator
except ImportError:
    # Define locally if not available
    import hashlib
    class NodeIDGenerator:
        @staticmethod
        def generate_id(entity_type: str, source_doc: str, text: str) -> str:
            source_doc = source_doc or 'unknown'
            text = text or 'unknown'
            text_hash = hashlib.md5(text.lower().strip().encode()).hexdigest()[:8]
            return f"{entity_type.lower()}_{source_doc.lower().replace(' ', '_')}_{text_hash}"

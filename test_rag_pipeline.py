"""
Test RAG Pipeline
End-to-end test of the complete RAG system.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import os
from dotenv import load_dotenv
from rag.answer_generator import AnswerGenerator

load_dotenv()


def test_rag_pipeline():
    """Test complete RAG pipeline with multiple queries."""
    print("="*80)
    print("RAG PIPELINE END-TO-END TEST")
    print("="*80)
    
    # Initialize generator
    print("\n1. Initializing RAG pipeline...")
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    generator = AnswerGenerator(api_key=gemini_api_key)
    print(f"    Generator initialized with Gemini hierarchy: {generator.reasoner.model_hierarchy}")
    
    # Test queries (Assignment critical tests)
    test_queries = [
        {
            "query": "What GMP requirements apply to aseptic filling?",
            "description": "CRITICAL TEST 1: Aseptic filling requirements",
            "top_k": 7
        },
        {
            "query": "Find cross-references between MHRA and EU GMP Annex 1",
            "description": "CRITICAL TEST 2: Cross-references between regulations",
            "top_k": 8
        },
        {
            "query": "What are the dependencies for sterility testing requirements?",
            "description": "TEST 3: Trace requirement dependencies",
            "top_k": 6
        },
        {
            "query": "Are there conflicts between EU GMP and ICH guidelines on temperature monitoring?",
            "description": "TEST 4: Conflict detection",
            "top_k": 6
        }
    ]
    
    all_results = []
    
    for i, test in enumerate(test_queries, 1):
        print("\n" + "="*80)
        print(f"TEST {i}: {test['description']}")
        print("="*80)
        print(f"Query: {test['query']}")
        
        try:
            print(f"\nGenerating answer...")
            answer = generator.generate_answer(test['query'], top_k=test['top_k'])
            
            print("\n✓ Answer generated successfully!")
            print("\n" + answer.format_for_display())
            
            # Save individual result
            output_path = Path(f'data/processed_docs/test_answer_{i}.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(answer.to_json())
            print(f"\nSaved to: {output_path}")
            
            all_results.append({
                "test_number": i,
                "query": test['query'],
                "success": True,
                "confidence": answer.confidence_score,
                "sources_count": len(answer.sources)
            })
            
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "test_number": i,
                "query": test['query'],
                "success": False,
                "error": str(e)
            })
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    successful = sum(1 for r in all_results if r['success'])
    print(f"Tests Passed: {successful}/{len(test_queries)}")
    for result in all_results:
        status = "✓ PASS" if result['success'] else "✗ FAIL"
        if result['success']:
            print(f"{status} - Test {result['test_number']}: {result['confidence']:.0%} confidence, {result['sources_count']} sources")
        else:
            print(f"{status} - Test {result['test_number']}: {result.get('error', 'Unknown error')}")
    print("="*80)
    
    return successful == len(test_queries)


if __name__ == "__main__":
    success = test_rag_pipeline()
    exit(0 if success else 1)

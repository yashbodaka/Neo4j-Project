"""
Model Benchmark Script
Tests multiple Groq models on same chunks to compare quality and performance.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import json
import jsonlines
import time
from typing import List, Dict
from datetime import datetime
from dotenv import load_dotenv
import os

from extraction.entity_extractor import GroqClient, EntityExtractor

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelBenchmark:
    """Benchmark multiple models on same dataset."""
    
    def __init__(self):
        self.models = [
            {
                'name': 'llama-3.1-8b-instant',
                'display_name': 'Llama 3.1 8B Instant',
                'tier': 'Llama Tier 1',
                'tpd': 500000,
                'tpm': 6000
            },
            {
                'name': 'llama-3.3-70b-versatile',
                'display_name': 'Llama 3.3 70B Versatile',
                'tier': 'Llama Tier 1',
                'tpd': 100000,
                'tpm': 12000
            },
            {
                'name': 'llama-4-scout-17b-16e-instruct',
                'display_name': 'Llama 4 Scout 17B',
                'tier': 'Llama Tier 2',
                'tpd': 500000,
                'tpm': 30000
            },
            {
                'name': 'llama-4-maverick-17b-128e-instruct',
                'display_name': 'Llama 4 Maverick 17B',
                'tier': 'Llama Tier 2',
                'tpd': 500000,
                'tpm': 6000
            },
            {
                'name': 'moonshotai/kimi-k2-instruct',
                'display_name': 'Kimi K2 Instruct',
                'tier': 'Kimi',
                'tpd': 300000,
                'tpm': 10000
            },
            {
                'name': 'moonshotai/kimi-k2-instruct-0905',
                'display_name': 'Kimi K2 0905',
                'tier': 'Kimi',
                'tpd': 300000,
                'tpm': 10000
            }
        ]
        
        self.results = {model['name']: {
            'successful_extractions': 0,
            'failed_extractions': 0,
            'json_parse_failures': 0,
            'total_entities': 0,
            'total_time': 0,
            'avg_time_per_chunk': 0,
            'entities_by_type': {},
            'errors': []
        } for model in self.models}
    
    def load_test_chunks(self, start_idx: int = 665, count: int = 30) -> List[Dict]:
        """Load test chunks from chunks.jsonl."""
        chunks_file = Path('data/processed_docs/chunks.jsonl')
        
        with jsonlines.open(chunks_file) as reader:
            all_chunks = list(reader)
        
        test_chunks = all_chunks[start_idx:start_idx + count]
        logger.info(f"Loaded {len(test_chunks)} test chunks (indices {start_idx}-{start_idx+count-1})")
        
        return test_chunks
    
    def test_model(self, model_config: Dict, chunks: List[Dict]) -> Dict:
        """Test a single model on all chunks."""
        logger.info(f"\n{'='*70}")
        logger.info(f"Testing: {model_config['display_name']}")
        logger.info(f"Model: {model_config['name']}")
        logger.info(f"{'='*70}")
        
        results = self.results[model_config['name']]
        
        # Create client for this model
        try:
            client = GroqClient(model=model_config['name'])
            extractor = EntityExtractor(llm_client=client)
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            results['errors'].append(f"Initialization failed: {str(e)}")
            return results
        
        # Test on each chunk
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"  Chunk {i}/{len(chunks)}: {chunk['chunk_id'][:40]}...")
            
            start_time = time.time()
            
            try:
                entities = extractor.extract_from_chunk(chunk)
                elapsed = time.time() - start_time
                
                results['successful_extractions'] += 1
                results['total_entities'] += len(entities)
                results['total_time'] += elapsed
                
                # Count by type
                for entity in entities:
                    entity_type = entity.entity_type
                    results['entities_by_type'][entity_type] = \
                        results['entities_by_type'].get(entity_type, 0) + 1
                
                logger.info(f"    ✓ {len(entities)} entities in {elapsed:.2f}s")
                
            except json.JSONDecodeError as e:
                elapsed = time.time() - start_time
                results['json_parse_failures'] += 1
                results['failed_extractions'] += 1
                results['total_time'] += elapsed
                logger.warning(f"    ✗ JSON parse error: {str(e)[:100]}")
                results['errors'].append(f"Chunk {i}: JSON parse failed")
                
            except Exception as e:
                elapsed = time.time() - start_time
                results['failed_extractions'] += 1
                results['total_time'] += elapsed
                
                error_msg = str(e)
                if 'rate_limit' in error_msg.lower() or '429' in error_msg:
                    logger.error(f"    ✗ Rate limit hit - stopping this model")
                    results['errors'].append(f"Rate limit hit at chunk {i}")
                    break
                else:
                    logger.error(f"    ✗ Error: {error_msg[:100]}")
                    results['errors'].append(f"Chunk {i}: {error_msg[:100]}")
            
            # Small delay to avoid rate limits
            time.sleep(1)
        
        # Calculate averages
        total_attempts = results['successful_extractions'] + results['failed_extractions']
        if total_attempts > 0:
            results['avg_time_per_chunk'] = results['total_time'] / total_attempts
            results['success_rate'] = (results['successful_extractions'] / total_attempts) * 100
            results['parse_success_rate'] = ((total_attempts - results['json_parse_failures']) / total_attempts) * 100
        
        return results
    
    def run_benchmark(self, start_idx: int = 665, count: int = 30):
        """Run benchmark on all models."""
        logger.info(f"\n{'#'*70}")
        logger.info(f"MODEL BENCHMARK - Testing {len(self.models)} models on {count} chunks")
        logger.info(f"{'#'*70}\n")
        
        # Load test chunks
        chunks = self.load_test_chunks(start_idx, count)
        
        if not chunks:
            logger.error("No chunks loaded for testing!")
            return
        
        # Test each model
        for model_config in self.models:
            try:
                self.test_model(model_config, chunks)
                time.sleep(2)  # Cooldown between models
            except KeyboardInterrupt:
                logger.warning("\nBenchmark interrupted by user")
                break
            except Exception as e:
                logger.error(f"Model {model_config['name']} crashed: {e}")
                self.results[model_config['name']]['errors'].append(f"Crashed: {str(e)}")
        
        # Generate report
        self.generate_report()
        self.save_results()
    
    def generate_report(self):
        """Generate comparison report."""
        print("\n" + "="*90)
        print("BENCHMARK RESULTS - MODEL COMPARISON")
        print("="*90)
        
        # Group by tier
        for tier in ['Llama Tier 1', 'Llama Tier 2', 'Kimi']:
            tier_models = [m for m in self.models if m['tier'] == tier]
            if not tier_models:
                continue
            
            print(f"\n{tier}:")
            print("-" * 90)
            
            for model in tier_models:
                name = model['name']
                results = self.results[name]
                
                total = results['successful_extractions'] + results['failed_extractions']
                
                print(f"\n{model['display_name']}:")
                print(f"  Success Rate:      {results.get('success_rate', 0):.1f}%")
                print(f"  Parse Success:     {results.get('parse_success_rate', 0):.1f}%")
                print(f"  Total Entities:    {results['total_entities']}")
                print(f"  Avg Time/Chunk:    {results['avg_time_per_chunk']:.2f}s")
                print(f"  Successful:        {results['successful_extractions']}/{total}")
                print(f"  JSON Failures:     {results['json_parse_failures']}")
                
                if results['entities_by_type']:
                    print(f"  Entity Breakdown:  ", end="")
                    print(", ".join([f"{k}={v}" for k, v in sorted(results['entities_by_type'].items())]))
                
                if results['errors']:
                    print(f"  Errors:            {len(results['errors'])} - {results['errors'][0] if results['errors'] else 'None'}")
        
        # Recommendations
        print("\n" + "="*90)
        print("RECOMMENDATIONS")
        print("="*90)
        
        # Filter models with good performance
        good_models = []
        for model in self.models:
            name = model['name']
            results = self.results[name]
            
            success_rate = results.get('success_rate', 0)
            parse_rate = results.get('parse_success_rate', 0)
            
            if success_rate >= 80 and parse_rate >= 90:
                good_models.append({
                    'config': model,
                    'success_rate': success_rate,
                    'entities': results['total_entities'],
                    'speed': results['avg_time_per_chunk']
                })
        
        if good_models:
            # Sort by success rate, then by entities extracted
            good_models.sort(key=lambda x: (x['success_rate'], x['entities']), reverse=True)
            
            print(f"\n✓ {len(good_models)} models passed quality threshold (>80% success, >90% parse):\n")
            
            for i, m in enumerate(good_models, 1):
                print(f"{i}. {m['config']['display_name']}")
                print(f"   Success: {m['success_rate']:.1f}% | Entities: {m['entities']} | Speed: {m['speed']:.2f}s/chunk")
            
            print(f"\n→ RECOMMENDED: Use these {len(good_models)} models in rotation system")
            
            # Calculate combined capacity
            total_tpd = sum(m['config']['tpd'] for m in good_models)
            print(f"→ Combined capacity: {total_tpd:,} tokens/day")
        else:
            print("\n⚠ No models passed quality threshold")
            print("→ Review individual results above to adjust thresholds")
        
        print("="*90)
    
    def save_results(self):
        """Save benchmark results to JSON."""
        output_file = Path('data/processed_docs/model_benchmark_results.json')
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_info': {
                'chunk_start_idx': 665,
                'chunk_count': 30,
                'models_tested': len(self.models)
            },
            'models': []
        }
        
        for model in self.models:
            model_report = {
                'name': model['name'],
                'display_name': model['display_name'],
                'tier': model['tier'],
                'quotas': {
                    'tpd': model['tpd'],
                    'tpm': model['tpm']
                },
                'results': self.results[model['name']]
            }
            report['models'].append(model_report)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark multiple LLM models for entity extraction')
    parser.add_argument('--start', type=int, default=665, help='Starting chunk index (default: 665)')
    parser.add_argument('--count', type=int, default=30, help='Number of chunks to test (default: 30)')
    
    args = parser.parse_args()
    
    if not os.getenv("GROQ_API_KEY"):
        logger.error("GROQ_API_KEY not found in environment")
        exit(1)
    
    try:
        benchmark = ModelBenchmark()
        benchmark.run_benchmark(start_idx=args.start, count=args.count)
        
        print("\n✓ Benchmark complete!")
        print("→ Review results above to select models for production extraction")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        exit(1)

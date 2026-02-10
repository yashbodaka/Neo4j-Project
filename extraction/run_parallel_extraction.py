"""
Parallel Multi-Model Entity Extraction
=======================================
Runs 2 threads with separate model pools, each using 2-cycle retry.

Thread 1: Kimi K2 0905 (primary) -> Llama 3.1 8B Instant (fallback)
Thread 2: Kimi K2 Instruct (primary) -> Llama 3.3 70B Versatile (fallback)

Each thread proactively switches from primary to fallback at ~100 chunks
to conserve daily token quotas. Model-specific delays prevent TPM rate limits.

2-Cycle Retry per chunk:
  Cycle 1: primary(@retry 3x) -> fallback(@retry 3x) -> wait 60s
  Cycle 2: primary(@retry 3x) -> fallback(@retry 3x) -> halt

Usage:
    pipenv run python extraction/run_parallel_extraction.py
    pipenv run python extraction/run_parallel_extraction.py --dry-run
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import json
import jsonlines
import time
import random
import threading
from typing import List, Dict, Tuple
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv
import os

from extraction.entity_extractor import GroqClient, EntityExtractor, ExtractedEntity

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(threadName)-18s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# THREAD CONFIGURATIONS
# ============================================================================
THREAD_CONFIGS = [
    {
        'thread_id': 1,
        'thread_name': 'Thread-Kimi0905',
        'primary_model': 'moonshotai/kimi-k2-instruct-0905',
        'fallback_model': 'llama-3.1-8b-instant',
        'primary_delay': 10,    # Aggressive: ~85% TPM utilization (reduced from 15s)
        'fallback_delay': 17,   # Llama 3.1 aggressive (reduced from 26s)
        'primary_quota': 100,   # proactively switch to fallback after N chunks
        'output_file': 'data/processed_docs/entities_thread1.jsonl'
    },
    {
        'thread_id': 2,
        'thread_name': 'Thread-KimiK2',
        'primary_model': 'moonshotai/kimi-k2-instruct',
        'fallback_model': 'llama-3.1-8b-instant',
        'primary_delay': 10,    # Aggressive: ~85% TPM utilization (reduced from 15s)
        'fallback_delay': 17,   # Llama 3.1 aggressive (increased from 9s, matches Thread 1)
        'primary_quota': 100,   # proactively switch to fallback after N chunks
        'output_file': 'data/processed_docs/entities_thread2.jsonl'
    }
]

# Safety constants
MAX_CONSECUTIVE_FAILURES = 3   # Halt thread after N consecutive total failures
SAVE_INTERVAL = 5              # Save checkpoint every N chunks
THREAD_START_STAGGER = 3       # Seconds between starting each thread


# ============================================================================
# CHUNK LOADING & SPLITTING
# ============================================================================
def load_remaining_chunks(
    chunks_file: str = 'data/processed_docs/chunks.jsonl',
    existing_entities_file: str = 'data/processed_docs/entities.jsonl'
) -> List[Dict]:
    """
    Load chunks that haven't been processed yet.
    
    Checks existing entities.jsonl AND thread-specific output files
    to support resuming from partial runs.
    """
    # Load all chunks
    with jsonlines.open(chunks_file) as reader:
        all_chunks = list(reader)
    logger.info(f"Total chunks in dataset: {len(all_chunks)}")
    
    # Collect already-processed chunk IDs from all sources
    processed_ids = set()
    
    # Check main entities file
    existing = Path(existing_entities_file)
    if existing.exists():
        with jsonlines.open(existing) as reader:
            for entity in reader:
                processed_ids.add(entity['source_chunk_id'])
        logger.info(f"Already processed (main file): {len(processed_ids)} unique chunks")
    
    # Check thread-specific output files (for resume support)
    for config in THREAD_CONFIGS:
        thread_file = Path(config['output_file'])
        if thread_file.exists():
            thread_chunk_ids = set()
            with jsonlines.open(thread_file) as reader:
                for entity in reader:
                    chunk_id = entity['source_chunk_id']
                    thread_chunk_ids.add(chunk_id)
                    processed_ids.add(chunk_id)
            if thread_chunk_ids:
                logger.info(f"Already processed ({config['thread_name']}): {len(thread_chunk_ids)} chunks")
    
    # Filter remaining
    remaining = [c for c in all_chunks if c['chunk_id'] not in processed_ids]
    logger.info(f"Remaining chunks to process: {len(remaining)}")
    
    return remaining


def split_chunks(chunks: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Split chunks between 2 threads, balanced for completion time.
    
    Thread 1 gets ~43% — Kimi 0905 (fast) + Llama 3.1 8B (slow fallback)
    Thread 2 gets ~57% — Kimi K2 (fast) + Llama 3.3 70B (fast fallback)
    
    This balances so both threads finish around the same time (~64 min for 354 chunks).
    """
    split_point = int(len(chunks) * 0.43)
    
    # Ensure at least 1 chunk per thread if there are chunks
    if len(chunks) >= 2:
        split_point = max(1, min(split_point, len(chunks) - 1))
    
    thread1_chunks = chunks[:split_point]
    thread2_chunks = chunks[split_point:]
    
    logger.info(f"Thread 1 allocation: {len(thread1_chunks)} chunks")
    logger.info(f"Thread 2 allocation: {len(thread2_chunks)} chunks")
    
    return thread1_chunks, thread2_chunks


# ============================================================================
# FILE I/O
# ============================================================================
def save_entities(entities: List[ExtractedEntity], output_file: str):
    """Save entities to JSONL file (each thread writes to its own file)."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(output_file, mode='w') as writer:
        for entity in entities:
            if isinstance(entity, ExtractedEntity):
                writer.write(entity.to_dict())
            else:
                writer.write(entity)


def merge_results(
    existing_file: str = 'data/processed_docs/entities.jsonl',
    thread_files: List[str] = None,
    output_file: str = 'data/processed_docs/entities.jsonl'
) -> int:
    """
    Merge existing entities with thread outputs into final unified file.
    De-duplicates by (chunk_id, entity_type, text_prefix) key.
    """
    if thread_files is None:
        thread_files = [c['output_file'] for c in THREAD_CONFIGS]
    
    all_entities = []
    seen_keys = set()
    
    def entity_key(entity: Dict) -> tuple:
        """Create dedup key from entity."""
        return (
            entity.get('source_chunk_id', ''),
            entity.get('entity_type', ''),
            entity.get('text', '')[:100]
        )
    
    # Load existing entities first
    existing = Path(existing_file)
    if existing.exists():
        with jsonlines.open(existing) as reader:
            for entity in reader:
                key = entity_key(entity)
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_entities.append(entity)
        logger.info(f"Loaded {len(all_entities)} existing entities from main file")
    
    # Load and merge each thread's output
    for tf in thread_files:
        tf_path = Path(tf)
        if tf_path.exists():
            new_count = 0
            with jsonlines.open(tf_path) as reader:
                for entity in reader:
                    key = entity_key(entity)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        all_entities.append(entity)
                        new_count += 1
            logger.info(f"Merged {new_count} new entities from {tf}")
    
    # Write unified output
    with jsonlines.open(output_file, mode='w') as writer:
        for entity in all_entities:
            writer.write(entity)
    
    logger.info(f"Merged total: {len(all_entities)} entities -> {output_file}")
    return len(all_entities)


# ============================================================================
# THREAD WORKER
# ============================================================================
def worker(
    config: Dict,
    chunks: List[Dict],
    stats_lock: threading.Lock,
    global_stats: Dict,
    halt_event: threading.Event
):
    """
    Worker thread for parallel extraction.
    
    Implements:
    - Proactive model switching at quota threshold (prevents TPD exhaustion)
    - Model-specific delays with jitter (prevents TPM rate limits)
    - Consecutive failure detection -> thread halt (prevents quota waste)
    - Incremental JSONL saving every N chunks
    """
    thread_id = config['thread_id']
    thread_name = config['thread_name']
    output_file = config['output_file']
    
    logger.info(f"[{thread_name}] Starting extraction of {len(chunks)} chunks")
    logger.info(f"[{thread_name}] Primary: {config['primary_model']} (delay: {config['primary_delay']}s)")
    logger.info(f"[{thread_name}] Fallback: {config['fallback_model']} (delay: {config['fallback_delay']}s)")
    logger.info(f"[{thread_name}] Proactive switch at chunk: {config['primary_quota']}")
    
    # Create thread-local extractor with its own API clients
    primary_client = GroqClient(model=config['primary_model'])
    fallback_client = GroqClient(model=config['fallback_model'])
    
    extractor = EntityExtractor(llm_client=primary_client)
    extractor.fallback_llm = fallback_client
    
    # Thread state
    thread_entities = []
    chunks_on_primary = 0
    using_fallback_as_primary = False
    consecutive_failures = 0
    chunks_completed = 0
    
    # Progress bar (each thread gets its own line)
    pbar = tqdm(
        total=len(chunks),
        desc=f"{thread_name:<18}",
        position=thread_id - 1,
        leave=True,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]'
    )
    
    for i, chunk in enumerate(chunks):
        # Check halt signals
        if halt_event.is_set():
            logger.warning(f"[{thread_name}] Halt event detected at chunk {i+1}, stopping.")
            break
        
        # ===========================================
        # PROACTIVE MODEL SWITCHING
        # ===========================================
        if not using_fallback_as_primary and chunks_on_primary >= config['primary_quota']:
            logger.info(
                f"[{thread_name}] === PROACTIVE SWITCH === "
                f"Switching to {config['fallback_model']} "
                f"(primary exhausted {chunks_on_primary} chunks / "
                f"{config['primary_quota']} quota)"
            )
            # Swap roles: fallback becomes primary, primary becomes fallback
            extractor.llm = fallback_client
            extractor.fallback_llm = primary_client
            using_fallback_as_primary = True
        
        # ===========================================
        # EXTRACT (uses 2-cycle retry internally)
        # ===========================================
        entities = extractor.extract_from_chunk(chunk)
        thread_entities.extend(entities)
        chunks_completed += 1
        
        if not using_fallback_as_primary:
            chunks_on_primary += 1
        
        # ===========================================
        # CONSECUTIVE FAILURE DETECTION
        # ===========================================
        is_substantial = len(chunk.get('text', '')) > 200
        if len(entities) == 0 and is_substantial:
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.error(
                    f"[{thread_name}] {MAX_CONSECUTIVE_FAILURES} consecutive failures "
                    f"detected at chunk {i+1}. Halting to prevent quota waste."
                )
                halt_event.set()
                break
        else:
            consecutive_failures = 0
        
        # ===========================================
        # STATS & PROGRESS
        # ===========================================
        with stats_lock:
            global_stats['chunks_processed'] += 1
            global_stats['total_entities'] += len(entities)
            global_stats[f'thread{thread_id}_processed'] += 1
            global_stats[f'thread{thread_id}_entities'] += len(entities)
        
        current_model = config['fallback_model'].split('/')[-1] if using_fallback_as_primary else config['primary_model'].split('/')[-1]
        pbar.update(1)
        pbar.set_postfix({
            'ent': len(thread_entities),
            'model': current_model[:20],
            'fails': consecutive_failures
        })
        
        # ===========================================
        # INCREMENTAL SAVE
        # ===========================================
        if (i + 1) % SAVE_INTERVAL == 0:
            save_entities(thread_entities, output_file)
        
        # ===========================================
        # MODEL-SPECIFIC DELAY WITH JITTER
        # ===========================================
        base_delay = config['fallback_delay'] if using_fallback_as_primary else config['primary_delay']
        jitter = random.uniform(-2, 2)
        actual_delay = max(5, base_delay + jitter)  # Minimum 5s to be safe
        time.sleep(actual_delay)
    
    pbar.close()
    
    # Final save
    save_entities(thread_entities, output_file)
    
    logger.info(
        f"[{thread_name}] Complete. "
        f"Chunks: {chunks_completed}/{len(chunks)} | "
        f"Entities: {len(thread_entities)} | "
        f"Primary chunks used: {chunks_on_primary}"
    )


# ============================================================================
# MAIN
# ============================================================================
def main():
    """Main: parallel dual-thread extraction with model rotation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Parallel multi-model entity extraction')
    parser.add_argument('--dry-run', action='store_true', help='Show plan without executing')
    parser.add_argument('--max-chunks', type=int, default=None, help='Limit total chunks (for testing)')
    args = parser.parse_args()
    
    print("=" * 80)
    print("PARALLEL MULTI-MODEL ENTITY EXTRACTION")
    print("=" * 80)
    print()
    print(f"  Thread 1: {THREAD_CONFIGS[0]['primary_model']}")
    print(f"         -> {THREAD_CONFIGS[0]['fallback_model']} (switch at chunk {THREAD_CONFIGS[0]['primary_quota']})")
    print(f"         Delays: {THREAD_CONFIGS[0]['primary_delay']}s / {THREAD_CONFIGS[0]['fallback_delay']}s")
    print()
    print(f"  Thread 2: {THREAD_CONFIGS[1]['primary_model']}")
    print(f"         -> {THREAD_CONFIGS[1]['fallback_model']} (switch at chunk {THREAD_CONFIGS[1]['primary_quota']})")
    print(f"         Delays: {THREAD_CONFIGS[1]['primary_delay']}s / {THREAD_CONFIGS[1]['fallback_delay']}s")
    print()
    print(f"  Retry:   2-cycle (primary->fallback->60s wait->primary->fallback->halt)")
    print(f"  Safety:  Halt after {MAX_CONSECUTIVE_FAILURES} consecutive failures")
    print(f"  Save:    Every {SAVE_INTERVAL} chunks per thread")
    print("=" * 80)
    
    # Check API key
    if not os.getenv("GROQ_API_KEY"):
        logger.error("GROQ_API_KEY not found. Set it in .env file.")
        sys.exit(1)
    
    # Load remaining chunks
    remaining = load_remaining_chunks()
    
    if not remaining:
        logger.info("All chunks already processed. Nothing to do.")
        return
    
    # Optional chunk limit for testing
    if args.max_chunks:
        remaining = remaining[:args.max_chunks]
        logger.info(f"Limited to {args.max_chunks} chunks (testing mode)")
    
    # Split between threads
    t1_chunks, t2_chunks = split_chunks(remaining)
    
    # Estimate time
    t1_primary_chunks = min(len(t1_chunks), THREAD_CONFIGS[0]['primary_quota'])
    t1_fallback_chunks = max(0, len(t1_chunks) - t1_primary_chunks)
    t1_time = (t1_primary_chunks * THREAD_CONFIGS[0]['primary_delay']) + \
              (t1_fallback_chunks * THREAD_CONFIGS[0]['fallback_delay'])
    
    t2_primary_chunks = min(len(t2_chunks), THREAD_CONFIGS[1]['primary_quota'])
    t2_fallback_chunks = max(0, len(t2_chunks) - t2_primary_chunks)
    t2_time = (t2_primary_chunks * THREAD_CONFIGS[1]['primary_delay']) + \
              (t2_fallback_chunks * THREAD_CONFIGS[1]['fallback_delay'])
    
    est_time = max(t1_time, t2_time)
    print(f"\n  Estimated time: ~{est_time/60:.0f} minutes")
    print(f"  Thread 1: {t1_primary_chunks} primary + {t1_fallback_chunks} fallback = ~{t1_time/60:.0f} min")
    print(f"  Thread 2: {t2_primary_chunks} primary + {t2_fallback_chunks} fallback = ~{t2_time/60:.0f} min")
    
    if args.dry_run:
        print("\n  [DRY RUN] Exiting without processing.\n")
        return
    
    print()
    
    # Shared state
    stats_lock = threading.Lock()
    halt_event = threading.Event()
    global_stats = {
        'chunks_processed': 0,
        'total_entities': 0,
        'thread1_processed': 0,
        'thread1_entities': 0,
        'thread2_processed': 0,
        'thread2_entities': 0,
    }
    
    # Create threads
    threads = [
        threading.Thread(
            target=worker,
            args=(THREAD_CONFIGS[0], t1_chunks, stats_lock, global_stats, halt_event),
            name=THREAD_CONFIGS[0]['thread_name'],
            daemon=True
        ),
        threading.Thread(
            target=worker,
            args=(THREAD_CONFIGS[1], t2_chunks, stats_lock, global_stats, halt_event),
            name=THREAD_CONFIGS[1]['thread_name'],
            daemon=True
        )
    ]
    
    # Start threads with stagger to avoid initial API burst
    start_time = time.time()
    logger.info("Starting parallel extraction...\n")
    
    for t in threads:
        t.start()
        time.sleep(THREAD_START_STAGGER)
    
    # Wait for completion (or keyboard interrupt)
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user. Setting halt event...")
        halt_event.set()
        # Give threads time to save checkpoints
        for t in threads:
            t.join(timeout=30)
    
    elapsed = time.time() - start_time
    
    # ==========================================
    # SUMMARY
    # ==========================================
    print("\n\n" + "=" * 80)
    print("EXTRACTION SUMMARY")
    print("=" * 80)
    print(f"  Total time:          {elapsed/60:.1f} minutes ({elapsed:.0f}s)")
    print(f"  Chunks processed:    {global_stats['chunks_processed']}/{len(remaining)}")
    print(f"  Total new entities:  {global_stats['total_entities']}")
    print(f"  Thread 1:            {global_stats['thread1_processed']} chunks -> {global_stats['thread1_entities']} entities")
    print(f"  Thread 2:            {global_stats['thread2_processed']} chunks -> {global_stats['thread2_entities']} entities")
    
    if global_stats['chunks_processed'] > 0:
        avg_speed = elapsed / global_stats['chunks_processed']
        print(f"  Avg speed:           {avg_speed:.1f}s per chunk (wall-clock)")
    
    if halt_event.is_set():
        print(f"\n  WARNING: Extraction halted early (consecutive failures or interrupt)")
        print(f"  -> Re-run this script to resume from where it stopped")
    
    # ==========================================
    # MERGE
    # ==========================================
    print("\nMerging results...")
    total = merge_results()
    
    print(f"\n  Final entity count: {total}")
    print(f"  Output: data/processed_docs/entities.jsonl")
    print("=" * 80)


if __name__ == "__main__":
    main()

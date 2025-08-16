#!/usr/bin/env python3
"""Test script to verify Graphwalks token filtering functionality."""

import sys
import warnings
# Suppress numpy warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


def test_token_counting():
    """Test that token counting works for sample texts."""
    from openbench.utils.text import get_token_count
    
    test_prompts = [
        "Short prompt for testing.",
        "This is a medium length prompt that contains more text to test the token counting functionality properly.",
        "A" * 1000,  # Long repetitive text
    ]
    
    print("Testing token counting:")
    for i, prompt in enumerate(test_prompts, 1):
        token_count = get_token_count(prompt)
        print(f"  Prompt {i}: {len(prompt)} chars -> {token_count} tokens")
    print()


def test_dataset_filtering():
    """Test that dataset filtering works with max_context_size."""
    from openbench.datasets.graphwalks import get_dataset
    
    print("Testing dataset filtering:")
    
    # Test without filtering
    dataset_all = get_dataset(split="train")
    print(f"  Dataset without filtering: loading...")
    
    # Test with filtering at 2000 tokens
    dataset_filtered = get_dataset(split="train", max_context_size=2000)
    print(f"  Dataset with max_context_size=2000: loading...")
    
    # Note: We can't easily count samples without loading the entire dataset
    # which might be large. In production, you'd want to sample a few items.
    print("  Dataset objects created successfully")
    print()


def test_record_mapper():
    """Test the record_to_sample mapper function."""
    from openbench.datasets.graphwalks import record_to_sample
    
    print("Testing record_to_sample mapper:")
    
    # Create test records
    test_records = [
        {
            "prompt": "Find the shortest path from A to B.",
            "answer_nodes": ["A", "C", "B"],
            "problem_type": "bfs",
            "prompt_chars": 35,
        },
        {
            "prompt": "A" * 5000,  # Very long prompt
            "answer_nodes": ["X", "Y", "Z"],
            "problem_type": "parents",
            "prompt_chars": 5000,
        },
    ]
    
    # Test without filtering
    mapper_no_filter = record_to_sample(max_context_size=None)
    print("  Without filtering:")
    for i, record in enumerate(test_records, 1):
        result = mapper_no_filter(record)
        if isinstance(result, list) and len(result) == 0:
            print(f"    Record {i}: Filtered out")
        else:
            print(f"    Record {i}: Included (token count: {result.metadata.get('raw_input_tok_cnt', 'N/A')})")
    
    # Test with filtering at 100 tokens
    mapper_with_filter = record_to_sample(max_context_size=100)
    print("  With max_context_size=100:")
    for i, record in enumerate(test_records, 1):
        result = mapper_with_filter(record)
        if isinstance(result, list) and len(result) == 0:
            print(f"    Record {i}: Filtered out")
        else:
            print(f"    Record {i}: Included (token count: {result.metadata.get('raw_input_tok_cnt', 'N/A')})")
    print()


def test_task_creation():
    """Test that tasks can be created with max_context_size parameter."""
    from openbench.evals.graphwalks import graphwalks, graphwalks_bfs
    
    print("Testing task creation:")
    
    # Create tasks with different configurations
    task1 = graphwalks(split="train")
    print("  Created graphwalks task without filtering")
    
    task2 = graphwalks(split="train", max_context_size=4096)
    print("  Created graphwalks task with max_context_size=4096")
    
    task3 = graphwalks_bfs(split="train", max_context_size=2048)
    print("  Created graphwalks_bfs task with max_context_size=2048")
    
    print("  All tasks created successfully")
    print()


def main():
    """Run all tests."""
    print("=" * 60)
    print("Graphwalks Token Filtering Test Suite")
    print("=" * 60)
    print()
    
    test_token_counting()
    test_record_mapper()
    test_dataset_filtering()
    test_task_creation()
    
    print("=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
    print()
    print("Usage examples:")
    print("  # Run evaluation with token filtering:")
    print("  openbench eval graphwalks -T max_context_size=4096")
    print()
    print("  # Or in Python:")
    print("  from openbench.evals.graphwalks import graphwalks")
    print("  task = graphwalks(split='train', max_context_size=4096)")


if __name__ == "__main__":
    main()

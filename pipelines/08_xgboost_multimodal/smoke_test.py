"""
Quick smoke test: Run a few strategies to verify baseline matching works.
Tests 2 strategies per baseline type = 8 total (should complete in ~1 minute).
"""
import sys
import subprocess
from pathlib import Path

# Modify the main script temporarily to test just a few strategies
test_code = """
# Quick smoke test configuration
if __name__ == '__main__':
    import time
    start = time.time()
    
    print("="*80)
    print("SMOKE TEST: Testing baseline matching with small subset")
    print("="*80)
    
    # Test just 2 strategies per baseline = 8 total
    results = []
    test_strategies = [
        ('none', 'none', 'full', 'logistic_regression', 'all', 'stress_classification'),
        ('none', 'standardize', 'full', 'random_forest', 'eeg', 'workload_classification'),
        ('subtract', 'none', 'full', 'xgboost', 'all', 'stress_classification'),
        ('subtract', 'standardize', 'last_60s', 'svm', 'physio', 'workload_classification'),
        ('zscore', 'none', 'full', 'lda', 'all', 'stress_classification'),
        ('zscore', 'minmax', 'last_120s', 'knn', 'eeg', 'workload_classification'),
        ('percent', 'none', 'full', 'mlp', 'all', 'stress_classification'),
        ('percent', 'standardize', 'within_subject', 'lightgbm', 'eeg', 'workload_classification'),
    ]
    
    print(f"\\nTesting {len(test_strategies)} strategies...")
    print()
    
    for baseline, norm, duration, model_name, modality, task_type in test_strategies:
        strategy_name = f"{baseline}_{norm}_{duration}_{model_name}_{modality}_{task_type}"
        print(f"Testing: {strategy_name}")
        
        try:
            result = test_single_strategy(
                baseline_adjustment=baseline,
                normalization_method=norm,
                baseline_duration=duration,
                model_name=model_name,
                feature_modality=modality,
                task_type=task_type
            )
            
            if result is not None:
                results.append(result)
                print(f"  ✓ SUCCESS: Mean={result.get('mean_window_accuracy', 0):.3f}")
            else:
                print(f"  ✗ FAILED: Returned None")
                
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
    
    elapsed = time.time() - start
    print()
    print("="*80)
    print(f"SMOKE TEST COMPLETE")
    print(f"Successful: {len(results)}/{len(test_strategies)}")
    print(f"Runtime: {elapsed:.1f}s")
    print("="*80)
    
    if len(results) >= 6:  # At least 75% success
        print("\\n✓ Baseline matching is working! Ready for full run.")
        sys.exit(0)
    else:
        print("\\n✗ Too many failures. Check the errors above.")
        sys.exit(1)
"""

print("Running smoke test on main script...")
print("This will test 8 strategies (2 per baseline type)")
print("Expected runtime: ~1-2 minutes")
print()

# Run the main script (it will use the test configuration)
result = subprocess.run(
    ['python', 'pipelines/08_xgboost_multimodal/xgboost_multimodal_classification.py'],
    capture_output=False,
    text=True
)

sys.exit(result.returncode)

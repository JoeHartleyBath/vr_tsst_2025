"""Quick smoke test - run one strategy with baseline adjustment."""
import subprocess
import sys

print("="*70)
print("SMOKE TEST: Testing baseline adjustment with one strategy")
print("="*70)
print("\nThis will test: subtract_within_subject_full_xgboost_all_stress")
print("Expected: Should complete without 'No objects to concatenate' error\n")

# Run with --quick flag and capture output
result = subprocess.run(
    [sys.executable, 'pipelines/08_xgboost_multimodal/xgboost_multimodal_classification.py', '--quick'],
    capture_output=True,
    text=True
)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

if "No objects to concatenate" in result.stdout or "No objects to concatenate" in result.stderr:
    print("\n" + "="*70)
    print("❌ FAILED: Still getting 'No objects to concatenate' error")
    print("="*70)
    sys.exit(1)
elif result.returncode == 0:
    print("\n" + "="*70)
    print("✓ SUCCESS: Baseline adjustment working!")
    print("="*70)
    sys.exit(0)
else:
    print("\n" + "="*70)
    print(f"⚠ Script exited with code {result.returncode}")
    print("="*70)
    sys.exit(result.returncode)

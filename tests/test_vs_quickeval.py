"""
Test that our refactored benchmark produces identical results to QuickEval.

This test runs both our code and QuickEval code, then compares the results.

Usage:
    python tests/test_vs_quickeval.py --target GPP --agg daily
    python tests/test_vs_quickeval.py --target Qle --agg daily-2017
"""

import argparse
import subprocess
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path


def run_command(cmd, description, cwd=None):
    """Run a shell command and check for errors."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*70}")

    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        cwd=cwd or os.getcwd()
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode != 0:
        print(f"\n✗ ERROR: Command failed with return code {result.returncode}")
        return False

    print(f"✓ Command completed successfully")
    return True


def compare_results(our_file, quickeval_file, tolerance=0.01):
    """
    Compare our results against QuickEval results.

    Args:
        our_file: path to our results CSV
        quickeval_file: path to QuickEval results CSV
        tolerance: relative tolerance for comparison (default: 1%)

    Returns:
        bool: True if results match within tolerance
    """
    print(f"\n{'='*70}")
    print("COMPARING RESULTS")
    print(f"{'='*70}")

    # Load results
    if not os.path.exists(our_file):
        print(f"✗ ERROR: Our results file not found: {our_file}")
        return False

    if not os.path.exists(quickeval_file):
        print(f"✗ ERROR: QuickEval results file not found: {quickeval_file}")
        return False

    our_results = pd.read_csv(our_file)
    qe_results = pd.read_csv(quickeval_file)

    qe_envs = set(qe_results['env'].unique())
    # our_results1 = our_results[our_results['group'] == our_results['group'][0]]
    our_results = our_results[our_results['env'].isin(qe_envs)]

    # print(f"\nOur results:       {len(our_results)} groups")
    # print(f"QuickEval results: {len(qe_results)} groups")

    # Metrics to compare
    metrics = ['rmse', 'mse', 'nse']

    all_match = True
    results_table = []

    for metric in metrics:
        if metric not in our_results.columns:
            print(f"\n⚠ WARNING: Metric '{metric}' not found in our results")
            continue
        if metric not in qe_results.columns:
            print(f"\n⚠ WARNING: Metric '{metric}' not found in QuickEval results")
            continue

        # Calculate summary statistics
        our_mean = our_results[metric].mean()
        qe_mean = qe_results[metric].mean()

        our_median = our_results[metric].median()
        qe_median = qe_results[metric].median()

        our_max = our_results[metric].max()
        qe_max = qe_results[metric].max()

        # Calculate relative differences
        def rel_diff(a, b):
            return abs(a - b) / abs(b) if b != 0 else abs(a - b)

        mean_diff = rel_diff(our_mean, qe_mean)
        median_diff = rel_diff(our_median, qe_median)
        max_diff = rel_diff(our_max, qe_max)

        # Check if within tolerance
        mean_match = mean_diff <= tolerance
        median_match = median_diff <= tolerance
        max_match = max_diff <= tolerance

        # Store results
        results_table.append({
            'metric': metric.upper(),
            'statistic': 'mean',
            'ours': our_mean,
            'quickeval': qe_mean,
            'diff_%': mean_diff * 100,
            'match': '✓' if mean_match else '✗'
        })
        results_table.append({
            'metric': '',
            'statistic': 'median',
            'ours': our_median,
            'quickeval': qe_median,
            'diff_%': median_diff * 100,
            'match': '✓' if median_match else '✗'
        })
        results_table.append({
            'metric': '',
            'statistic': 'max',
            'ours': our_max,
            'quickeval': qe_max,
            'diff_%': max_diff * 100,
            'match': '✓' if max_match else '✗'
        })

        if not (mean_match and median_match and max_match):
            all_match = False

    # Print results table
    print(f"\n{'Metric':<8} {'Stat':<8} {'Ours':>12} {'QuickEval':>12} {'Diff %':>10} {'Match':>6}")
    print("=" * 70)
    for row in results_table:
        print(f"{row['metric']:<8} {row['statistic']:<8} {row['ours']:>12.4f} "
              f"{row['quickeval']:>12.4f} {row['diff_%']:>10.3f} {row['match']:>6}")

    print(f"\n{'='*70}")
    if all_match:
        print("✓ ALL TESTS PASSED - Results match within tolerance!")
        print(f"  (Tolerance: {tolerance*100:.1f}%)")
    else:
        print("✗ SOME TESTS FAILED - Results differ beyond tolerance")
        print(f"  (Tolerance: {tolerance*100:.1f}%)")
    print(f"{'='*70}")

    return all_match


def main():
    parser = argparse.ArgumentParser(
        description="Test our implementation against QuickEval"
    )
    parser.add_argument("--target", type=str, choices=['GPP', 'Qle'],
                        default='GPP', help="Target variable (GPP or Qle)")
    parser.add_argument("--agg", type=str,
                        choices=['daily', 'daily-2017', 'daily-100-2017'],
                        default='daily', help="Data aggregation level")
    parser.add_argument("--tolerance", type=float, default=0.01,
                        help="Relative tolerance for comparison (default: 1%%)")
    parser.add_argument("--override", action='store_true',
                        help="Override existing results of our implementation")

    args = parser.parse_args()

    # Get root directory
    root_dir = Path(__file__).parent.parent
    tests_dir = Path(__file__).parent

    print("\n" + "="*70)
    print("FLUXNET BENCHMARK: Test vs QuickEval")
    print("="*70)
    print(f"Target:    {args.target}")
    print(f"Data:      {args.agg}")
    print(f"Tolerance: {args.tolerance*100:.1f}%")
    print("="*70)

    # Step 1: Run our implementation
    exp_name = f"test_{args.target}_{args.agg}_lr"
    our_results_file = root_dir / f"results/{exp_name}.csv"
    
    cmd = (
        f"python run_experiment.py "
        f"--target {args.target} "
        f"--agg {args.agg} "
        f"--setting spatial-easy "
        f"--model_name lr "
        f"--experiment_name {exp_name} "
        f"{"--override" if args.override else ""}"
    )

    success = run_command(cmd, "Our Implementation", cwd=str(root_dir))
    if not success:
        print("\n✗ Failed to run our implementation")
        sys.exit(1)

    # Step 2: Run QuickEval
    print(f"\n{'='*70}")
    print("Running QuickEval")
    print(f"{'='*70}")

    target = 'LE' if args.target == 'Qle' else args.target
    cmd = (
        f"python quick_comp.py "
        f"--target {target} "
        "--our-setting "
        f"{"--subset2017" if args.agg != "daily" else ""} "
        f"{"--subset100" if args.agg == 'daily-100-2017' else ""} "
    )
    success = run_command(cmd, "QuickEval", cwd=str(tests_dir / "QuickEval"))

    if not success:
        print("\n✗ Failed to run QuickEval")
        sys.exit(1)

    quickeval_results_file = tests_dir / f"QuickEval/results/quickeval_{args.agg}_{target}_lr.csv"
    
    # Step 3: Compare results
    passed = compare_results(
        str(our_results_file),
        str(quickeval_results_file),
        args.tolerance
    )

    # Print file locations
    print(f"\nOur results:       {our_results_file}")
    print(f"QuickEval results: {quickeval_results_file}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

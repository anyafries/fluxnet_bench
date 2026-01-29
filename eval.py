"""
Script to load and compare results from multiple experiments.

This script loads pre-computed metrics if available, or computes them from
predictions. Results are saved to results/metrics/ for subsequent runs.

Usage:
    python eval.py --setting spatial-easy --target GPP
"""

import pandas as pd

from dataloader import load_predictions
from utils.eval_utils import load_metrics, compute_and_save_metrics
from utils.plots import plot_metric_grid, plot_cdf_grid, create_leaderboard
from utils.utils import setup_logging, find_available_experiments

logger = setup_logging(__name__)

def get_metrics(setting, target, model_name, override=False):
    """Get metrics for an experiment, computing if necessary."""
    # Try loading existing metrics
    if not override:
        metrics_df = load_metrics(setting, target, model_name)
        if metrics_df is not None:
            return metrics_df

    # Load predictions
    predictions_df = load_predictions(setting, target, model_name)
    if predictions_df is None:
        return None

    # Compute and save metrics
    metrics_df = compute_and_save_metrics(predictions_df, setting, target, model_name)
    
    return metrics_df


def load_all_metrics(settings=None, targets=None, models=None, scales=None, override=False):
    """
    Load or compute metrics for all specified experiments.

    Args:
        settings: List of settings to include (default: all found)
        targets: List of targets to include (default: all found)
        models: List of models to include (default: all found)
        scales: List of scales to include (default: all scales in data)
        override: If True, recompute all metrics from predictions

    Returns:
        pd.DataFrame with all metrics combined
    """
    available = find_available_experiments()
    if not available:
        logger.error("No experiments found in results/")
        return pd.DataFrame()

    # Filter experiments
    if settings:
        available = [(s, t, m) for s, t, m in available if s in settings]
    if targets:
        available = [(s, t, m) for s, t, m in available if t in targets]
    if models:
        available = [(s, t, m) for s, t, m in available if m in models]

    logger.info(f"Processing {len(available)} experiments")

    all_results = []
    for setting, target, model_name in available:
        metrics_df = get_metrics(setting, target, model_name, override=override)
        if metrics_df is not None:
            if scales and 'scale' in metrics_df.columns:
                metrics_df = metrics_df[metrics_df['scale'].isin(scales)]
            all_results.append(metrics_df)

    if all_results:
        return pd.concat(all_results, ignore_index=True)

    return pd.DataFrame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load and compare results from FLUXNET experiments"
    )
    parser.add_argument("--setting", type=str, default=None,
                        help="Filter by setting (e.g., 'spatial-easy')")
    parser.add_argument("--target", type=str, default=None,
                        help="Filter by target (e.g., 'GPP')")
    parser.add_argument("--model", type=str, default=None,
                        help="Filter by model (e.g., 'lr')")
    parser.add_argument("--scale", type=str, default=None,
                        help="Filter by scale (e.g., 'daily', 'weekly', 'monthly')")
    parser.add_argument("--override", action='store_true',
                        help="Recompute metrics from predictions")

    args = parser.parse_args()

    # Parse filters
    settings = [args.setting] if args.setting else None
    targets = [args.target] if args.target else None
    models = [args.model] if args.model else None
    scales = [args.scale] if args.scale else None

    # Load results
    results = load_all_metrics(
        settings=settings,
        targets=targets,
        models=models,
        scales=scales,
        override=args.override,
    )

    print(results.head())
    print(results.tail())

    # Generate plots for all targets
    for target in results['target'].unique():
        plot_metric_grid(results, target)
        plot_cdf_grid(results, target, scale='daily', metric='rmse')
        create_leaderboard(results, target, metric='rmse', 
                           filename='results/plots/medals.html')

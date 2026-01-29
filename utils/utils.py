"""Utility functions for the FLUXNET benchmark."""

import logging
import os
import pandas as pd

RESULTS_DIR = 'results'

def setup_logging(name=None):
    """
    Set up logging configuration and return a logger.

    Args:
        name: Logger name (typically __name__ from the calling module).
            If None, returns the root logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(name)


logger = setup_logging(__name__)


# ----------- Path helper functions -----------

def get_exp_name(setting, target, model_name):
    """Build canonical experiment name."""
    return f"{setting}_{target}_{model_name}"


def get_predictions_path(setting, target, model_name):
    """Get path for predictions file."""
    exp_name = get_exp_name(setting, target, model_name)
    return os.path.join(RESULTS_DIR, 'models', f"{exp_name}_predictions.csv")


def get_metrics_path(setting, target, model_name):
    """Get path for metrics file."""
    exp_name = get_exp_name(setting, target, model_name)
    return os.path.join(RESULTS_DIR, 'metrics', f"{exp_name}.csv")


# ----------- Generic I/O functions -----------

def save_csv(df, path):
    """Generic CSV save with directory creation and logging."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved: {path}")


def load_csv(path):
    """Generic CSV load with logging. Returns None if file doesn't exist."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    logger.info(f"Loaded: {path} ({len(df)} rows)")
    return df


# ----------- Experiment discovery -----------

def find_available_experiments(results_dir=None):
    """
    Find all available prediction files.

    Returns:
        list of tuples: (setting, target, model_name)
    """
    if results_dir is None:
        results_dir = os.path.join(RESULTS_DIR, 'models')

    if not os.path.exists(results_dir):
        return []

    experiments = []
    for filename in os.listdir(results_dir):
        if filename.endswith('_predictions.csv'):
            # Parse filename: {setting}_{target}_{model}_predictions.csv
            parts = filename.replace('_predictions.csv', '').rsplit('_', 2)
            if len(parts) == 3:
                setting, target, model_name = parts
                experiments.append((setting, target, model_name))
    return experiments


#!/usr/bin/env python3
"""
Train EEGNet on the full dataset (train + validation combined) for final submission.

This script uses a fixed number of epochs determined from prior experiments,
since there's no held-out validation set for early stopping.

Usage:
    python examples/train_eegnet_full.py
    python examples/train_eegnet_full.py --epochs 50
"""

import argparse
from pathlib import Path

import pandas as pd
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from brainstorm.download import download_train_validation_data
from brainstorm.loading import load_raw_data
from brainstorm.ml.eegnet import EEGNet


# =============================================================================
# Default Configuration (Best model from prior experiments)
# =============================================================================

DATA_PATH = Path("./data")

# Best EEGNet configuration from validation experiments
BEST_CONFIG = {
    "projected_channels": 32,
    "window_size": 1600,
    "F1": 8,
    "D": 2,
    "dropout": 0.25,
}

# Training parameters
DEFAULT_EPOCHS = 45  # Fixed epochs (best epoch from val experiments + buffer)
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 1e-3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train EEGNet on full dataset (train + validation) for final submission"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})",
    )
    return parser.parse_args()


def load_combined_data(data_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and combine train + validation data for final training.

    Args:
        data_path: Path to the data directory.

    Returns:
        Tuple of (combined_features, combined_labels) DataFrames.
    """
    rprint("[bold cyan]Loading training data...[/]")
    train_features, train_labels = load_raw_data(data_path, step="train")
    rprint(f"  Train samples: {len(train_features)}")

    rprint("[bold cyan]Loading validation data...[/]")
    val_features, val_labels = load_raw_data(data_path, step="validation")
    rprint(f"  Validation samples: {len(val_features)}")

    # Offset validation time indices to avoid collision with train indices
    time_offset = train_features.index[-1] + 0.001
    val_features = val_features.copy()
    val_labels = val_labels.copy()
    val_features.index = val_features.index + time_offset
    val_labels.index = val_labels.index + time_offset

    # Concatenate train and validation data
    combined_features = pd.concat([train_features, val_features], axis=0)
    combined_labels = pd.concat([train_labels, val_labels], axis=0)

    rprint(f"[bold green]Combined samples: {len(combined_features)}[/]")

    return combined_features, combined_labels


def main() -> None:
    args = parse_args()

    rprint("\n[bold cyan]=" * 60)
    rprint("[bold cyan]EEGNet Full Dataset Training (Train + Validation Combined)[/]")
    rprint("[bold cyan]=" * 60 + "\n")

    # Download data if needed
    if not DATA_PATH.exists() or not any(DATA_PATH.glob("*.parquet")):
        rprint("[bold yellow]Downloading data from Hugging Face...[/]\n")
        download_train_validation_data()
        rprint("[bold green]Data downloaded.[/]\n")

    # Load and combine data
    combined_features, combined_labels = load_combined_data(DATA_PATH)

    # Display dataset info
    console = Console()
    table = Table(
        title="Combined Dataset Overview",
        show_header=True,
        header_style="bold magenta",
    )

    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Value", style="green")

    table.add_row("Total Samples", str(len(combined_features)))
    table.add_row("Channels", str(combined_features.shape[1]))
    table.add_row(
        "Time Range (s)",
        f"{combined_features.index[0]:.2f} -> {combined_features.index[-1]:.2f}",
    )
    table.add_row(
        "Unique Labels", str(sorted(combined_labels["label"].unique().tolist()))
    )

    console.print(table)
    print()

    # Display model configuration
    rprint("[bold green]Model Configuration (Best from validation experiments):[/]")
    for key, value in BEST_CONFIG.items():
        rprint(f"  {key}: {value}")
    print()

    rprint("[bold green]Training Configuration:[/]")
    rprint(f"  Epochs: {args.epochs}")
    rprint(f"  Batch size: {args.batch_size}")
    rprint(f"  Learning rate: {args.learning_rate}")
    rprint("[bold yellow]  Note: No validation data - using fixed epochs[/]")
    print()

    # Create model with best configuration
    model = EEGNet(
        input_size=combined_features.shape[1],
        projected_channels=BEST_CONFIG["projected_channels"],
        window_size=BEST_CONFIG["window_size"],
        F1=BEST_CONFIG["F1"],
        D=BEST_CONFIG["D"],
        dropout=BEST_CONFIG["dropout"],
    )

    # Train on full dataset (no validation data = no early stopping)
    rprint("[bold green]Starting training on full dataset...[/]\n")

    model.fit(
        X=combined_features.values,
        y=combined_labels["label"].values,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        verbose=True,
        # NO validation data - train on all samples with fixed epochs
    )

    rprint("\n[bold green]=" * 60)
    rprint("[bold green]Training complete! Model saved to model.pt[/]")
    rprint("[bold green]=" * 60 + "\n")

    # Display model file info
    model_path = Path("model.pt")
    if model_path.exists():
        size_kb = model_path.stat().st_size / 1024
        rprint(f"[bold cyan]Model size: {size_kb:.1f} KB[/]")


if __name__ == "__main__":
    main()

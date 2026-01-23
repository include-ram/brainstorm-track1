#!/usr/bin/env python3
"""
Train and evaluate the EEGNet model for continuous classification.

This script demonstrates training the EEGNet-based model with:
    1. Optional PCA channel projection (1024 -> 64 channels)
    2. Temporal windowing for context
    3. Compact convolutional architecture

Usage:
    python examples/train_eegnet.py

    # Train without PCA (use all 1024 channels)
    python examples/train_eegnet.py --no-pca

The trained model and metadata are saved to the repository root.
"""

import argparse
from pathlib import Path
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from brainstorm.download import download_train_validation_data
from brainstorm.loading import load_raw_data
from brainstorm.evaluation import ModelEvaluator
from brainstorm.ml.eegnet import EEGNet


# =============================================================================
# Configuration
# =============================================================================

DATA_PATH = Path("./data")

# EEGNet parameters
PROJECTED_CHANNELS = 64  # Number of channels after PCA
WINDOW_SIZE = 128  # Temporal context window (128ms at 1000Hz)
F1 = 8  # Number of temporal filters
D = 2  # Depthwise multiplier
DROPOUT = 0.25

# Training parameters
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EEGNet for ECoG signal classification"
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS})"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--window-size", type=int, default=WINDOW_SIZE,
        help=f"Window size in samples (default: {WINDOW_SIZE})"
    )
    parser.add_argument(
        "--projected-channels", type=int, default=PROJECTED_CHANNELS,
        help=f"Number of PCA channels (default: {PROJECTED_CHANNELS})"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE})"
    )
    parser.add_argument(
        "--no-pca", action="store_true",
        help="Disable PCA and use all 1024 channels directly"
    )
    parser.add_argument(
        "--F1", type=int, default=F1,
        help=f"Number of temporal filters (default: {F1})"
    )
    parser.add_argument(
        "--D", type=int, default=D,
        help=f"Depthwise multiplier (default: {D})"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_pca = not args.no_pca

    rprint("\n[bold cyan]EEGNet Training Script[/]\n")

    # Download data if needed
    if not DATA_PATH.exists() or not any(DATA_PATH.glob("*.parquet")):
        rprint("[bold yellow]Downloading data from Hugging Face...[/]\n")
        download_train_validation_data()
        rprint("[bold green]Data downloaded.[/]\n")

    rprint(f"[bold cyan]Loading data from:[/] {DATA_PATH}\n")
    train_features, train_labels = load_raw_data(DATA_PATH, step="train")
    validation_features, validation_labels = load_raw_data(DATA_PATH, step="validation")

    # Display dataset info
    console = Console()
    table = Table(
        title="Dataset Overview", show_header=True, header_style="bold magenta"
    )

    table.add_column("Split", style="cyan", width=10)
    table.add_column("Features Shape", style="green")
    table.add_column("Labels Shape", style="green")
    table.add_column("Time Range (s)", style="yellow")
    table.add_column("Unique Labels", style="blue")

    table.add_row(
        "Train",
        str(train_features.shape),
        str(train_labels.shape),
        f"{train_features.index[0]:.2f} -> {train_features.index[-1]:.2f}",
        str(sorted(train_labels["label"].unique().tolist())),
    )

    table.add_row(
        "Validation",
        str(validation_features.shape),
        str(validation_labels.shape),
        f"{validation_features.index[0]:.2f} -> {validation_features.index[-1]:.2f}",
        str(sorted(validation_labels["label"].unique().tolist())),
    )

    console.print(table)
    print()

    # Display model configuration
    config_table = Table(
        title="Model Configuration", show_header=True, header_style="bold magenta"
    )
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")

    config_table.add_row("Architecture", "EEGNet")
    config_table.add_row("Use PCA", str(use_pca))
    if use_pca:
        config_table.add_row("Projected Channels", str(args.projected_channels))
    else:
        config_table.add_row("Input Channels", str(train_features.shape[1]))
    config_table.add_row("Window Size", f"{args.window_size} samples ({args.window_size}ms)")
    config_table.add_row("Temporal Filters (F1)", str(args.F1))
    config_table.add_row("Depthwise Multiplier (D)", str(args.D))
    config_table.add_row("Epochs", str(args.epochs))
    config_table.add_row("Batch Size", str(args.batch_size))
    config_table.add_row("Learning Rate", str(args.learning_rate))

    console.print(config_table)
    print()

    # Create and train model
    rprint("\n[bold green]Training EEGNet model...[/]\n")

    model = EEGNet(
        input_size=train_features.shape[1],
        projected_channels=args.projected_channels,
        window_size=args.window_size,
        F1=args.F1,
        D=args.D,
        dropout=DROPOUT,
        use_pca=use_pca,
    )

    model.fit(
        X=train_features.values,
        y=train_labels["label"].values,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        verbose=True,
        X_val=validation_features.values,
        y_val=validation_labels["label"].values,
    )

    # Evaluate on validation set
    rprint("\n[bold green]Evaluating on validation set...[/]\n")
    evaluator = ModelEvaluator(
        test_features=validation_features,
        test_labels=validation_labels[["label"]],
    )

    metrics = evaluator.evaluate()
    evaluator.print_summary(metrics)

    rprint("\n[bold green]Training and evaluation complete.[/]\n")


if __name__ == "__main__":
    main()

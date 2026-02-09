#!/usr/bin/env python3
"""
Train and evaluate EMA Network for BCI classification.

This script demonstrates training a QSimeonEMANet model on ECoG data
for continuous frequency classification task.

Usage:
    python examples/train_ema_net.py

The trained model is saved to model.pt and metadata to model_metadata.json.
"""

import argparse
from pathlib import Path
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from brainstorm.download import download_train_validation_data
from brainstorm.loading import load_raw_data
from brainstorm.evaluation import ModelEvaluator
from brainstorm.ml.qsimeon_ema_net import QSimeonEMANet

DATA_PATH = Path("./data")

# EMA Network Hyperparameters
PROJECTED_CHANNELS = 64  # After PCA reduction from 1024
EMA_NODES = 32           # Number of EMA nodes
WINDOW_SIZE = 1600        # 1600ms at 1000Hz sampling rate
TEMPERATURE = 1.0        # Initial Gumbel-Softmax temperature
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

def parse_args():
    parser = argparse.ArgumentParser(description="Train EMA Network for BCI classification")
    
    # Model architecture
    parser.add_argument("--projected-channels", type=int, default=PROJECTED_CHANNELS,
                        help=f"Number of channels after PCA projection (default: {PROJECTED_CHANNELS})")
    parser.add_argument("--ema-nodes", type=int, default=EMA_NODES,
                        help=f"Number of EMA nodes (default: {EMA_NODES})")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE,
                        help=f"Temporal context window in samples (default: {WINDOW_SIZE})")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                        help=f"Initial Gumbel-Softmax temperature (default: {TEMPERATURE})")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Number of training epochs (default: {EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help=f"Learning rate (default: {LEARNING_RATE})")

    return parser.parse_args()
def main() -> None:
    """Train and evaluate EMA Network."""
    args = parse_args()

    rprint("\n[bold cyan]EMA Network Training for BCI Classification[/]\n")

    # Download data if not already present
    if not DATA_PATH.exists() or not any(DATA_PATH.glob("*.parquet")):
        rprint("[yellow]Downloading data...[/]")
        download_train_validation_data()
        rprint("[green]✓ Data downloaded successfully![/]\n")

    # Load data
    rprint(f"[cyan]Loading data from:[/] {DATA_PATH}\n")
    train_features, train_labels = load_raw_data(DATA_PATH, step="train")
    val_features, val_labels = load_raw_data(DATA_PATH, step="validation")

    # Display dataset info
    console = Console()
    table = Table(title="Dataset Overview", show_header=True, header_style="bold magenta")
    table.add_column("Split", style="cyan", width=10)
    table.add_column("Features", style="green")
    table.add_column("Labels", style="green")
    table.add_column("Unique Classes", style="blue")

    table.add_row(
        "Train",
        str(train_features.shape),
        str(train_labels.shape),
        str(sorted(train_labels["label"].unique().tolist()))
    )
    table.add_row(
        "Val",
        str(val_features.shape),
        str(val_labels.shape),
        str(sorted(val_labels["label"].unique().tolist()))
    )
    console.print(table)
    print()

    # Create and train model
    rprint("[bold green]Training EMA Network...[/]\n")
    model = QSimeonEMANet(
        input_size=train_features.shape[1],
        projected_channels=PROJECTED_CHANNELS,
        ema_nodes=EMA_NODES,
        window_size=WINDOW_SIZE,
        temperature=TEMPERATURE,
    )

    # fit() will call fit_model(), save the model, validate it, and save metadata
    model.fit(
        X=train_features.values,
        y=train_labels["label"].values,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        X_val=val_features.values,
        y_val=val_labels["label"].values,
    )

    # Evaluate on validation set
    rprint("\n[bold green]Evaluating on validation set...[/]\n")
    evaluator = ModelEvaluator(
        test_features=val_features,
        test_labels=val_labels[["label"]],
    )

    metrics = evaluator.evaluate()
    evaluator.print_summary(metrics)

    rprint("\n[bold green]✓ Training and evaluation complete![/]\n")


if __name__ == "__main__":
    main()

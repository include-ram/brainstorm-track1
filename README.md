# BrainStorm 2026 - Track 1

> **My contribution summary** — see [What I Built](#what-i-built) below.

Welcome to the BrainStorm 2026 Brain-Computer Interface (BCI) Hackathon! Build accurate, fast, and lightweight neural decoders for real-time auditory stimulus classification from ECoG recordings.

## Quick Start

```bash
# 1. Install dependencies
make install
source .venv/bin/activate

# 2. Download data
uv run python -c "from brainstorm.download import download_train_validation_data; download_train_validation_data()"

# 3. Train and evaluate
uv run python examples/example_local_train_and_evaluate.py
```

## Documentation

### Getting Started

1. **[Overview](docs/overview.md)** - Understand the problem, constraints, and scoring
2. **[Installation](docs/installation.md)** - Set up your environment
3. **[Dataset](docs/dataset.md)** - Learn about the ECoG data format

### Building Your Solution

4. **[Defining a Model](docs/defining_a_model.md)** - Create custom models
5. **[Evaluation](docs/evaluation.md)** - Test locally and understand scoring
6. **[Submissions](docs/submissions.md)** - Submit for test set evaluation
7. **[FAQ](docs/faq.md)** - Common questions

## Project Structure

```
brainstorm2026-track1/
├── brainstorm/           # Core library
│   ├── ml/              # Model implementations
│   ├── loading.py       # Data loading
│   ├── spatial.py       # Spatial utilities
│   └── plotting.py      # Visualization
├── docs/                # Documentation
├── examples/            # Example scripts
└── tests/               # Test suite
```

## Rules

✅ **Allowed:** Any Python libraries, ensemble models, AI coding tools, pre-trained models

❌ **Not Allowed:** Non-causal models, modifying evaluation code, models >25MB

See the [FAQ](docs/faq.md) for detailed rules.

---

## What I Built

### Problem
Decode auditory stimuli in real-time from 1024-channel micro-ECoG brain recordings on a streaming, sample-by-sample basis — with hard constraints on model size (<25 MB), inference latency, and causality (no future data).

### My Contributions

#### 1. QSimeonEMANet — Novel Streaming Architecture (`brainstorm/ml/qsimeon_ema_net.py`)
Designed and implemented a custom neural architecture combining **Exponential Moving Averages (EMA)** with **Gumbel-Softmax channel mixing** for differentiable, learned channel selection.

**How it works:**
- **PCA Projection**: Reduces 1024 channels → 64 for efficiency
- **EMA Layer**: Each node maintains a causal temporal state:
  `h[k](t) = α[k] · selected_input[k](t) + (1 - α[k]) · h[k](t-1)`
- **Gumbel-Softmax**: Learns which channels to attend to via differentiable discrete sampling
- **Temperature Annealing**: Smooths the discrete channel selection during training (1.0 → 0.5)
- **Streaming Buffer**: 1600-sample sliding window for single-sample causal inference

**Why this matters:** Standard neural nets are non-causal or stateless. EMANet is inherently causal and stateful — it learns what to remember without needing future data, making it suitable for real implanted hardware.

#### 2. EEGNet Adaptation (`brainstorm/ml/eegnet.py`)
Adapted the classic Lawhern et al. (2018) depthwise separable CNN to the 1024-channel ECoG setting:
- Added **PCA preprocessing** to handle the high channel count
- Integrated **class-weighted loss** for the imbalanced stimulus distribution
- Added **cosine annealing LR schedule** and gradient clipping
- Implemented **streaming inference** with a causal sliding window buffer

#### 3. Training Scripts (`examples/train_ema_net.py`, `examples/train_eegnet.py`)
- Standalone, configurable training scripts for both models
- Automatic device selection (CUDA → MPS → CPU)
- Validation-based checkpointing to save best model
- AdamW optimizer with weight decay

#### 4. Benchmarking & Comparison (`examples/benchmark_eegnet.py`, `examples/compare_models.py`)
- Built a local benchmarking framework to measure all three competition metrics (accuracy, latency, size)
- Compared EEGNet vs Wav2Vec2Classifier side-by-side
- Grid-searched EEGNet window size and PCA channel configurations

### Scoring Formula
```
Score = 50 × balanced_accuracy
      + 25 × exp(-6 × latency_ms / 500)
      + 25 × exp(-4 × size_mb / 5)
```

### Key Design Decisions
| Decision | Reason |
|---|---|
| PCA to 64 channels | Reduces 1024-channel dimensionality without losing spatial structure |
| EMA-based state | Causal, stateful, and hardware-friendly — no attention or future lookback |
| Gumbel-Softmax channel mixing | End-to-end differentiable channel selection |
| Depthwise separable convolutions (EEGNet) | Minimizes parameter count for size budget |
| Sliding window buffer in all models | Enables single-sample streaming inference required by evaluator |

---

Good luck, and happy hacking! 🧠⚡

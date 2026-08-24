# Fine-tune OpenVLA on LIBERO Spatial

This guide installs a local environment and starts supervised fine-tuning
(SFT) of the base `openvla/openvla-7b` checkpoint on the
`lerobot/libero_spatial_image` dataset. The checked-in recipe uses one node
with eight NVIDIA GPUs and trains a LoRA adapter.

The OpenVLA SFT path has passed integration and partial-training checks, but a
complete training run has not yet been used to establish a reference success
rate. This page therefore documents setup and launch only.

## Install the environment

The environment requires Python 3.10 and an NVIDIA driver compatible with
PyTorch 2.7.1. On Ubuntu 22.04, install the system packages from the repository
root:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  ffmpeg \
  git \
  libgl1 \
  libglib2.0-0 \
  python3.10-dev \
  python3.10-venv
```

Create a repository-local virtual environment and install the pinned LeRobot
runtime before verl-vla:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip 'setuptools>=71,<81' wheel
python -m pip install --requirement requirements-lerobot.txt
python -m pip install --no-deps lerobot==0.4.4
python -m pip install --editable '.[openvla-oft]'
```

Activate the environment again in each new terminal:

```bash
source .venv/bin/activate
```

This SFT workflow reads an offline LIBERO dataset and does not start the
simulator, so installing the LIBERO simulator assets is not required.

## Start training

From the repository root, run:

```bash
source .venv/bin/activate
bash examples/fine_tuning/openvla/libero_spatial/run_train.sh
```

On the first launch, the script downloads the dataset, computes its action
normalization statistics, downloads the base OpenVLA checkpoint through the
standard Hugging Face cache, and starts distributed SFT on eight GPUs.

The launcher accepts these environment variables when the default locations
are unsuitable:

```bash
OPENVLA_MODEL=/path/to/openvla-7b \
DATASET_ROOT=/path/to/libero_spatial_image \
NORM_STATS_PATH=/path/to/norm_stats.json \
bash examples/fine_tuning/openvla/libero_spatial/run_train.sh
```

`OPENVLA_MODEL` may be either a local native OpenVLA checkpoint or a Hugging
Face model identifier. The default is the base `openvla/openvla-7b` checkpoint,
not a checkpoint already fine-tuned on LIBERO. The default dataset directory is
`.data/libero_spatial_image`; when `NORM_STATS_PATH` is not set, it resolves to
`norm_stats.json` under that directory.

Training checkpoints and native Hugging Face exports are written under:

```text
outputs/train/openvla-sft/libero-spatial-base/checkpoints
```

TensorBoard event files are written under:

```text
outputs/train/openvla-sft/libero-spatial-base/tensorboard
```

Additional Hydra overrides can be appended to the same command. For example,
use a different GPU count only together with matching worker and batch settings
appropriate for that machine.

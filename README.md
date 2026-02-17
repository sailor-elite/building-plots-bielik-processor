# Building Plots Bielik Processor

A specialized Jupyter Notebook designed to process listings using the **Bielik-1.5B-v3** Large Language Model.

## Workflow
1. **Fetch:** Connects to a remote VPS database to download raw scraped listings.
2. **Process:** Uses [Bielik-1.5B-v3](https://huggingface.co/speakleash/Bielik-1.5B-v3) to analyze text and extract structured information.
3. **Upload:** Sends the processed data back to the VPS.

## Setup

### Hardware Requirements
* **GPU:** NVIDIA GPU (e.g., RTX 4050) with at least 6GB VRAM.
* **Driver:** Latest NVIDIA Drivers with CUDA 12.1+ support.

## Installation

This project uses `uv` for fast, reproducible dependency management.

1. **Clone the repository and sync the environment:**
   ```bash
   uv sync
   ```
2. **Force install CUDA PyTorch version:**
   ```bash
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
   ```
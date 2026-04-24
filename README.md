# Building Plots Bielik Processor

Jupyter Notebook designed to process listings using the **Bielik-1.5B-v3** Large Language Model.

### Related Repositories
* **Data Processing & LLM:** [building-plots-bielik-processor](https://github.com/sailor-elite/building-plots-bielik-processor)
* **Transformation & EDA:** [Estate-data-transformation](https://github.com/sailor-elite/Estate-data-transformation/)

## Workflow
1. **Fetch:** Connects to a remote VPS database to download scraped listings.
2. **Process:** Uses [Bielik-1.5B-v3](https://huggingface.co/speakleash/Bielik-1.5B-v3) to analyze text and extract structured information.
3. **Upload:** Sends the processed data back to the VPS.

## Setup

### Hardware Requirements
* **GPU:** NVIDIA GPU (e.g., RTX 4050) with at least 6GB VRAM.
* **Driver:** Latest NVIDIA Drivers with CUDA 12.1+ support.

## Installation

This project uses `uv` for reproducible dependency management.

1. **Clone the repository and sync the environment:**
   ```bash
   uv sync
   ```
2. **Force install CUDA PyTorch version:**
   ```bash
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
   ```

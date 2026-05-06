# MUSE GAN: A MULTIVIEW MODIFIED GAN ARCHITECTURE FOR SATELLITE IMAGERY SUPER RESOLUTION

**Team Members:**
- Devika Krishna Iyer (22BAI1281)
- Shaurya (22BAI1173)
- Vishesh Panchal (22BAI1226)

**School of Computer Science and Engineering, Vellore Institute of Technology, Chennai**

---

## Abstract

Satellite imagery frequently suffers from low resolution and quality degradation due to atmospheric interference. To address these challenges, MUSE-GAN implements a robust **Multi-View Modified GAN** architecture for super-resolution. By leveraging the **WorldStrat dataset**, this project utilizes an N-to-1 super-resolution pipeline that fuses multiple low-resolution Sentinel-2 images (temporal) with machine-vision derived structural features to generate high-quality, spectrally accurate high-resolution images comparable to commercial Maxar imagery (4x upscaling).

## Project Overview

This project addresses the gap between freely available low-resolution satellite imagery and the high-resolution requirements for applications like urban planning, agriculture, and disaster management.

**Key Features:**
- **Multi-Modal Input**: Integrates temporal data (8 Sentinel-2 frames) and structural priors (Canny Edges, LBP, Sobel Gradients).
- **Scientific Integrity**: Prioritizes not just visual realism but also radiometric and spectral consistency using a custom composite loss function.
- **Robustness**: Trained on the diverse WorldStrat dataset to handle real-world conditions like haze, clouds, and seasonal changes.

## Dataset: WorldStrat

We utilize the **WorldStrat** dataset, which provides:
- **High-Resolution (HR)**: Airbus SPOT 6/7 imagery (1.5m/pixel).
- **Low-Resolution (LR)**: Sentinel-2 imagery (10m/pixel).
- **Structure**: N-to-1 pairing, where 8 LR temporal revisits are paired with 1 HR ground truth.
- **Diversity**: Stratified sampling across settlement density, land-use classes (forests, agriculture), and underrepresented humanitarian sites.

## Project Structure

```
MUSE-GAN/
├── Notebook/               # Jupyter notebooks for experimentation
├── Reports/                # Project reports and documentation
├── src/                    # Source code directory
│   ├── dataset.py          # Dataset loading and pipeline management
│   ├── evaluate.py         # Evaluation metrics and scripts
│   ├── losses.py           # Custom loss functions (Adversarial, Content, Spectral)
│   ├── models.py           # Model architectures (Generator, Discriminator)
│   ├── train.py            # Main training script
│   ├── utils.py            # Utility functions
│   └── visualize.py        # Visualization tools
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```

## Architecture

### 1. Input Formulation (27-Channel Tensor)
The model processes a rich, high-dimensional input:
- **Temporal Data**: 8 Low-Resolution Sentinel-2 frames stacked channel-wise (8 frames × 3 RGB = 24 channels).
- **Structural Priors**: 3 single-channel feature maps extracted from a reference LR frame:
    - **Canny Edges**: For boundary detection.
    - **Local Binary Patterns (LBP)**: For texture analysis.
    - **Sobel Gradients**: For spatial transitions.
- **Total Input**: `(B, 27, 160, 160)`

### 2. Generator (MUSE-GAN)
A U-Net-based architecture designed for 4x super-resolution:
- **Encoder**: Captures hierarchical features using **Enhanced Residual Blocks** with **CBAM (Convolutional Block Attention Module)**.
- **Bottleneck**: Uses **Residual Dense Blocks (RDBs)** for deep feature aggregation and reuse.
- **Decoder**: Reconstructs high-resolution details using skip connections and progressive upsampling.
- **Upsampling**: Two-stage progressive upsampling (2x -> 2x) to reach the target 640x640 resolution.

### 3. Discriminator
A **Conditional PatchGAN** discriminator that evaluates local image patches for realism:
- **Input**: Concatenation of the generated/real image and the upsampled vision feature maps.
- **Spectral Normalization**: Applied to all layers to stabilize training.
- **Output**: A grid of "realness" scores, forcing the generator to produce plausible high-frequency textures.

## Loss Functions

A composite objective function balances three competing goals:
1.  **Adversarial Loss**: For perceptual realism (making images look natural).
2.  **Content Loss (L1)**: For pixel-level structural accuracy.
3.  **Spectral Fidelity Loss (SAM)**: Uses the **Spectral Angle Mapper** to preserve the angular relationship between spectral vectors, ensuring scientific accuracy for remote sensing analysis.

## Results

The model was evaluated on 393 diverse test samples from the WorldStrat dataset.

| Metric | Mean Score | Description |
| :--- | :--- | :--- |
| **PSNR** | **8.11 dB** | ~2x improvement over traditional interpolation; 35% over SRCNN. |
| **SSIM** | **0.206** | Strong structural preservation, especially in urban scenes. |
| **SAM** | **10.54°** | Excellent spectral fidelity (lower is better), outperforming standard GANs. |
| **LPIPS** | **0.919** | Consistent perceptual quality. |
| **FID** | **281.52** | Good distributional similarity to real HR imagery. |

## Installation & Usage

### Prerequisites
- Python 3.8+
- PyTorch
- CUDA (recommended)

### Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Training
To train the model, use the `src/train.py` script.
```bash
python -m src.train --dataset_path /path/to/worldstrat --epochs 30 --batch_size 4
```

## Future Work

- **Dynamic Temporal Fusion**: Implementing attention mechanisms (e.g., Transformers) to intelligently weight temporal frames based on quality (e.g., ignoring cloudy frames).
- **Progressive Growing**: Adopting progressive training strategies for better stability at higher resolutions.
- **Diffusion Models**: Exploring diffusion-based generation for potentially higher perceptual quality.

## References
Based on the project report "MUSE GAN: A MULTIVIEW MODIFIED GAN ARCHITECTURE FOR SATELLITE IMAGERY SUPER RESOLUTION" (Nov 2025).

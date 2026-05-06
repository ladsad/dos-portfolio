# ArtResGAN

ArtResGAN is a deep learning framework for the restoration of degraded artworks, designed to preserve both the structural integrity and the unique artistic style of paintings.The project leverages a hybrid U - Net + ResNet generator, a PatchGAN discriminator, and a VGG - based style extractor, integrating classical machine vision techniques to achieve high - quality, stylistically faithful restorations.This repository accompanies an[unpublished paper draft](./ paper - draft / final_draft - ArtResGAN % 20A % 20GAN - Based % 20Approach % 20for% 20Image % 20Restoration % 20and % 20Style % 20Preservation.pdf) detailing the methodology and results.

### Authors
  - ** Shaurya **: [GitHub Profile](https://github.com/ladsad)
- ** Devika Iyer **: [GitHub Profile](https://github.com/DevikaIyer23)
- ** Vishesh Panchal ** : [GitHub Profile](/)

---

** Repository:** [ArtResGAN on GitHub](https://github.com/ladsad/ArtResGAN)
** Notebook Version:** [Kaggle Notebook](https://www.kaggle.com/code/shaurya22bai1173/artresgan-final/)

    ---

## Features

  - ** Hybrid Generator:** Combines U - Net and ResNet architectures for capturing both global context and fine details.
- ** PatchGAN Discriminator:** Evaluates local patches to enforce realistic texture and style.
- ** VGG - based Style Loss:** Preserves the original artistic style using perceptual losses.
- ** Machine Vision Integration:** Incorporates edge detection, morphological operations, wavelet transforms, and local binary patterns for enhanced detail and texture preservation.
- ** Optional ESRGAN Upscaler:** Enables high - resolution output for museum - quality restorations.
- ** Comprehensive Loss Functions:** Balances adversarial, content, style, and total variation losses for optimal restoration results.


## Attached Paper

This repository includes an[unpublished paper draft](./ paper - draft / final_draft - ArtResGAN % 20A % 20GAN - Based % 20Approach % 20for% 20Image % 20Restoration % 20and % 20Style % 20Preservation.pdf) that provides a detailed description of the architecture, methodology, and experimental results.The paper outlines the motivation behind ArtResGAN, its technical innovations, and its performance on benchmark datasets.

## Quick Start

  ** Requirements **

  - Python 3.7 +
    - See `requirements.txt` for all dependencies

**Installation**

1. Clone the repository:

```bash
git clone https://github.com/ladsad/ArtResGAN.git
cd ArtResGAN
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```


**Dataset**

- The model is trained on the [WikiArt dataset](https://www.wikiart.org/).
- Prepare your dataset following the structure and preprocessing steps outlined in the paper or the notebook.

**Training**

To train the model:

```bash
python train.py --config config.py
```

- Adjust hyperparameters and paths in `config.py` as needed.

**Testing/Evaluation**

To evaluate on test images:

```bash
python test.py --config config.py --input_dir <path_to_degraded_images> --output_dir <path_to_save_results>
```

**Notebook Version**

A ready-to-run notebook version is available on Kaggle for easy experimentation and demonstration:

- [ArtResGAN Kaggle Notebook](https://www.kaggle.com/code/shaurya22bai1173/artresgan-final/)


## Repository Structure

- `models/` - Model definitions (Generator, Discriminator, etc.)
- `utils/` - Utility functions for preprocessing, metrics, etc.
- `notebook/` - Jupyter notebooks for experiments and demonstrations
- `paper-draft/` - Unpublished paper draft (PDF)
- `config.py` - Configuration file for hyperparameters and paths
- `requirements.txt` - Python dependencies
- `train.py` - Training script
- `test.py` - Testing/inference script


## How It Works

ArtResGAN restores degraded artworks by:

- Extracting structural features using machine vision techniques
- Feeding both the degraded image and extracted features into a hybrid generator
- Using a PatchGAN discriminator to enforce realism at the patch level
- Applying VGG-based perceptual losses to maintain content and stylistic fidelity
- Optionally upscaling outputs for high-resolution restoration

For a detailed explanation of the architecture and methodology, refer to the [unpublished paper draft](./paper-draft/final_draft-ArtResGAN%20A%20GAN-Based%20Approach%20for%20Image%20Restoration%20and%20Style%20Preservation.pdf).

## Citation

If you use this work, please cite the attached paper draft or acknowledge the authors.

---

**Authors:**
Shaurya, Devika Iyer, Vishesh Panchal

For questions or contributions, please open an issue or pull request on GitHub.

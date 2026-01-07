export const portfolioData = {
  header: {
    name: "Shaurya Kumar",
    location: "New Delhi, India",
    email: "emailofshauryak@gmail.com",
    linkedin: "https://www.linkedin.com/in/shaurya-kumar-22262b236/",
    github: "https://github.com/ladsad"
  },
  education: [
    {
      institution: "Vellore Institute of Technology (VIT), Chennai",
      degree: "B.Tech. Computer Science (AI & ML)",
      period: "Sept 2022 – July 2026",
      details: [
        "CGPA: 8.94/10.0",
        "Coursework: Statistics, Data Structures, Machine/Deep Learning, Database Systems, Software Engineering"
      ]
    }
  ],
  experience: [
    {
      role: "Computer Vision & Data Engineering Intern",
      company: "Bidaal",
      period: "June 2025 – July 2025",
      highlights: [
        "Developed scalable data pipelines and ML inference systems using DeepStream and Python, reducing algorithmic latency by 25% and broadening deployment scalability by 40%",
        "Conducted exploratory data analysis, feature engineering and model diagnostics for large-scale computer vision datasets"
      ]
    },
    {
      role: "Software Engineer Intern (Data Infrastructure)",
      company: "eMudhra",
      period: "June 2024 – July 2024",
      highlights: [
        "Designed and built robust backend data services using Java Server Pages and Apache Tomcat, leveraging RESTful microservices. Implemented Java Servlets for optimized data processing, achieving 10–100× performance boost for analytics workloads",
        "Utilized advanced MySQL clustering for high-speed, reliable data retrieval in real-time scenarios"
      ]
    }
  ],
  projects: [
    {
      name: "Churn HTE: Causal ML",
      link: "https://github.com/ladsad/churn-hte",
      category: "Causal ML & Production AI",
      highlights: [
        "Beyond simple churn prediction: Identifies *who* to target using Heterogeneous Treatment Effects (HTE)",
        "Implements Doubly Robust Estimation and Causal Forests (EconML) for unbiased causal inference",
        "Deployable FastAPI service for real-time intervention scoring and A/B test simulation"
      ],
      content: `# Churn Prediction with Heterogeneous Treatment Effects

A causal ML system that predicts customer churn **and** determines the best personalized intervention for each customer using causal inference.

## Key Innovation
Most churn systems just predict "who churns". This project uses **Causal Forests** to estimate **Conditional Average Treatment Effects (CATE)**, identifying customers who will specifically respond to an intervention, rather than wasting resources on lost causes or loyal customers.

## Methodology
1.  **Churn Prediction**: LightGBM model (AUC ~0.82) to identify at-risk customers.
2.  **Causal Effect Estimation**: Doubly Robust Estimation to isolate true intervention effects, removing selection bias.
3.  **Heterogeneity**: Causal Forests (econml) to estimate per-customer treatment effects.
4.  **Production**: FastAPI service for real-time recommendations.

## Results
-   **Average Treatment Effect**: -12% (Intervention reduces churn probability by 12pp).
-   **A/B Test Validation**: Targeted interventions on high-CATE segments yielded a **24% relative churn reduction** compared to random targeting.
-   **Business Impact**: Positive ROI (~2x) by optimizing intervention costs.

## Tech Stack
-   **Causal ML**: EconML, CausalML, DoWhy
-   **Machine Learning**: LightGBM, Scikit-Learn
-   **Production**: FastAPI, Docker
-   **Analysis**: Jupyter, Pandas, Matplotlib
`
    },
    {
      name: "CodeWhisper",
      link: "https://github.com/ladsad/codewhisper",
      category: "Developer Tools & AI",
      highlights: [
        "Intelligent tool for auto-generating documentation and analyzing code quality using CodeT5+",
        "Features a VS Code extension for real-time docstring generation and a FastAPI backend with a Streamlit dashboard",
        "Fine-tuned CodeT5-small on CodeXGLUE (Python/Java) using QLoRA for efficient model training"
      ],
      content: `# CodeWhisper

CodeWhisper is an intelligent tool for auto-generating documentation and analyzing code quality.

## Project Structure

- \`backend/\`: Python FastAPI backend.
- \`vscode-extension/\`: VS Code extension for IDE integration.
- \`Documents/\`: Project documentation.

## Features

- **Code Analysis**: Calculates cyclomatic complexity, maintainability index, and detects anomalies.
- **Auto-Documentation**: Generates docstrings using **CodeT5+** fine-tuned with **QLoRA**.
- **Dashboard**: Visualizes project health and metrics (Streamlit MVP).
- **VS Code Extension**: Right-click context menu for real-time documentation generation.

## Model Training

The documentation generation model uses **CodeT5-small** fine-tuned on **CodeXGLUE** (Python/Java) with QLoRA.

### Model Performance

| Metric | Score |
| :--- | :--- |
| **BLEU** | 36.65 |
| **ROUGE-L** | 62.17 |
| **BERTScore** | 0.93 |
`
    },
    {
      name: "MicroSegNet Optimizer",
      link: "https://github.com/ladsad/Modified-MicroSegNet",
      category: "ML Pipeline Engineering",
      highlights: [
        "Built automated ML training pipelines with hyperparameter tuning, reducing model training time by 40%",
        "Engineered validation and cross-validation systems using TensorFlow and advanced statistics for model integrity"
      ],
      content: `# Modified-MicroSegNet
Modified Official PyTorch implementation of: 

[MicroSegNet: A Deep Learning Approach for Prostate Segmentation on Micro-Ultrasound Images](https://www.sciencedirect.com/science/article/pii/S089561112400003X) (CMIG 2024)

In colaboration by:
 - Shaurya: github.com/ladsad
 - Devika Iyer: github.com/DevikaIyer23
 - Adisha Shaikh: github.com/adisha-shaikh

## Requirements
* Python==3.9.13
* torch==2.1.0
* torchvision==0.16.0
* numpy
* opencv-python
* tqdm
* tensorboard
* tensorboardX
* ml-collections
* medpy
* SimpleITK
* scipy
* \`pip install -r requirements.txt\`

## Dataset
- Micro-Ultrasound Prostate Segmentation Dataset
- Dataset can be accessed here: https://zenodo.org/records/10475293.

## Usage
### 1. Download Google pre-trained ViT models
* [Get models in this link](https://console.cloud.google.com/storage/vit_models/). "imagenet21k/R50+ViT-B_16.npz" is used here.
* Rename your model as: R50-ViT-B_16, ViT-B_16, ViT-L_16.....
* Save your model into folder "model/vit_checkpoint/imagenet21k/".
* If you want to use models pretrained on imagenet21k+imagenet2012, please add configs in ["TransUNet/networks/vit_seg_configs.py"](TransUNet/networks/vit_seg_configs.py)

### 2. Prepare data
* Please go to https://zenodo.org/records/10475293 to download our dataset.
* After downloading, extract the file and put it into folder "data/". The directory structure should be as follows:

\`\`\`bash
.
├── data
│   ├── Micro_Ultrasound_Prostate_Segmentation_Dataset
│   │   ├── train
│   │   ├── test
│   ├── preprocessing.py
│
├── model
│   ├── vit_checkpoint
│   │     ├── imagenet21k
│   │       ├── R50+ViT-B_16.npz
│   │       ├── *.npz
│   ├── TransUNet

\`\`\`

* Run the preprocessing script, which would generate training images in folder "train_png/", data list files in folder "lists/" and data.csv for overview.
\`\`\`
python preprocessing.py
\`\`\`
* Training images are preprocessed to 224*224 to feed into networks.

### 3. Train/Test
* Please go to the folder "TransUNet/" and it's ready for you to train and test the model.
\`\`\`
python train_MicroUS.py
python test_MicroUS.py
\`\`\`
The hard region weight here is set to 4 as default, while you can train models with different weight by specifying it in the command line as follows:
\`\`\`
python train_MicroUS.py --weight 10
python test_MicroUS.py --weight 10
\`\`\`

## References
* Some part of the code is adapted from [TransUNet](https://github.com/Beckschen/TransUNet) ,
which provides a very good implementation to start with.
* [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
* [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)
* [MicroSegNet](https://github.com/mirthAI/MicroSegNet)

## Citations
\`\`\`
@article{jiang2024microsegnet,
  title={MicroSegNet: A deep learning approach for prostate segmentation on micro-ultrasound images},
  author={Jiang, Hongxu and Imran, Muhammad and Muralidharan, Preethika and Patel, Anjali and Pensa, Jake and Liang, Muxuan and Benidir, Tarik and Grajo, Joseph R and Joseph, Jason P and Terry, Russell and others},
  journal={Computerized Medical Imaging and Graphics},
  pages={102326},
  year={2024},
  publisher={Elsevier}
}
\`\`\`
`
    },
    {
      name: "Attention-Enhanced RHN",
      link: "https://github.com/ladsad/Integrating-Attention-mechanisms-into-Recurrent-Highway-Networks-with-Grouped-Auxiliary-Memory",
      category: "NLP Architectures",
      highlights: [
        "Improved sequence modeling by integrating attention mechanisms into RHNs, tested on Penn TreeBank data",
        "Designed global auxiliary memory for effective retention of contextual info—boosting NLP model performance"
      ],
      content: `# Attention Mechanism on GAM-RHN

This repository provides the implementation of attention mechanisms integrated into Grouped Auxiliary Memory Recurrent Highway Networks (GAM-RHN). It includes modules for data preprocessing, model training, evaluation, and implementation of various attention mechanisms.

This work is based on a yet-to-be-published paper titled [Integrating Attention Mechanisms into Recurrent Highway Networks with Grouped Auxiliary Memory](https://shorturl.at/tU8X7), developed collaboratively by:

- **Shaurya**: [GitHub Profile](https://github.com/ladsad)
- **Devika**: [GitHub Profile](https://github.com/DevikaIyer23)

---

## Project Structure

\`\`\`
Attention_Mechanism_on_GAM-RHN/
├── README.md                   # Project overview, installation instructions, and usage.
├── requirements.txt            # Python dependencies.
├── config.py                   # Configuration for hyperparameters, paths, etc.
├── data/
│   ├── data_loader.py          # Data loading and preprocessing scripts.
│   └── preprocess.py           # Functions for data preprocessing (e.g., tokenization).
├── models/
│   ├── rhn_cell.py             # Core RHN model architecture.
│   ├── gam_rhn.py              # Implementation of Grouped Auxiliary Memory logic.
│   ├── gam_rhn_attention.py    # RHN+GAM model with attention mechanisms.
│   └── attention.py            # Attention mechanism implementations.
├── train.py                    # Script for model training.
├── evaluate.py                 # Script for model evaluation.
└── notebooks/
    └── colab_notebook.ipynb    # Google Colab notebook for demonstration.
\`\`\`

---

## Installation

Follow these steps to set up the project:

1. **Clone the repository:**
   \`\`\`sh
   git clone https://github.com/yourusername/Attention_Mechanism_on_GAM-RHN.git
   cd Attention_Mechanism_on_GAM-RHN
   \`\`\`

2. **Set up a virtual environment:**
   \`\`\`sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use \`venv\\Scripts\\activate\`
   \`\`\`

3. **Install dependencies:**
   \`\`\`sh
   pip install -r requirements.txt
   \`\`\`

4. **Download save-state files:**
   Download the pretrained model files from the following Google Drive link:
   [Download Save-State Files](https://drive.google.com/drive/folders/1cR_AYvDu26eNPHIlp91wycJ5-UBMh4lD?usp=drive_link)

   After downloading, place the files in the \`models/save_state/\` directory.

---

## Usage

### Data Preprocessing

Prepare the dataset by running the preprocessing script:
\`\`\`sh
python data/preprocess.py
\`\`\`

### Training

Train the model using the training script:
\`\`\`sh
python train.py
\`\`\`

### Evaluation

Evaluate the trained model using the evaluation script:
\`\`\`sh
python evaluate.py
\`\`\`

---

## Configuration

All configurable settings such as hyperparameters, data paths, and model parameters are defined in the \`config.py\` file. Adjust the configurations as required to tailor the model for your specific use case.

---

## Models

The \`models/\` directory contains key components of the project:

- **\`rhn_cell.py\`**: Core implementation of the Recurrent Highway Network (RHN).
- **\`gam_rhn.py\`**: Grouped Auxiliary Memory (GAM) logic.
- **\`gam_rhn_attention.py\`**: GAM-RHN model with integrated attention mechanisms.
- **\`attention.py\`**: Implementation of various attention mechanisms.

---

## Contributing

We welcome contributions! If you have ideas for improvements or fixes, please open an issue or submit a pull request. 

---

## References

 - Shaurya: github.com/ladsad
 - Devika: github.com/devikaiyer23

# Problem Statement
Our project aims at building a consultancy platform that streamlines service delivery by connecting clients with skilled professionals. Clients browse and request diverse projects, while consultants and providers deliver their expertise. This platform will bridge the gap, ensuring clients get the services they need and the consultancy thrives with new projects. By implementing a strong database system, consultancy services can achieve significant improvements in operational efficiency, client service, and overall business performance.

# Problem Description
Currently consultancy services rely on extensive spreadsheets and manual entry to store and retrieve data pertaining to different professional services. This leads to distribution of extensive amounts of data across multiple systems and files thus making it difficult to access and report. This leads to loss in productivity as employees spend extensive amounts of time searching for and managing data. Lack of a centralized database also results in inefficient workflows and data redundancy. Thus it is essential to create a database which stores all the information pertaining to the different services available. This makes requesting for a service and initiating and working on projects more efficient for clients and also for the consultants building the project.

# ER Diagram
<img src="https://i.imgur.com/OaxM1Vp.png"/>

# Tech Stack
 - Frontend: ReactJS, Vanilla CSS
 - Backend: NodeJS, ExpressJS
 - Database: MySQL

# How to Run
 - Clone the repository.
 - Import the database from the database folder onto your MySQL local server.
 - Input the database information into \`\`\`backend/server.js\`\`\`.
 - Open the terminal in the project root folder and run \`\`\`npm start\`\`\`.
 - You can now use the project.

# Future Plans
 - Add the further functionalities for the consultants.
 - Rehall the UI using Tailwind CSS and React Libraries.
 - Move the database to cloud.
`
    },
    {
      name: "Mustard Archives",
      link: "https://github.com/ladsad/Mustard-Archives",
      category: "Full Stack Consultancy Platform",
      highlights: [
        "Developed a centralized consultancy platform connecting clients with skilled professionals",
        "Implemented a robust MySQL database to streamline service delivery and reduce data redundancy",
        "Built a responsive React frontend and Express/Node.js backend for efficient project management"
      ],
      content: `
# Mustard Archive

## Overview
A full-stack consultancy platform designed to streamline service delivery by connecting clients with skilled professionals. It addresses the inefficiencies of manual data management by providing a centralized system for browsing services, requesting projects, and managing consultancy workflows.

## Problem Statement
Currently, consultancy services rely on extensive spreadsheets and manual entry, leading to data fragmentation, redundancy, and inefficiency. Mustard Archive bridges this gap by offering a unified platform that enhances operational efficiency, client service, and overall business performance.

## Key Features
- **Service Discovery**: Clients can easily browse and request diverse professional services.
- **Project Management**: Streamlines the process of initiating and working on projects for both clients and consultants.
- **Centralized Database**: Replaces disparate files with a structured MySQL database, ensuring data integrity and easy access.
- **Efficiency**: Reduces time spent searching for data and managing manual workflows.

## Architecture
- **Frontend**: ReactJS with Vanilla CSS for a clean and responsive user interface.
- **Backend**: NodeJS and ExpressJS for a scalable and robust server-side architecture.
- **Database**: MySQL for structured and reliable data storage.

## Future Plans
- **Enhanced Consultant Features**: Adding more tools and functionalities specifically for consultants.
- **UI Overhaul**: Migrating to Tailwind CSS and modern React libraries for a polished look.
- **Cloud Migration**: Moving the database to the cloud for better accessibility and scalability.
`
    },
    {
      name: "AWS Sentiment Analysis",
      link: "https://github.com/ladsad/AWS-SentimentAnalysisRedit-Frontend",
      category: "Cloud AI Architecture",
      highlights: [
        "Architected scalable cloud-native NLP solutions leveraging AWS Lambda, EC2, and API Gateway",
        "Designed RESTful APIs and data communication strategies for seamless client-server interaction",
        "Utilized AWS Comprehend for text classification and analytics, S3 for robust cloud data warehousing"
      ],
      content: `
# AWS Sentiment Analysis Platform

## Overview
A full-stack application that performs real-time sentiment analysis on Reddit comments from specific subreddits. It leverages AWS cloud services for serverless processing and provides a visual dashboard for sentiment distribution.

## Architecture
- **Frontend**: React.js with Chart.js for data visualization.
- **Backend**: Express.js (Local) & AWS Lambda (Serverless).
- **Cloud Infrastructure**: 
    - **AWS API Gateway**: Entry point for triggering analysis.
    - **AWS Lambda**: Serverless compute for fetching and processing Reddit data.
    - **AWS Comprehend** (Implied): For Natural Language Processing (NLP) and sentiment scoring.
    - **DynamoDB** (Implied): For storing analyzed comment data.

## Key Features
- **Real-time Analysis**: Fetches live comments from subreddits like r/aws, r/python, and r/askreddit.
- **Interactive Dashboard**: Visualizes sentiment distribution (Positive, Negative, Neutral) using dynamic Pie charts.
- **Detailed Insights**: Displays individual comments with their calculated sentiment scores and metadata.
- **Hybrid Architecture**: Demonstrates integration between local development servers and cloud-native AWS services.

## Tech Stack
- **Frontend**: React, Axios, Chart.js
- **Cloud**: AWS API Gateway, Lambda
- **Backend**: Node.js, Express
`
    },
    {
      name: "ArtResGAN",
      link: "https://github.com/ladsad/ArtResGAN",
      category: "Vision & GAN Systems",
      highlights: [
        "Engineered hybrid U-Net plus ResNet GAN architectures for restoring art images (WikiArt dataset)",
        "Leveraged classical vision and deep learning to enhance feature separation, texture, and structure",
        "Achieved high-fidelity results via adversarial, content, and style loss optimization"
      ],
      content: `# ArtResGAN

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
    - See \`requirements.txt\` for all dependencies

**Installation**

1. Clone the repository:

\`\`\`bash
git clone https://github.com/ladsad/ArtResGAN.git
cd ArtResGAN
\`\`\`

2. Install dependencies:

\`\`\`bash
pip install -r requirements.txt
\`\`\`


**Dataset**

- The model is trained on the [WikiArt dataset](https://www.wikiart.org/).
- Prepare your dataset following the structure and preprocessing steps outlined in the paper or the notebook.

**Training**

To train the model:

\`\`\`bash
python train.py --config config.py
\`\`\`

- Adjust hyperparameters and paths in \`config.py\` as needed.

**Testing/Evaluation**

To evaluate on test images:

\`\`\`bash
python test.py --config config.py --input_dir <path_to_degraded_images> --output_dir <path_to_save_results>
\`\`\`

**Notebook Version**

A ready-to-run notebook version is available on Kaggle for easy experimentation and demonstration:

- [ArtResGAN Kaggle Notebook](https://www.kaggle.com/code/shaurya22bai1173/artresgan-final/)


## Repository Structure

- \`models/\` - Model definitions (Generator, Discriminator, etc.)
- \`utils/\` - Utility functions for preprocessing, metrics, etc.
- \`notebook/\` - Jupyter notebooks for experiments and demonstrations
- \`paper-draft/\` - Unpublished paper draft (PDF)
- \`config.py\` - Configuration file for hyperparameters and paths
- \`requirements.txt\` - Python dependencies
- \`train.py\` - Training script
- \`test.py\` - Testing/inference script


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
`
    },
    {
      name: "MUSE-GAN",
      link: "https://github.com/ladsad/MUSE-GAN",
      category: "Satellite Imagery Super Resolution",
      highlights: [
        "Multi-View Modified GAN architecture for satellite imagery super resolution",
        "Integrates temporal data and structural priors for high-quality results",
        "Trained on WorldStrat dataset"
      ],
      content: `# MUSE GAN: A MULTIVIEW MODIFIED GAN ARCHITECTURE FOR SATELLITE IMAGERY SUPER RESOLUTION

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

\`\`\`
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
\`\`\`

## Architecture

### 1. Input Formulation (27-Channel Tensor)
The model processes a rich, high-dimensional input:
- **Temporal Data**: 8 Low-Resolution Sentinel-2 frames stacked channel-wise (8 frames × 3 RGB = 24 channels).
- **Structural Priors**: 3 single-channel feature maps extracted from a reference LR frame:
    - **Canny Edges**: For boundary detection.
    - **Local Binary Patterns (LBP)**: For texture analysis.
    - **Sobel Gradients**: For spatial transitions.
- **Total Input**: \`(B, 27, 160, 160)\`

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
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

### Training
To train the model, use the \`src/train.py\` script.
\`\`\`bash
python -m src.train --dataset_path /path/to/worldstrat --epochs 30 --batch_size 4
\`\`\`

## Future Work

- **Dynamic Temporal Fusion**: Implementing attention mechanisms (e.g., Transformers) to intelligently weight temporal frames based on quality (e.g., ignoring cloudy frames).
- **Progressive Growing**: Adopting progressive training strategies for better stability at higher resolutions.
- **Diffusion Models**: Exploring diffusion-based generation for potentially higher perceptual quality.

## References
Based on the project report "MUSE GAN: A MULTIVIEW MODIFIED GAN ARCHITECTURE FOR SATELLITE IMAGERY SUPER RESOLUTION" (Nov 2025).
`
    }
  ],
  skills: {
    programming: "Python, Java, SQL, C++, JavaScript",
    ml_data: "TensorFlow, PyTorch, Scikit-Learn, NumPy, Pandas, DeepStream",
    databases: "MySQL, PostgreSQL, MongoDB",
    cloud_infra: "AWS, S3, Lambda, API Gateway, Docker, Maven, Git",
    full_stack: "React, Node.js, Express.js, RESTful APIs"
  },
  awards: [
    "Member, Microsoft Innovations Club (2023–2025); Google Developer Student Club (2022–2023); Secretary, Bits N Bytes Club (2021–2022)",
    "Awards: 2nd Place Blitz(K)rieg Game Design (2019), Best Documentary – INTACH FilmIt (2018), 1st Place – Design Championship (2017)"
  ],
  resumeData: {
    header: {
      name: "Shaurya Kumar",
      location: "New Delhi, India",
      email: "emailofshauryak@gmail.com",
      phone: "+91 99711 53775",
      linkedin: "https://www.linkedin.com/in/shaurya-kumar-22262b236/",
      github: "https://github.com/ladsad"
    },
    education: [
      {
        institution: "Vellore Institute of Technology (VIT), Chennai",
        degree: "B.Tech. Computer Science (AI & ML)",
        dates: "Sept 2022 – July 2026",
        details: [
          "CGPA: 8.94/10.0",
          "Coursework: Data Structures, Database Systems, Java Programming, Software Engineering, Statistical Methods"
        ]
      }
    ],
    experience: [
      {
        role: "Software Engineer Intern",
        company: "eMudhra",
        dates: "June 2024 – July 2024",
        points: [
          "Designed and developed robust backend services using Java Server Pages and Apache Tomcat framework, implementing RESTful microservices for scalable enterprise applications",
          "Implemented Java Servlets to optimize application performance, achieving 10–100× improvement in data processing efficiency",
          "Utilized MySQL Clusters for rapid, reliable data retrieval, increasing access speeds by 3–5x."
        ]
      },
      {
        role: "Computer Vision Engineering Intern",
        company: "Bidaal",
        dates: "June 2025 – July 2025",
        points: [
          "Developed scalable data processing pipelines using DeepStream and Python, achieving 25% latency reduction and 40% deployment capability expansion",
          "Collaborated with cross-functional teams to deliver end-to-end analytical solutions and communicated technical insights to stakeholders"
        ]
      }
    ],
    projects: [
      {
        name: "Churn HTE: Causal Inference System",
        link: "https://github.com/ladsad/churn-hte",
        type: "Machine Learning / Data Science",
        points: [
          "Architected a causal inference system using Causal Forests and Doubly Robust Estimation to target persudable customers effectively",
          "Developed a FastAPI production service for real-time intervention scoring, achieving ~24% relative churn reduction in simulations",
          "Implemented complex causal estimators (EconML) to identify Heterogeneous Treatment Effects (CATE) across customer segments"
        ]
      },
      {
        name: "CodeWhisper: Intelligent Documentation Tool",
        link: "https://github.com/ladsad/codewhisper",
        type: "Developer Tools",
        points: [
          "Developed an AI-powered tool for auto-generating documentation and code quality analysis using CodeT5+ and QLoRA",
          "Built a VS Code extension for seamless IDE integration and a FastAPI/Streamlit backend for metrics visualization",
          "Achieved 36.65 BLEU and 62.17 ROUGE-L scores by fine-tuning CodeT5-small on CodeXGLUE dataset"
        ]
      },
      {
        name: "Mustard Archives: Scalable Web Analytics",
        link: "https://github.com/ladsad/Mustard-Archives",
        type: "Full-Stack Platform",
        points: [
          "Developed comprehensive full-stack web platform using React frontend and Express.js backend with RESTful API architecture",
          "Implemented robust MySQL database design with optimized schemas and complex queries, reducing data redundancy by 30%.",
          "Built a secure user authentication system using SHA256 encryption algorithm for improved security"
        ]
      },
      {
        name: "AWS Sentiment Analysis Platform",
        link: "https://github.com/ladsad/AWS-SentimentAnalysisRedit-Frontend",
        type: "Cloud Architecture",
        points: [
          "Developed cloud-based sentiment analysis system using AWS services including Lambda, API Gateway, and EC2 for scalable deployment",
          "Designed and implemented RESTful APIs using Amazon API Gateway to facilitate seamless communication between frontend and backend services",
          "Utilized AWS Comprehend for NLP processing and S3 for data storage, demonstrating proficiency with cloud platforms and distributed systems"
        ]
      },
      {
        name: "MicroSegNet Optimizer: ML Training and Automation",
        link: "https://github.com/ladsad/Modified-MicroSegNet",
        type: "Machine Learning Infrastructure",
        points: [
          "Built automated ML training pipelines with hyperparameter optimization, reducing training time by 40%",
          "Implemented data validation and cross-validation frameworks using TensorFlow and statistical analysis",
          "Deployed scalable model inference infrastructure with performance monitoring and quality assurance"
        ]
      },
      {
        name: "Attention-Enhanced Recurrent Highway Networks",
        link: "https://github.com/ladsad/Integrating-Attention-mechanisms-into-Recurrent-Highway-Networks-with-Grouped-Auxiliary-Memory",
        type: "NLP",
        points: [
          "Enhanced RHN model with attention mechanisms to improve long term dependency capture in sequence tasks, performed on the Penn TreeBank dataset",
          "Integrated grouped auxiliary memory for effective retention of contextual information in RNNs"
        ]
      },
      {
        name: "ArtResGAN: Distributed Training for Art Restoration",
        link: "https://github.com/ladsad/ArtResGAN",
        type: "Computer Vision",
        points: [
          "Implemented a hybrid U-Net+ResNet GAN model for restoring degraded artwork on WikiArt dataset",
          "Incorporated classical machine vision techniques to enhance texture and structural detail in restored images",
          "Achieved high fidelity restorations using adversarial, content and style loss functions"
        ]
      }
    ],
    technicalSkills: {
      programming: "Java, Python, SQL, C++, JavaScript",
      backend: "Express.js, RESTful APIs, Django",
      databases: "MySQL, PostgreSQL, MongoDB",
      toolsCloud: "AWS, Git, Maven, Docker",
      frontend: "React, Node.js",
      mlAi: "TensorFlow, PyTorch, Scikit-Learn, NumPy, Pandas"
    },
    awards: [
      "Member, Microsoft Innovations Club (2023–2025); Member, Google Developer Student Club (2022–2023); Secretary, Bits N Bytes Club (2021–2022)",
      "Awards: 2nd Place – Blitz(K)rieg Game Design (2019), Best Documentary – INTACH FilmIt (2018), 1st Place – Design Championship (2017)"
    ]
  }
};

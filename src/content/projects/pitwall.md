# Pitwall: F1 Race Prediction Analytics Platform

```
  ____  _ _                 _ _ 
 |  _ \(_) |_ __      ____ _| | |
 | |_) | | __|\ \ /\ / / _` | | |
 |  __/| | |_  \ V  V / (_| | | |
 |_|   |_|\__|  \_/\_/ \__,_|_|_|
```

> **Big Data Lakehouse & Deep Learning analytics platform for Formula 1 race predictions, powered by a PySpark Medallion pipeline and a 1D PatchTST-style Masked Autoencoder (MAE) in PyTorch.**

---

## 🎯 Overview & Problem Statement

Formula 1 telemetry represents one of the most high-frequency, complex sensor environments in sports analytics, with cars broadcasting 200 Hz streams across throttle, brake, RPM, gear, DRS, and speed. Traditional models fail to capture long-range temporal dependencies across entire race stints and often suffer from target leakage or memory exhaustion during feature engineering.

**Pitwall** is an end-to-end Big Data & Deep Learning analytics platform. It ingests 200 Hz telemetry across 70+ seasons using a **PySpark Medallion Lakehouse** (Bronze/Silver/Gold Parquet), trains a self-supervised **1D PatchTST-style Masked Autoencoder (MAE)** in **PyTorch**, achieves **0.996 top-3 prediction accuracy**, and visualizes predictions via a live **Next.js** dashboard on Vercel backed by **Supabase**.

---

## 🏗️ System Architecture & Data Flow

```
                      +-----------------------------+
                      |     FastF1 Telemetry Feed   |
                      |   (200 Hz Live/Historical)  |
                      +--------------+--------------+
                                     |
                         Raw JSON & Telemetry Logs
                                     v
                      +-----------------------------+
                      |   PySpark Medallion Lake    |
                      |                             |
                      |  +-----------------------+  |
                      |  | Bronze (Raw Parquet)  |  |
                      |  +-----------+-----------+  |
                      |              v              |
                      |  +-----------------------+  |
                      |  | Silver (Cleaned Data) |  |
                      |  +-----------+-----------+  |
                      |              v              |
                      |  +-----------------------+  |
                      |  | Gold (Feature Marts)  |  |
                      |  +-----------------------+  |
                      +--------------+--------------+
                                     |
                        16GB Engineered Sensor Arrays
                                     v
                      +-----------------------------+
                      |   PyTorch Model Pipelines   |
                      |                             |
                      |  +-----------------------+  |
                      |  | Random Forest Base    |  |
                      |  +-----------------------+  |
                      |  | 1D PatchTST MAE (75%  |  |
                      |  | Random Patch Masking) |  |
                      |  +-----------------------+  |
                      +--------------+--------------+
                                     |
                        Predictions & Occlusion Attribution
                                     v
                      +-----------------------------+
                      |   Supabase PostgreSQL DB    |
                      +--------------+--------------+
                                     |
                        Client Cached Data Stream
                                     v
                      +-----------------------------+
                      |   Next.js / Vercel Web App  |
                      | (Dual Model, TikZ Heatmap)  |
                      +-----------------------------+
```

---

## ⚡ Key Features & Engineering Innovations

- **Medallion Lakehouse on PySpark**: Scalable ETL architecture transforming over 16GB of raw telemetry into structured Bronze, Silver, and Gold Parquet tables with custom anomaly filtering for `NaN` sensor packets.
- **1D PatchTST Masked Autoencoder (MAE)**: Custom 6-block Transformer encoder in **PyTorch** pre-trained with 75% random patch masking over continuous multi-channel telemetry series, learning intrinsic vehicle dynamics without label leakage.
- **0.996 Top-3 Race Prediction Accuracy**: Outperforms classical baselines over 199 training epochs while overcoming Kaggle/Colab GPU memory constraints via per-epoch checkpointing.
- **Driver-Specific Occlusion Attribution**: Implemented sliding-window occlusion sensitivity analysis to isolate exact cornering and throttle behaviors contributing to race outcome probabilities.
- **Zero-Tear Next.js Frontend**: Deployed on **Vercel** with **Supabase PostgreSQL** backend, featuring client-side `useRef` caching to eliminate UI flickering during rapid driver/season comparisons.
- **Publication-Ready LaTeX TikZ Visualization**: Dynamically rendered architecture diagrams including a 1152x384 QKV attention projection weight heatmap directly extracted from the PyTorch model checkpoint.

---

## 📊 Benchmark Metrics & Performance

| Metric | Measured Value | Baseline / Notes |
| :--- | :--- | :--- |
| **Top-3 Finish Accuracy** | **0.996 (99.6%)** | Evaluated across 199 pre-training epochs |
| **Processed Telemetry Volume** | **16 GB Parquet** | 200 Hz feeds: Speed, Throttle, Brake, RPM, Gear, DRS |
| **Transformer Architecture** | **6 Encoder Blocks** | 75% random masking ratio with 1D Conv patch embedding |
| **Frontend Caching Latency** | **< 10ms (useRef)** | Instant state transitions with zero UI tearing |
| **Historical Data Coverage** | **70+ Seasons (1950–Present)**| Granular lap-by-lap telemetry & weather telemetry |

---

## 🛠️ Tech Stack & Badges

- **Big Data & Processing**: `PySpark`, `Apache Spark MLlib`, `Parquet`, `Databricks`
- **Deep Learning & ML**: `PyTorch`, `Masked Autoencoders (MAE)`, `Vision Transformers (ViT)`, `FastF1`, `Scikit-Learn`
- **Database & Storage**: `Supabase`, `PostgreSQL`
- **Frontend & Visualization**: `Next.js 14`, `React`, `Tailwind CSS`, `LaTeX TikZ`, `Chart.js`
- **Hosting & CI/CD**: `Vercel`, `GitHub Actions`

---

## 🔗 Repository & Live Deployment

- **GitHub Repository**: [https://github.com/ladsad/pitwall](https://github.com/ladsad/pitwall)
- **Live Platform**: [https://pitwall-f1-six.vercel.app/](https://pitwall-f1-six.vercel.app/)

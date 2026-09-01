# Confoundr: Causal Validity Linter & Diagnostic Platform

```
   ____             __                     _      
  / ___|___  _ __  / _| ___  _   _ _ __   __| |_ __ 
 | |   / _ \| '_ \| |_ / _ \| | | | '_ \ / _` | '__|
 | |__| (_) | | | |  _| (_) | |_| | | | | (_| | |   
  \____\___/|_| |_|_|  \___/ \__,_|_| |_|\__,_|_|   
```

> **Automated causal validity testing and ML dataset diagnostics with plugin-based statistical checks, asynchronous job execution, and LLM-powered fix explanations.**

---

## 🎯 Overview & Problem Statement

Standard machine learning evaluation metrics (like ROC-AUC, RMSE, and F1-score) frequently mask critical causal failure modes. Submitting datasets with unmeasured confounding, target leakage, or severe positivity violations yields models that achieve high offline accuracy but fail catastrophically in production interventions.

**Confoundr** is an open-source Python diagnostic library and asynchronous web platform designed to detect and triage causal validity violations in tabular datasets. It pairs rigorous statistical tests with an asynchronous **FastAPI + Redis** worker architecture, sandboxed execution, and an **LLM explainer layer** (Groq / LLaMA 3.1) that translates statistical failures into actionable engineering recommendations.

---

## 🏗️ System Architecture & Data Flow

```
                      +-----------------------------+
                      |   Data Scientist / ML Dev   |
                      |   (CLI or Next.js Web App)  |
                      +--------------+--------------+
                                     |
                          Upload CSV / Job Config
                                     v
                      +-----------------------------+
                      |    FastAPI Gateway API      |
                      |  (POST /api/v1/async-check) |
                      +--------------+--------------+
                                     |
                         Enqueue Diagnostic Task
                                     v
                      +-----------------------------+
                      |      Redis Job Queue        |
                      |     (ARQ / Redis Store)     |
                      +--------------+--------------+
                                     |
                          Fetch Job & Dataset
                                     v
                      +-----------------------------+
                      |  Sandboxed Docker Worker    |
                      |                             |
                      |  +-----------------------+  |
                      |  | Target Leakage Check  |  |
                      |  +-----------------------+  |
                      |  | Confounding Check     |  |
                      |  +-----------------------+  |
                      |  | Positivity Check      |  |
                      |  +-----------------------+  |
                      +--------------+--------------+
                                     |
                        Statistical Violations Vector
                                     v
                      +-----------------------------+
                      |     LLM Explainer Layer     |
                      |    (Groq / LLaMA 3.1 8B)    |
                      +--------------+--------------+
                                     |
                         Structured JSON Report
                                     v
                      +-----------------------------+
                      |   Supabase PostgreSQL DB    |
                      |  & Prometheus / Grafana     |
                      +-----------------------------+
```

---

## ⚡ Key Features & Engineering Innovations

- **Plugin-Based Causal Linter Core**: Standalone extensible Python engine evaluating datasets for:
  - **Target Leakage**: Detects post-treatment feature correlations and temporal ordering inversions.
  - **Unmeasured Confounding**: Conducts sensitivity analysis and Oster bounds on omitted variable bias.
  - **Positivity / Overlap**: Identifies extreme propensity divergence (<5% overlap) between treated and control cohorts.
- **Asynchronous Scalable Platform**: High-concurrency **FastAPI** backend orchestrating background validation jobs via a **Redis** queue with automatic retries and exponential backoff.
- **AI-Powered Explainer Engine**: Integrates **Groq / LLaMA 3.1** to automatically interpret complex statistical violation metrics and provide plain-language data remediation steps.
- **Sandboxed Multi-Tenant Execution**: Ephemeral Docker worker containers enforce strict CPU, memory, and timeout limits for untrusted user-submitted datasets.
- **Enterprise Observability & Persistence**: Real-time Prometheus metrics monitoring queue latency and job failure rates, persisting complete audit trails to **Supabase PostgreSQL**.

---

## 📊 Benchmark Metrics & Performance

| Metric | Measured Value | Description / Scope |
| :--- | :--- | :--- |
| **Standard Dataset Execution** | **< 1.2 seconds** | Full diagnostic run on 100k-row tabular datasets |
| **Positivity Overlap Detection** | **< 5% threshold** | Flags acute treatment assignment distribution disparity |
| **Worker Resource Isolation** | **1 CPU / 1GB RAM cap** | Strict sandboxed container execution constraints |
| **AI Explanation Latency** | **~600ms via Groq API** | Near-instant plain-language triage recommendations |
| **Observability Coverage** | **100% trace/metric sync** | Real-time Prometheus queue & duration monitoring |

---

## 🛠️ Tech Stack & Badges

- **Languages**: `Python 3.11`, `SQL`
- **Causal Inference & ML**: `DoWhy`, `EconML`, `Scikit-Learn`, `Pandas`, `NumPy`
- **Backend & APIs**: `FastAPI`, `Pydantic v2`, `ARQ / Redis Queue`, `Uvicorn`
- **Generative AI & LLMs**: `Groq Cloud API`, `Meta LLaMA 3.1 8B`, `Prompt Engineering`
- **Database & Storage**: `Supabase`, `PostgreSQL 16`, `Redis 7`
- **Frontend & UI**: `Next.js 14`, `Tailwind CSS`, `Lucide Icons`
- **Infrastructure & Monitoring**: `Docker`, `Prometheus`, `Grafana`, `GitHub Actions`

---

## 🔗 Repository & Deployment

- **GitHub Repository**: [https://github.com/ladsad/confoundr](https://github.com/ladsad/confoundr)
- **API Documentation**: Available at `/docs` (FastAPI Swagger UI) upon launching backend stack.

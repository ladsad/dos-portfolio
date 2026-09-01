# RiskShield: Real-Time Fraud Detection Platform

```
  ____  _     _    ____  _     _      _     _ 
 |  _ \(_)___| | _/ ___|| |__ (_) ___| | __| |
 | |_) | / __| |/ /\___ \| '_ \| |/ _ \ |/ _` |
 |  _ <| \__ \   <  ___) | | | | |  __/ | (_| |
 |_| \_\_|___/_|\_\|____/|_| |_|_|\___|_|\__,_|
```

> **Production-grade, event-driven streaming fraud detection engine engineered with Go, Python, Redpanda, and Cloudflare Workers AI.**

---

## 🎯 Overview & Problem Statement

Modern financial ecosystems require sub-millisecond fraud evaluation on high-velocity transaction streams. Traditional batch architectures introduce unacceptable detection latency, while pure cloud LLM inference introduces prohibitive latency spikes and API failure risks.

**RiskShield** solves this by implementing a 4-stage microservice streaming pipeline operating over **Redpanda** (Kafka-compatible message broker). It combines low-latency feature enrichment in **Redis**, an intelligent dual-path ML scoring engine with circuit-breaker protection (Cloudflare Workers AI + local **scikit-learn IsolationForest** fallback), and a deterministic decision engine persisting to **PostgreSQL**.

---

## 🏗️ System Architecture & Data Flow

```
                      +-----------------------------+
                      |   HTTP / Synthetic Client   |
                      |   (10 TPS Fraud Generator)  |
                      +--------------+--------------+
                                     |
                           POST /api/v1/transactions
                                     v
                      +-----------------------------+
                      |     Ingestion Service       |
                      |     (Go / Redis Dedup)      |
                      +--------------+--------------+
                                     |
                         Kafka: 'transactions.raw'
                                     v
                      +-----------------------------+
                      |     Enrichment Service      |
                      |  (Python / Redis Velocity)  |
                      +--------------+--------------+
                                     |
                       Kafka: 'transactions.enriched'
                                     v
                      +-----------------------------+
                      |       Scoring Service       |
                      |    (Dual-Path ML Engine)    |
                      +-------+-------------+-------+
                              |             |
           Primary (>99.9%)   |             | Fallback (<5ms)
                              v             v
                    +----------------+  +----------------+
                    | Cloudflare AI  |  | IsolationForest|
                    | (LLaMA-3.2-1B) |  | Local Scikit   |
                    +----------------+  +----------------+
                              \             /
                               v           v
                        Kafka: 'transactions.scored'
                                     |
                                     v
                      +-----------------------------+
                      |      Decision Service       |
                      |   (Rules Engine / Ledger)   |
                      +--------------+--------------+
                                     |
                     Persist Decision & Audit Log
                                     v
                      +-----------------------------+
                      |   PostgreSQL / REST APIs    |
                      |  ALLOW / CHALLENGE / DENY   |
                      +-----------------------------+
```

---

## ⚡ Key Features & Engineering Innovations

- **4-Stage Microservice Stream Pipeline**: Modular, decoupled services (Ingestion, Enrichment, Scoring, Decision) communicating via Redpanda Kafka topics.
- **Dual-Path ML Scoring & Circuit Breaker**: Primary inference via **Cloudflare Workers AI** (`@cf/meta/llama-3.2-1b-instruct`) with an automatic sub-5ms fallback to local **scikit-learn IsolationForest**. The circuit breaker automatically trips after 5 consecutive failures and resets after 60s.
- **Real-Time Feature Enrichment**: Computes sliding-window transaction velocities and Haversine geospatial distances against historical cardholder patterns in **Redis**.
- **Deterministic Decision Engine & Ledger**: Evaluates scored transactions against threshold rules (`ALLOW < 0.35`, `CHALLENGE 0.35–0.74`, `DENY >= 0.75`) and supports analyst override REST APIs with complete audit trails.
- **Infrastructure as Code & CI/CD**: Cloud infrastructure provisioned via **Terraform**, with automated GitHub Actions CI/CD pipelines enforcing a **≥95% test coverage gate**.

---

## 📊 Benchmark Metrics & Performance

| Metric | Measured Value | Target / Baseline |
| :--- | :--- | :--- |
| **Local ML Fallback Latency** | **< 5ms** | < 15ms SLA |
| **Circuit Breaker Auto-Reset** | **60 seconds** | 5 consecutive failure threshold |
| **Synthetic Transaction Generator** | **10 TPS (8% anomaly ratio)** | Configurable load simulation |
| **Cloudflare Workers AI Optimization**| **10,000 neurons/day budget** | Cost-effective edge inference |
| **Automated Test Coverage Gate** | **≥ 95% combined (Go + Python)** | Production CI/CD gate |

---

## 🛠️ Tech Stack & Badges

- **Languages**: `Go (Golang)`, `Python 3.11`, `SQL`
- **Streaming & Messaging**: `Redpanda (Kafka-compatible)`, `Kafka Consumer/Producer APIs`
- **Storage & Caching**: `PostgreSQL 16`, `Redis 7 (Sliding Windows, Dedup)`
- **Machine Learning & AI**: `Cloudflare Workers AI (LLaMA-3.2-1B)`, `Scikit-Learn (IsolationForest)`
- **Frameworks & Web**: `FastAPI`, `Go HTTP Standard Library`, `Uvicorn`
- **DevOps & Observability**: `Terraform (IaC)`, `Docker & Docker Compose`, `Prometheus`, `Grafana`, `GitHub Actions CI/CD`

---

## 🔗 Repository & Deployment

- **GitHub Repository**: [https://github.com/ladsad/RiskShield](https://github.com/ladsad/RiskShield)
- **Local Deployment**: Docker Compose stack (`docker compose up --build`) exposing Ingestion (`:8080`), Decision API (`:8083`), Prometheus (`:9090`), Grafana (`:3000`).

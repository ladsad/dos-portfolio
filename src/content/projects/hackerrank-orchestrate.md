# HackerRank Orchestrate: Message Notification Router

```
   ___           _               _             _       
  / _ \ _ __ ___| |__   ___  ___| |_ _ __ __ _| |_ ___ 
 | | | | '__/ __| '_ \ / _ \/ __| __| '__/ _` | __/ _ \
 | |_| | | | (__| | | |  __/\__ \ |_| | | (_| | ||  __/
  \___/|_|  \___|_| |_|\___||___/\__|_|  \__,_|\__\___|
```

> **AI-powered multimodal message routing pipeline classifying WhatsApp streams into 'notify', 'digest', or 'mute' with deterministic prompt injection defenses — securing Top 200 globally.**

---

## 🎯 Overview & Problem Statement

Users receive hundreds of high-frequency group notifications, OTPs, promotional alerts, media attachments, and voice notes on messaging platforms daily. Traditional rule-based filters fail on nuanced conversational context, while naive LLM classifiers are vulnerable to prompt injection attacks, media evasion, and costly inference overhead.

**HackerRank Orchestrate** is a multimodal AI notification routing engine engineered for the HackerRank Orchestrate Hackathon (August 2026), achieving a **Top 200 global ranking** and **93.3% benchmark accuracy**. It processes multimodal message streams (text, OCR extracted from images, and audio transcribed via FFmpeg), combines them with 10+ relational context tables, queries **Meta LLaMA 3.1 8B via Cloudflare Workers AI**, and applies strict post-LLM security overrides.

---

## 🏗️ System Architecture & Data Flow

```
                      +-----------------------------+
                      | Incoming WhatsApp Message   |
                      |   (Text, Image, Voice Note) |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Media Ingestion & Extraction|
                      |  - Tesseract OCR (Images)   |
                      |  - FFmpeg + Whisper (Audio) |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Prompt Framing & Security   |
                      |  - XML <untrusted_message>  |
                      |  - Relational Context Join  |
                      +--------------+--------------+
                                     |
                         Sanitized Prompt Payload
                                     v
                      +-----------------------------+
                      | Cloudflare Workers AI       |
                      | (@cf/meta/llama-3.1-8b-inst)|
                      +--------------+--------------+
                                     |
                         Raw Routing Decision JSON
                                     v
                      +-----------------------------+
                      | Post-LLM Security & Overrides|
                      |  - Regex Scam / Phishing    |
                      |  - Muted Group Enforcement  |
                      |  - Unopened Promo Rule      |
                      +--------------+--------------+
                                     |
                         Final Structured Decision
                                     v
                      +-----------------------------+
                      |  NOTIFY  /  DIGEST  /  MUTE |
                      +-----------------------------+
```

---

## ⚡ Key Features & Engineering Innovations

- **Multimodal Context Ingestion**: Extracts text from images via **Tesseract OCR** and converts voice notes via **FFmpeg**, feeding unified context into the AI routing engine.
- **Strict Prompt Injection Shielding**: Isolates untrusted user content within XML `<untrusted_message>` tags and runs proactive regex injection scanners to prevent deceptive system instruction overrides.
- **Deterministic Post-LLM Overrides**: Enforces zero-tolerance safety rules (flagging OTP/PIN/password requests as scams, auto-muting blacklisted groups, and muting promotional broadcasts if user historical open rates are zero).
- **Edge LLM Inference**: Deployed on **Cloudflare Workers AI** using `meta/llama-3.1-8b-instruct` with strict JSON schema parsing and zero unhandled fallbacks.
- **Relational Context Synthesis**: Integrates historical user behavior, sender relationship graphs, and group activity levels across 10+ relational database tables.

---

## 📊 Benchmark Metrics & Performance

| Metric | Measured Value | Scope / Significance |
| :--- | :--- | :--- |
| **Classification Accuracy** | **93.3%** | Evaluated on full multimodal benchmark test set |
| **Global Placement** | **Top 200 Worldwide** | HackerRank Orchestrate Hackathon (August 2026) |
| **Batch Determinism** | **100% (110 / 110 messages)**| Zero unhandled exceptions or invalid schemas |
| **Edge Inference Latency**| **Sub-second response** | Cloudflare Workers AI edge processing |
| **Prompt Injection Defense**| **0% bypass rate** | XML encapsulation + regex rule backstops |

---

## 🛠️ Tech Stack & Badges

- **Languages & Core**: `Python 3.11`, `JSON Schema Validation`
- **Multimodal AI & LLM**: `Cloudflare Workers AI`, `Meta LLaMA 3.1 8B`, `Tesseract OCR`, `FFmpeg`
- **Security & Safety**: `Prompt Injection Mitigation`, `XML Tag Framing`, `Regex Scam Detection`
- **Data & Context**: `Relational Metadata Joins`, `Historical Engagement Modeling`

---

## 🔗 Repository & Documentation

- **GitHub Repository**: [https://github.com/ladsad/hackerrank-orchestra](https://github.com/ladsad/hackerrank-orchestra)
- **Competition**: HackerRank Orchestrate Hackathon (August 2026)

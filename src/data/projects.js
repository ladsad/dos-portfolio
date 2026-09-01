import React from 'react';
import content_0 from '../content/projects/churn-hte-causal-ml.md?raw';
import content_1 from '../content/projects/codewhisper.md?raw';
import content_2 from '../content/projects/microsegnet-optimizer.md?raw';
import content_3 from '../content/projects/attention-enhanced-rhn.md?raw';
import content_4 from '../content/projects/mustard-archives.md?raw';
import content_5 from '../content/projects/aws-sentiment-analysis.md?raw';
import content_6 from '../content/projects/artresgan.md?raw';
import content_7 from '../content/projects/muse-gan.md?raw';
import content_riskshield from '../content/projects/riskshield.md?raw';
import content_kestrel from '../content/projects/kestrel.md?raw';
import content_confoundr from '../content/projects/confoundr.md?raw';
import content_pitwall from '../content/projects/pitwall.md?raw';
import content_finflow from '../content/projects/finflow.md?raw';
import content_hackerrank_orchestrate from '../content/projects/hackerrank-orchestrate.md?raw';

export const projects = [
  {
    name: "Churn HTE: Causal ML",
    link: "https://github.com/ladsad/churn-hte",
    category: "Causal ML & Production AI",
    highlights: ["Beyond simple churn prediction: Identifies *who* to target using Heterogeneous Treatment Effects (HTE)","Implements Doubly Robust Estimation and Causal Forests for unbiased causal inference","Productionized FastAPI endpoint with real-time recommendations, driving a 24% relative churn reduction in A/B tests."],
    content: content_0,
  },
  {
    name: "CodeWhisper",
    link: "https://github.com/ladsad/codewhisper",
    category: "Developer Tools & AI",
    highlights: ["Full-stack Agentic AI coding assistant integrating a fine-tuned CodeT5+ LLM with a highly responsive web interface and FastAPI backend","Deployed the service via CI/CD pipelines on AWS, improving developer productivity and documentation acceptance by +12%","Fine-tuned CodeT5-small on CodeXGLUE (Python/Java) using QLoRA for efficient model training"],
    content: content_1,
  },
  {
    name: "MicroSegNet Optimizer",
    link: "https://github.com/ladsad/Modified-MicroSegNet",
    category: "ML Pipeline Engineering",
    highlights: ["Built automated ML training pipelines with hyperparameter tuning, reducing model training time by 40%","Engineered validation and cross-validation systems using TensorFlow and advanced statistics for model integrity"],
    content: content_2,
  },
  {
    name: "Attention-Enhanced RHN",
    link: "https://github.com/ladsad/Integrating-Attention-mechanisms-into-Recurrent-Highway-Networks-with-Grouped-Auxiliary-Memory",
    category: "NLP Architectures",
    highlights: ["Improved sequence modeling by integrating attention mechanisms into RHNs, tested on Penn TreeBank data","Designed global auxiliary memory for effective retention of contextual info—boosting NLP model performance","Working Paper: Integrating Attention mechanisms into Recurrent Highway Networks with Grouped Auxiliary Memory"],
    content: content_3,
  },
  {
    name: "Mustard Archives",
    link: "https://github.com/ladsad/Mustard-Archives",
    category: "Full Stack Consultancy Platform",
    highlights: ["Developed a centralized consultancy platform connecting clients with skilled professionals","Implemented a robust MySQL database to streamline service delivery and reduce data redundancy","Built a responsive React frontend and Express/Node.js backend for efficient project management"],
    content: content_4,
  },
  {
    name: "AWS Sentiment Analysis",
    link: "https://github.com/ladsad/AWS-SentimentAnalysisRedit-Frontend",
    category: "Cloud AI Architecture",
    highlights: ["Architected scalable cloud-native NLP solutions leveraging AWS Lambda, EC2, and API Gateway","Designed RESTful APIs and data communication strategies for seamless client-server interaction","Utilized AWS Comprehend for text classification and analytics, S3 for robust cloud data warehousing"],
    content: content_5,
  },
  {
    name: "ArtResGAN",
    link: "https://github.com/ladsad/ArtResGAN",
    category: "Vision & GAN Systems",
    highlights: ["Engineered hybrid U-Net plus ResNet GAN architectures for restoring art images (WikiArt dataset)","Achieved high-fidelity results via adversarial, content, and style loss optimization","Working Paper: ArtResGAN: A GAN-Based Approach for Image Restoration and Style Preservation"],
    content: content_6,
  },
  {
    name: "MUSE-GAN",
    link: "https://github.com/ladsad/MUSE-GAN",
    category: "Satellite Imagery Super Resolution",
    highlights: ["Multi-View Modified GAN architecture for satellite imagery super resolution","Integrates temporal data and structural priors for high-quality results","Trained on WorldStrat dataset"],
    content: content_7,
  },
  {
    name: "RiskShield",
    link: "https://github.com/ladsad/RiskShield",
    category: "Stream Processing & ML Infrastructure",
    highlights: [
      "High-throughput, event-driven fraud detection platform in Go and Python across 4 microservices via Redpanda",
      "Dual-path ML scoring: Cloudflare Workers AI with local scikit-learn IsolationForest fallback (<5ms latency)",
      "Sliding-window velocity & Haversine geo-distance enrichment with Redis, configurable rules, and PostgreSQL ledger"
    ],
    content: content_riskshield,
  },
  {
    name: "Kestrel",
    link: "https://github.com/ladsad/kestrel",
    category: "Distributed Systems & Storage Engines",
    highlights: [
      "Distributed, fault-tolerant key-value store built in Go with a RESP2 TCP server compatible with redis-cli",
      "Strict AOF durability (always/everysec/no) and background snapshotting (367k writes replayed in ~318ms)",
      "Raft consensus leader failover in ~1.5s (VSR ~1.1s) and horizontal sharding via consistent hashing proxy"
    ],
    content: content_kestrel,
  },
  {
    name: "Confoundr",
    link: "https://github.com/ladsad/confoundr",
    category: "Causal Inference & ML Diagnostics",
    highlights: [
      "Causal validity linter for ML pipelines evaluating target leakage, unmeasured confounding, and positivity",
      "Scalable, multi-tenant diagnostic platform built with FastAPI, Redis job queue, and sandboxed Docker workers",
      "LLM-powered explainer layer (Groq / LLaMA 3.1) providing plain-language causal violation triage and fixes"
    ],
    content: content_confoundr,
  },
  {
    name: "Pitwall: F1 Race Prediction Platform",
    link: "https://github.com/ladsad/pitwall",
    category: "Big Data Engineering & Deep Learning",
    highlights: [
      "End-to-end Medallion data lakehouse (Bronze/Silver/Gold Parquet) in PySpark ingesting 200 Hz telemetry",
      "1D PatchTST-style Masked Autoencoder (MAE) in PyTorch achieving 0.996 top-3 prediction accuracy over 199 epochs",
      "Live Next.js dashboard on Vercel backed by Supabase with client-side caching and dynamic occlusion sensitivity"
    ],
    content: content_pitwall,
  },
  {
    name: "FinFlow",
    link: "https://github.com/ladsad/FinFlow",
    category: "Distributed Systems & FinTech",
    highlights: [
      "High-throughput payment processing engine in Java 21 & Spring Boot 3.3 handling 10k+ concurrent transactions",
      "Event-driven messaging via Apache Kafka (KRaft mode) partitioned by accountId with sliding-window rate limiting",
      "End-to-end OpenTelemetry / Jaeger distributed tracing and ELK centralized logging, reducing MTTR by 40%"
    ],
    content: content_finflow,
  },
  {
    name: "HackerRank Orchestrate: Message Notification Router",
    link: "https://github.com/ladsad/hackerrank-orchestra",
    category: "Agentic AI & Multimodal Systems",
    highlights: [
      "Multimodal AI routing pipeline in Python classifying WhatsApp streams into notify/digest/mute with 93.3% accuracy",
      "Top 200 global placement in HackerRank Orchestrate Hackathon with Cloudflare Workers AI (LLaMA 3.1 8B)",
      "Deterministic prompt injection defense with XML tags and regex scam/phishing safety backstops"
    ],
    content: content_hackerrank_orchestrate,
  },
];


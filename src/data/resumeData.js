export const resumeData = {
  "header": {
    "name": "Shaurya Kumar",
    "location": "New Delhi, India",
    "email": "emailofshauryak@gmail.com",
    "phone": "+91 9971153775",
    "linkedin": "https://www.linkedin.com/in/shaurya-kumar-22262b236",
    "github": "https://github.com/ladsad"
  },
  "education": [
    {
      "institution": "Vellore Institute of Technology (VIT), Chennai",
      "degree": "B.Tech. Computer Science (Artificial Intelligence & Machine Learning)",
      "dates": "Sept 2022 – July 2026",
      "details": [
        "CGPA: 8.93 / 10.0",
        "Coursework: Statistics, Machine/Deep Learning, NLP, Database Systems, Software Engineering, Web Development"
      ]
    }
  ],
  "experience": [
    {
      "role": "Data Engineer Intern",
      "company": "Data Mavericks",
      "dates": "Feb 2026 – May 2026",
      "points": [
        "Designed and implemented scalable ETL/ELT pipelines across heterogeneous OLTP sources using AWS Glue and Kinesis; integrated CDC streams, SCD Type 1/2 dimensional models, and schema validation into Snowflake, maintaining SLA adherence above 98% with full audit lineage.",
        "Operated and optimized internal data warehouse environments through clustering key tuning, materialized view strategies, and query plan analysis — improving analytical query performance by up to 50% and reducing compute costs.",
        "Re-architected an LLM-powered NL-to-SQL analytics accelerator (AWS Bedrock, Secrets Manager, Redis, Layered Lambda Architecture) serving enterprise Snowflake warehouses; slashed cold-start latency by ~40% and eliminated credential rotation incidents."
      ]
    },
    {
      "role": "Computer Vision & Data Engineering Intern",
      "company": "Bidaal",
      "dates": "June 2025 – July 2025",
      "points": [
        "Deployed YOLO-based safety detection pipeline on construction site edge devices via NVIDIA DeepStream; monitored PPE compliance across multiple sites, reducing manual safety inspections by 60%.",
        "Designed and operated event-driven streaming data pipelines for real-time ML inference workloads; implemented automated data quality validation frameworks, reducing bad-data incidents by ~80% and raising downstream model accuracy by 6pp.",
        "Optimized edge inference latency (25% reduction) and edge scalability (40% improvement) through model quantization, batching, and distributed stream processing."
      ]
    },
    {
      "role": "Software Engineer Intern (Data Infrastructure)",
      "company": "eMudhra",
      "dates": "June 2024 – July 2024",
      "points": [
        "Engineered batch ETL workflows and RESTful data services (Java Servlets, Apache Tomcat, MySQL clustering); elevated analytics throughput by 10–100× for enterprise reporting workloads.",
        "Restructured relational warehouse schemas and rewrote SQL query execution plans, reducing average query latency by 30% and enabling real-time self-service analytics for 50+ cross-functional business stakeholders."
      ]
    }
  ],
  "projects": [
    {
      "name": "Churn HTE: Causal Inference System",
      "link": "https://github.com/ladsad/churn-hte",
      "type": "Machine Learning / Data Science",
      "points": [
        "Architected a causal inference system using Causal Forests and Doubly Robust Estimation to target persudable customers effectively",
        "Developed a FastAPI production service for real-time intervention scoring, achieving ~24% relative churn reduction in simulations",
        "Implemented complex causal estimators (EconML) to identify Heterogeneous Treatment Effects (CATE) across customer segments"
      ]
    },
    {
      "name": "CodeWhisper: Intelligent Documentation Tool",
      "link": "https://github.com/ladsad/codewhisper",
      "type": "Developer Tools",
      "points": [
        "Developed an AI-powered tool for auto-generating documentation and code quality analysis using CodeT5+ and QLoRA",
        "Built a VS Code extension for seamless IDE integration and a FastAPI/Streamlit backend for metrics visualization",
        "Achieved 36.65 BLEU and 62.17 ROUGE-L scores by fine-tuning CodeT5-small on CodeXGLUE dataset"
      ]
    },
    {
      "name": "Mustard Archives: Distributed Analytics Data Platform",
      "link": "https://github.com/ladsad/Mustard-Archives",
      "type": "Big Data / Data Lake",
      "points": [
        "Built a scalable data lake platform on Amazon S3 with a PySpark / SparkSQL processing layer",
        "Migrated 100M+ session records from Pandas to Apache Spark using broadcast joins and Parquet partitioning, reducing query latency 2.1× and storage 4× (3.2 GB CSV → 0.8 GB Parquet)",
        "Orchestrated end-to-end ETL pipelines with Apache Airflow including automated data quality checks and Kimball-style star schema data marts"
      ]
    },
    {
      "name": "AWS Sentiment Analysis Platform",
      "link": "https://github.com/ladsad/AWS-SentimentAnalysisRedit-Frontend",
      "type": "Cloud Architecture",
      "points": [
        "Developed cloud-based sentiment analysis system using AWS services including Lambda, API Gateway, and EC2 for scalable deployment",
        "Designed and implemented RESTful APIs using Amazon API Gateway to facilitate seamless communication between frontend and backend services",
        "Utilized AWS Comprehend for NLP processing and S3 for data storage, demonstrating proficiency with cloud platforms and distributed systems"
      ]
    },
    {
      "name": "MicroSegNet Optimizer: ML Training and Automation",
      "link": "https://github.com/ladsad/Modified-MicroSegNet",
      "type": "Machine Learning Infrastructure",
      "points": [
        "Built automated ML training pipelines with hyperparameter optimization, reducing training time by 40%",
        "Implemented data validation and cross-validation frameworks using TensorFlow and statistical analysis",
        "Deployed scalable model inference infrastructure with performance monitoring and quality assurance"
      ]
    },
    {
      "name": "Attention-Enhanced Recurrent Highway Networks",
      "link": "https://github.com/ladsad/Integrating-Attention-mechanisms-into-Recurrent-Highway-Networks-with-Grouped-Auxiliary-Memory",
      "type": "NLP",
      "points": [
        "Enhanced RHN model with attention mechanisms to improve long term dependency capture in sequence tasks, performed on the Penn TreeBank dataset",
        "Integrated grouped auxiliary memory for effective retention of contextual information in RNNs"
      ]
    },
    {
      "name": "ArtResGAN: Distributed Training for Art Restoration",
      "link": "https://github.com/ladsad/ArtResGAN",
      "type": "Computer Vision",
      "points": [
        "Implemented a hybrid U-Net+ResNet GAN model for restoring degraded artwork on WikiArt dataset",
        "Incorporated classical machine vision techniques to enhance texture and structural detail in restored images",
        "Achieved high fidelity restorations using adversarial, content and style loss functions"
      ]
    },
    {
      "name": "RiskShield: Real-Time Fraud Detection Platform",
      "link": "https://github.com/ladsad/RiskShield",
      "type": "Stream Processing & ML Infrastructure",
      "points": [
        "Architected an event-driven fraud detection platform in Go and Python across four microservices processing streams end-to-end via Redpanda",
        "Engineered dual-path ML scoring: primary Cloudflare Workers AI with local scikit-learn IsolationForest circuit breaker fallback (<5ms latency)",
        "Built sliding-window velocity & Haversine geo-distance enrichment with Redis and configurable rules engine with PostgreSQL audit ledger"
      ]
    },
    {
      "name": "Kestrel: Distributed Key-Value Store",
      "link": "https://github.com/ladsad/kestrel",
      "type": "Distributed Systems & Storage Engines",
      "points": [
        "Built a distributed, fault-tolerant key-value store from scratch in Go, implementing a RESP2 TCP server compatible with redis-cli",
        "Engineered strict AOF durability (always/everysec/no) and background snapshotting, replaying 367,000 writes in ~318ms with zero data loss",
        "Implemented Raft consensus for leader election (~1.5s failover) and consistent hashing proxy for horizontal sharding"
      ]
    },
    {
      "name": "Confoundr: Causal Validity Linter & Diagnostic Platform",
      "link": "https://github.com/ladsad/confoundr",
      "type": "Causal Inference & ML Diagnostics",
      "points": [
        "Architected an open-source Python library for causal validity linting (target leakage, unmeasured confounding, positivity violations)",
        "Engineered a scalable multi-tenant platform with FastAPI, Redis-backed job queue, and sandboxed Docker worker execution",
        "Integrated an LLM-powered explainer layer (Groq / LLaMA 3.1) to translate statistical failures into plain-language triage and fixes"
      ]
    },
    {
      "name": "Pitwall: F1 Race Prediction Analytics Platform",
      "link": "https://github.com/ladsad/pitwall",
      "type": "Big Data Engineering & Deep Learning",
      "points": [
        "Architected Medallion data lakehouse (Bronze/Silver/Gold Parquet) using PySpark over 16GB telemetry dataset across 70+ seasons",
        "Pre-trained 1D PatchTST-style Masked Autoencoder (MAE) in PyTorch achieving 0.996 top-3 accuracy over 199 epochs",
        "Deployed Next.js dashboard on Vercel backed by Supabase PostgreSQL with client-side caching and dynamic occlusion sensitivity"
      ]
    },
    {
      "name": "FinFlow: Distributed Payment Processing System",
      "link": "https://github.com/ladsad/FinFlow",
      "type": "Distributed Systems & FinTech",
      "points": [
        "Architected high-throughput payment processing engine in Java 21 and Spring Boot 3.3 handling 10k+ concurrent transactions via Kafka KRaft",
        "Built ledger validation and sliding-window rate limiting (max 5 tx/60s) with PostgreSQL persistence and idempotency controls",
        "Implemented OpenTelemetry / Jaeger distributed tracing and ELK centralized logging, reducing MTTR for API failures by 40%"
      ]
    },
    {
      "name": "HackerRank Orchestrate: Message Notification Router",
      "link": "https://github.com/ladsad/hackerrank-orchestra",
      "type": "Agentic AI & Multimodal Systems",
      "points": [
        "Architected multimodal AI message routing pipeline in Python to classify WhatsApp streams into notify/digest/mute with 93.3% accuracy",
        "Secured Top 200 finish globally using Cloudflare Workers AI (LLaMA 3.1 8B) with Tesseract OCR and FFmpeg transcription",
        "Implemented XML prompt encapsulation and deterministic post-LLM regex scam backstops preventing prompt injection attacks"
      ]
    }
  ],
  "technicalSkills": {
    "programming": "Python, Java, Go, SQL, SparkSQL, JavaScript, TypeScript, C++, PL/SQL",
    "backend": "FastAPI, Spring Boot, Java Servlets, Node.js, RESTful APIs, WebSockets",
    "databases": "Snowflake (SnowPro Core), PostgreSQL, MySQL, Redis, Supabase, Pinecone",
    "toolsCloud": "AWS (Glue, Kinesis, Bedrock, Secrets Manager, Lambda, S3, EC2, MSK), Terraform, Docker, Kubernetes, CI/CD, Prometheus, Grafana",
    "frontend": "React, Next.js, HTML, CSS, Streamlit, Gradio",
    "mlAi": "PyTorch, PySpark, TensorFlow, Scikit-Learn, LightGBM, HuggingFace Transformers, LangGraph, OpenAI API, RAG, QLoRA Fine-Tuning, YOLO, Causal Inference"
  },
  "awards": [
    "Certifications: SnowPro Core Certified (Snowflake, 2026), SPN Gen AI Foundation — Snowflake Partner Network",
    "Awards: Top 200 — HackerRank Orchestrate (August 2026), Top 500 — Amazon ML Challenge (2025), First Place — NASSCOM Design Championships: Game Design (2018)"
  ]
};

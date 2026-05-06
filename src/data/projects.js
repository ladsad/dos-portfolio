import React from 'react';
import content_0 from '../content/projects/churn-hte-causal-ml.md?raw';
import content_1 from '../content/projects/codewhisper.md?raw';
import content_2 from '../content/projects/microsegnet-optimizer.md?raw';
import content_3 from '../content/projects/attention-enhanced-rhn.md?raw';
import content_4 from '../content/projects/mustard-archives.md?raw';
import content_5 from '../content/projects/aws-sentiment-analysis.md?raw';
import content_6 from '../content/projects/artresgan.md?raw';
import content_7 from '../content/projects/muse-gan.md?raw';

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
];

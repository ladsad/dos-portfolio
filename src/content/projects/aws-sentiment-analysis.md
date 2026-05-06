
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

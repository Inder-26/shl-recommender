---
title: SHL Assessment Recommender
emoji: 🎯
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# SHL Assessment Recommendation System

An intelligent system to analyze job descriptions and recommend relevant SHL assessments.

## 🚀 Deployment
This project is live on Hugging Face Spaces.
- **URL**: [https://huggingface.co/spaces/inderjeet/shl-recommender](https://huggingface.co/spaces/inderjeet/shl-recommender)

## 🛠️ Features
- **Query Analysis**: Uses Groq (Llama 3.3 70B) for skill extraction.
- **Semantic Matching**: SentenceTransformers + ChromaDB for assessment retrieval.
- **Type Balancing**: Automatically balances Knowledge (K) and Personality (P) recommendations.

## 🏗️ Architecture
The system follows a hybrid semantic-lexical retrieval pipeline.


## 💻 Tech Stack
- **Backend**: Flask
- **LLM**: Groq
- **Vector DB**: ChromaDB
- **Docker**: For consistent deployment
---
title: SHL Assessment Recommender
emoji: 🎯
colorFrom: blue
colorTo: teal
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# SHL Assessment Recommendation System

An intelligent recommendation system that analyzes job descriptions using **LLM-powered query understanding** and **semantic search** to recommend relevant SHL assessments.

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/inderjeet/shl-recommender)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/Inder-26/shl-recommender)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🌐 Live Demo

| Platform | URL |
|----------|-----|
| **Web App** | [https://inderjeet-shl-recommender.hf.space](https://inderjeet-shl-recommender.hf.space) |
| **API Endpoint** | [https://inderjeet-shl-recommender.hf.space/recommend](https://inderjeet-shl-recommender.hf.space/recommend) |
| **Health Check** | [https://inderjeet-shl-recommender.hf.space/health](https://inderjeet-shl-recommender.hf.space/health) |


![Web App - Home Page](https://raw.githubusercontent.com/Inder-26/shl-recommender/main/images/home-page.png)
*Screenshot: Web app home page (hosted on Hugging Face Spaces).* 

---

## 📋 Problem Statement

Recruiters and hiring managers struggle to find the right SHL assessments for their job roles. The current keyword-based search is slow and inefficient. This project builds an **intelligent recommendation system** that:

- Takes **natural language queries**, **job descriptions**, or **URLs**
- Returns **5-10 relevant SHL Individual Test Solutions**
- Balances **Knowledge (K)** and **Personality (P)** type assessments

---

## 🏗️ Architecture
![System overview](https://raw.githubusercontent.com/Inder-26/shl-recommender/main/images/system-overview.png)
*Figure: System overview showing data flow from user input to recommendations.*

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **LLM-Powered Analysis** | Uses Groq (Llama 3.3 70B) for intelligent skill extraction |
| **Semantic Search** | SentenceTransformers + ChromaDB for accurate assessment matching |
| **URL Support** | Automatically extracts job descriptions from URLs |
| **Balanced Results** | Interleaves Knowledge (K) and Personality (P) assessments |
| **Duration Filtering** | Respects time constraints mentioned in queries |
| **Fallback Mechanism** | Keyword-based analysis if LLM is unavailable |
| **REST API** | JSON API for programmatic access |
| **Web Interface** | User-friendly Bootstrap 5 frontend |

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Mean Recall@10** | 17.78% |
| **Total Assessments** | 377 |
| **Training Queries** | 10 |
| **Test Queries** | 9 |

### Per-Query Performance

| Query Type | Recall@10 |
|------------|-----------|
| Java Developers | 60% |
| Content Writer / English | 40% |
| Data Analyst | 30% |
| Radio Station JD | 20% |
| ICICI Bank Admin | 17% |
| Sales Graduates | 11% |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | Flask 3.0 |
| **LLM** | Groq (Llama 3.3 70B) |
| **Embeddings** | SentenceTransformers (all-MiniLM-L6-v2) |
| **Vector Database** | ChromaDB |
| **Frontend** | Bootstrap 5, Jinja2 |
| **Deployment** | Docker, Hugging Face Spaces |

---

## 🔌 API Reference

#### Health Check

**Request**

```
GET /health
```

**Response**

```json
{
  "status": "healthy"
}
```

---

#### Get Recommendations (POST)

**Request**

```
POST /recommend
Content-Type: application/json

{
  "query": "Java developer with team collaboration skills"
}
```

**Response**

```json
{
  "recommended_assessments": [
    {
      "name": "Core Java (Entry Level)",
      "url": "https://www.shl.com/solutions/products/product-catalog/view/core-java-entry-level-new/",
      "description": "Assessment for evaluating Java programming skills",
      "test_type": ["Knowledge & Skills"],
      "duration": 30,
      "remote_support": "Yes",
      "adaptive_support": "No"
    }
  ]
}
```

**GET example**

```
GET /recommend?query=Python+developer&format=json
```

📁 Project Structure

```text
shl-recommender/
├── app.py                    # Flask application & API endpoints
├── Dockerfile                # Docker configuration for deployment
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── src/
│   ├── __init__.py
│   ├── embeddings.py         # Vector store & embedding logic
│   └── scrapper.py           # SHL catalog scraper (file in repo; rename optional)
│
├── templates/
│   └── index.html            # Web UI template
│
├── data/
│   ├── chroma_db/
│   │   └── chroma.sqlite3    # ChromaDB local DB (vector store)
│   └── raw/
│       └── shl_catalog.json  # Scraped assessment data (377 items)
│
└── evaluation/
    └── predictions_test_set.csv  # Test set predictions
```

*Note: If you prefer to standardize naming, I can rename `src/scrapper.py` → `src/scraper.py` and update imports/README.*
🚀 Local Development

### Prerequisites
- Python 3.11+
- Groq API key (get a free key)

### Setup

```bash
# Clone the repository
git clone https://github.com/Inder-26/shl-recommender.git
cd shl-recommender

# Create virtual environment (Windows)
python -m venv .venv
.venv\Scripts\activate
# For Linux/macOS: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variable (Windows PowerShell)
# Replace with your key
setx GROQ_API_KEY "your_api_key_here"

# Run the application
python app.py
```

### Access
- **Web UI:** http://localhost:7860
- **API:** http://localhost:7860/recommend
🐳 Docker Deployment

```bash
# Build
docker build -t shl-recommender .

# Run
docker run -p 7860:7860 -e GROQ_API_KEY=your_key shl-recommender
```
📈 Algorithm Pipeline

#### Input Processing
- Detect if the input is a URL and extract job description text when needed
- Clean and normalize query text

#### LLM Analysis (Groq)
- Extract technical skills (e.g., Python, Java, SQL)
- Extract soft skills (communication, leadership, collaboration)
- Detect role type (technical, sales, managerial, etc.)
- Parse duration or time constraints if specified

#### Query Expansion
- Generate multiple specialized search queries
- Create role-specific queries for broader coverage

#### Vector Search
- Encode queries with SentenceTransformers
- Search ChromaDB for similar assessments
- Apply keyword boosting for exact matches

#### Result Balancing & Post-processing
- Interleave Knowledge (K) and Personality (P) assessment types
- Apply duration filters when specified
- Deduplicate and rank results

#### Response Formatting
- Return top 10 assessments with metadata (name, URL, type, duration)
📝 Example Queries

![Query example](https://raw.githubusercontent.com/Inder-26/shl-recommender/main/images/query-example.png)
*Illustration: sample query flow and expected match types.*

| Query | Expected Result Types |
|-------|----------------------|
| "Java developer with team collaboration skills" | Java tests (K) + Personality tests (P) |
| "Senior Data Analyst proficient in SQL, Python, Excel" | Technical tests (K) + Aptitude tests (A) |
| "Customer service representative with English communication" | Communication tests (K) + Personality tests (P) |
| "Entry-level sales role for graduates, 30 min max" | Sales assessments (P) filtered by duration |
🧪 Evaluation

Run evaluation on training set:

```bash
python evaluate.py
```

Generate predictions for test set:

```bash
python generate_predictions.py
```

📄 Submission Deliverables

| Deliverable | Location |
|-------------|----------|
| API Endpoint | https://inderjeet-shl-recommender.hf.space/recommend |
| GitHub Repository | https://github.com/Inder-26/shl-recommender |
| Web Frontend | https://inderjeet-shl-recommender.hf.space |
| Predictions CSV | evaluation/predictions_test_set.csv |

🔮 Future Improvements
- Fine-tune embedding model on SHL-specific data
- Add cross-encoder re-ranking for better precision
- Implement user feedback loop
- Cache frequent queries
- Add more assessment metadata
👤 Author
Inderjeet Singh

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
- **SHL** — for providing the assessment catalog
- **Groq** — for LLM inference and model access
- **Hugging Face** — for hosting the demo on Spaces
- **SentenceTransformers** — for embeddings and model utilities
- **ChromaDB** — for the vector database and search tooling

*Thanks to all upstream projects and contributors who made this work possible.*
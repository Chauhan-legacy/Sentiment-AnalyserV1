# AI-Based Sentiment Analyzer

A locally deployable sentiment analysis application for evaluating reviews, tweets, comments, and other textual feedback. It classifies text as **positive, neutral, or negative** and provides visual analytics to explore sentiment patterns across individual and batch datasets.

Built using Python, Flask, and SQLite, the application operates locally without requiring external APIs or cloud infrastructure.

## Overview

The Sentiment Analyzer combines a lightweight lexicon-based sentiment engine with optional transformer-based inference. It supports individual text evaluation, batch processing, historical analysis, and report generation through a web-based dashboard.

The project is designed to explore practical applications of natural language processing, text classification, and sentiment-driven data visualization.

## Features

### Sentiment Analysis
- Hybrid sentiment analysis using a rule-based lexicon engine.
- Negation handling, including phrases such as *"not good."*
- Intensity modifiers, capitalization, punctuation emphasis, emoticons, and contrastive clauses.
- Optional transformer-based sentiment inference through Hugging Face.

### Batch Processing
- Upload CSV and TXT files for bulk evaluation.
- Automatically identify review or text columns in CSV datasets.
- Analyze multiple text entries in a single operation.

### Analytics Dashboard
- Sentiment distribution visualization.
- Aggregate sentiment or mood score.
- Sentiment score histogram.
- Daily sentiment trend analysis.
- Positive and negative keyword summaries.

### Data Management
- Persistent analysis history using SQLite.
- Retrieve previous evaluations.
- Export stored analysis records to CSV.
- Clear locally stored records when required.

## Technology Stack

| Component | Technology |
|---|---|
| Backend | Python, Flask |
| Sentiment Engine | Rule-based NLP |
| Optional ML Inference | Hugging Face Transformers, PyTorch |
| Database | SQLite |
| Data Processing | Python |
| Visualization | Dashboard-based analytics |
| Interface | HTML, CSS, JavaScript |

## Installation and Setup

### Requirements

- Python 3.9 or newer
- pip
- A modern web browser

### 1. Clone the repository

```bash
git clone <repository-url>
cd sentiment-analyzer
```

### 2. Install dependencies

```bash
pip install flask
```

### 3. Launch the application

```bash
python app.py
```

### 4. Open the dashboard

Navigate to:

```text
http://127.0.0.1:5000
```

The application will run locally on your machine.

## Optional: Transformer-Based Sentiment Analysis

The application can optionally use Hugging Face Transformers for neural-network-based sentiment inference.

Install the additional dependencies:

```bash
pip install transformers torch "numpy<2.0.0"
```

Enable transformer mode before launching the application.

**Windows PowerShell:**

```powershell
$env:USE_TRANSFORMERS="1"
python app.py
```

**Linux / macOS:**

```bash
USE_TRANSFORMERS=1 python app.py
```

Transformer mode may require downloading model weights during initial setup. Once downloaded, supported models can be used locally.

> Transformer inference is optional. The default rule-based engine does not require an external API or cloud service.

## Project Structure

```text
sentiment-analyzer/
│
├── app.py
│   └── Flask application, sentiment engine,
│       API routes, and database operations
│
├── templates/
│   └── index.html
│       └── Dashboard interface
│
└── sentiment.db
    └── SQLite database (generated automatically)
```

The SQLite database is created automatically when the application initializes, provided the database initialization logic is implemented in `app.py`.

## API Reference

The application exposes REST-style endpoints for sentiment analysis, batch processing, reporting, and data management.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/analyze` | Analyze a single text entry |
| `POST` | `/api/batch` | Process multiple texts or uploaded files |
| `GET` | `/api/report` | Retrieve aggregated analytics |
| `GET` | `/api/history` | Retrieve the latest 100 analysis records |
| `GET` | `/api/export` | Export stored records as CSV |
| `POST` | `/api/clear` | Clear stored analysis history |

### Example: Single Text Analysis

**Request**

```http
POST /api/analyze
Content-Type: application/json
```

```json
{
  "text": "The product is excellent and works perfectly!"
}
```

The endpoint evaluates the supplied text and returns the sentiment result according to the application's response schema.

### Example: Batch Analysis

```json
{
  "texts": [
    "The service was excellent.",
    "The experience was disappointing.",
    "The product arrived yesterday."
  ]
}
```

The batch endpoint processes multiple entries in one request.

*Exact response fields depend on the implementation in `app.py`.*

## Data Storage and Privacy

- Analysis history is stored in a local SQLite database.
- No mandatory third-party API is required.
- Transformer mode can operate locally after model setup.
- CSV exports are generated from stored analysis records.

The application is intended for local experimentation and analysis. Avoid processing sensitive personal information unless appropriate safeguards are in place.

## Limitations

- Rule-based sentiment analysis may struggle with sarcasm, context, slang, and complex language.
- Sentiment classification is not equivalent to understanding human emotion.
- Transformer predictions depend on the selected model and its training data.
- Analytics quality depends on the quantity and quality of analyzed text.

## Future Development

Potential extensions include:

- Aspect-based sentiment analysis.
- Multilingual sentiment classification.
- Model comparison and evaluation metrics.
- Confidence scores and uncertainty visualization.
- Improved dataset preprocessing.
- Real-time sentiment monitoring.
- Advanced NLP models and domain-specific classifiers.

## License

This project is intended for educational, research, and portfolio development.

Add a `LICENSE` file before distributing the project under a specific open-source license.

---

**AI-Based Sentiment Analyzer**  
Natural Language Processing · Machine Learning · Data Analytics · Flask
# Jupiter FAQ Assistant: Semantic Retrieval and LLM-Powered Q&A System

An end-to-end AI system that scrapes community discussions from the [Jupiter Money Forum](https://community.jupiter.money) and answers user queries using a semantic search engine and open-source LLMs.

---

## 🔍 Overview

The **Jupiter Community FAQ Assistant** is a comprehensive information retrieval and question-answering system designed for:

- Extracting data from the Jupiter Community forum.
- Classifying user queries.
- Retrieving semantically relevant discussions.
- Generating responses using LLMs.

---

## 🧠 Components

### 1. Web Scraper (Playwright-based)

An asynchronous, robust Playwright-based scraper that:
- Crawls the `/c/help/27` category to gather topic URLs.
- Extracts all tag-specific topic URLs from `/tags`.
- Scrolls through dynamically loaded content to collect post URLs.
- For each topic, extracts:
  - Main question and all replies.
  - User metadata (name and title).
  - Tags, images, and links.

📦 **Output**: JSON files like `faq_data_raw.json`, `faq_data_test.json`.

### 2. LLM-Powered Assistant

Uses an open-source LLM API (e.g., Mixtral via Together.ai) with Sentence Transformers for multilingual semantic embeddings.

#### Key Pipeline Components:

| Function | Purpose |
|----------|---------|
| `clean_text()` | Cleans and normalizes HTML content |
| `ensure_english()` | Converts non-English text to English (if needed) |
| `classify_tags_with_open_llm()` | Classifies user query into relevant tags |
| `build_or_load_index_and_embeddings()` | Builds/loads FAISS semantic index |
| `retrieve_with_prefilter()` | Retrieves top-k relevant posts using tags & FAISS |
| `generate_answer_with_open_llm()` | Uses LLM to answer using retrieved discussions |
| `suggest_related()` | Recommends similar historical questions |

---

## 🚀 Workflow

1. Run the scraper to collect and save forum data.
2. Preprocess and embed the scraped content.
3. User enters a query:
   - Query is classified into tags.
   - Tag-filtered FAISS retrieval fetches top similar posts.
   - LLM generates a contextual answer from the retrieved content.

✅ Test mode is available to limit the crawl for development and debugging.

---

## 📁 Outputs

- `faq_data_raw.json` — Raw forum data.
- `faq_data_test.json` — Sample output in test mode.
- `faiss_index/` — Vector index for fast semantic retrieval.

---

## 🛠 Tech Stack

- **Playwright (async)** – Web scraping
- **FAISS** – Semantic similarity search
- **Sentence Transformers** – Multilingual embeddings
- **Open-source LLMs (e.g., Mixtral)** – Contextual answer generation
- **Python** – Core logic and orchestration

---

## 📌 Use Case

Ideal for automating customer support or building intelligent community assistants for forums like Jupiter, by bridging the gap between user intent and historical answers.

---


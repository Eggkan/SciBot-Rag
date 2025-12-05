# SciBot-Rag
# SciBot-RAG: Educational AI Assistant with Cross-Encoder Reranking

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)
![RAG](https://img.shields.io/badge/Architecture-RAG-orange?style=flat-square)
![Model](https://img.shields.io/badge/LLM-Microsoft%20Phi--1.5-blueviolet?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-success?style=flat-square)

**SciBot-RAG** is a domain-specific Question Answering (QA) system designed for science education. It leverages a **Retrieval-Augmented Generation (RAG)** architecture to provide accurate answers based on a structured dataset, minimizing hallucinations common in standard LLMs.

Unlike basic RAG implementations, this project utilizes a **two-stage retrieval process** (Vector Search + Cross-Encoder Reranking) to ensure high precision in context selection.

---

## 🏗 System Architecture

The pipeline operates entirely locally and consists of the following components:

1.  **Data Ingestion (ETL):** Raw CSV data is processed and converted into a structured JSON format suitable for embedding.
2.  **Vector Retrieval (Recall):**
    *   **Embedding Model:** `dbmdz/bert-base-turkish-cased`
    *   **Vector Database:** FAISS (Facebook AI Similarity Search) is used to retrieve the top-k most similar documents based on cosine similarity.
3.  **Reranking (Precision):**
    *   **Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
    *   The initial candidates are re-scored by a Cross-Encoder to filter out irrelevant context, significantly improving answer quality.
4.  **Generation:**
    *   **LLM:** `microsoft/phi-1_5`
    *   The refined context is fed into the Small Language Model (SLM) to generate a coherent, natural language response.

---

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `chatbot_rag.py` | Main inference script launching the **Gradio** web interface. |
| `create_vector_db.py` | Generates embeddings from the JSON dataset and builds the FAISS index. |
| `prepare_data_.py` | ETL script to clean and convert raw CSV data to JSON. |
| `req.py` | List of project dependencies. |
| `fen_bilimleri_soru_cevap.csv` | The raw source dataset (Science Q&A). |

---

## 🚀 Installation & Usage

Follow these steps to set up the project locally.

### 1. Clone the Repository
```bash
git clone https://github.com/eggkan/scibot-rag.git
cd scibot-rag
pip install -r req.py
# (Note: For GPU acceleration, ensure torch is installed with CUDA support).

## Build Database(IMPORTANT)
# Step 1: Convert CSV to JSON training data
python prepare_data_.py

# Step 2: Create Embeddings and FAISS Index
python create_vector_db.py

#Run the app
python chatbot_rag.py
## The application will be accessible at http://127.0.0.1:7860 in your browser.

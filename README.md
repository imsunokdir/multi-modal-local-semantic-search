# 🧠 Multimodal Local Semantic Search Engine

An AI-powered application built with **Python** and **Streamlit** that enables **context-aware file retrieval** from a local machine, supporting both natural language queries for **text documents** and **images**.

This project implements two different state-of-the-art AI models for semantic embedding and uses a high-performance vector database (**Faiss**) for near-instant search results. 

## ✨ Features

* **Multimodal Search:** Supports semantic search across both text (`.pdf`, `.txt`, etc.) and image (`.png`, `.jpg`) files.
* **Semantic Retrieval:** Uses deep learning models to search by **meaning and context**, not just keywords.
* **Vector Database:** Implements **Faiss** for blazing-fast indexing and similarity search of high-dimensional vectors.
* **Text-to-Image Search:** Enables searching your images using natural language descriptions (e.g., "A dog running in a park").
* **Interactive UI:** Hosted via a user-friendly web interface built with **Streamlit**.

## ⚙️ Technology Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **User Interface** | `streamlit` | Front-end web application framework. |
| **Vector Database** | **Faiss** | Efficient library for similarity search and clustering of dense vectors. |
| **Text Embedding** | `all-MiniLM-L6-v2` (Sentence Transformers) | Converts text into meaningful vectors for semantic search. |
| **Image Embedding** | **CLIP-ViT-B-32** (Sentence Transformers) | Cross-modal model that aligns text and images in a shared vector space. |
| **Core Language** | Python, NumPy | Main programming language and numerical processing. |

## 🚀 Getting Started

Follow these steps to set up and run the application on your local machine.

### 1. Prerequisites

Ensure you have Python (3.8+) and a virtual environment set up.

```bash
# Create and activate a virtual environment
python -m venv venv

# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# On macOS/Linux:
source venv/bin/activate

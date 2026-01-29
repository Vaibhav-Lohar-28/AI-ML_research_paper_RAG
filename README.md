# AI/ML Research Paper RAG System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A sophisticated Retrieval-Augmented Generation (RAG) system designed to intelligently interact with AI and Machine Learning research papers. This system combines advanced document processing, vector embeddings, and agentic workflows to enable semantic search and natural language querying of research literature.

## 🌟 Features

- **Intelligent Document Processing**: Automated ingestion and parsing of AI/ML research papers (PDF format)
- **Semantic Search**: Vector-based similarity search for finding relevant research papers and passages
- **RAG Pipeline**: State-of-the-art retrieval-augmented generation for context-aware responses
- **Agentic Workflow**: AI agent capabilities for autonomous research paper analysis and summarization
- **Interactive Interface**: User-friendly web application built with Streamlit
- **Modular Architecture**: Clean, maintainable code structure with separate components for each functionality

## 📋 Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Notebooks](#notebooks)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## 🏗️ Architecture

The system is built on four core components:

1. **Data Ingestion Module**: Handles PDF parsing, text extraction, and preprocessing
2. **Vector Store**: Manages document embeddings using state-of-the-art embedding models
3. **RAG Pipeline**: Implements retrieval and generation mechanisms for query answering
4. **Agentic Layer**: Provides autonomous capabilities for complex research tasks

```
┌─────────────────┐
│  Research Papers │
│     (PDFs)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Ingestion  │
│   & Chunking    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Embedding &    │
│  Vector Store   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│   RAG Pipeline  │◄─────┤ User Query   │
│   & Retrieval   │      └──────────────┘
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Agentic Workflow│
│   & Response    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Web Interface  │
│   (Streamlit)   │
└─────────────────┘
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Vaibhav Lohar**
- GitHub: [@Vaibhav-Lohar-28](https://github.com/Vaibhav-Lohar-28)


## 📧 Contact

For questions, suggestions, or collaboration opportunities, please open an issue on GitHub.


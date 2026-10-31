---
title: Industrial AI Assistant
emoji: 🏭
colorFrom: blue
colorTo: gray
sdk: streamlit
sdk_version: 1.66.1
python_version: "3.12"
app_file: app.py
pinned: false
---

# 🏭 Industrial AI Assistant

An agentic RAG assistant. A LangGraph tool-calling agent decides whether to answer
directly, search a local knowledge base on industrial AI (predictive maintenance,
quality control, MLOps, RAG, LLMs), or search the live web.

**Status:** Milestone 1, the baseline agent.

## Data attribution
The knowledge-base documents are Wikipedia articles, licensed CC BY-SA 4.0.
Each file in `data/docs/` keeps its source URL at the top.

## Run locally
```bash
pip install -r requirements-dev.txt
cp .env.example .env        # then edit it: set ENV_MODE and, for cloud mode, GROQ_API_KEY
python -m streamlit run app.py
```
```markdown
# InfoSpeakAI - Transform Data into Inspiring Speeches

Welcome to **InfoSpeakAI** – an autonomous AI agent designed to gather insights from company and country data and transform them into compelling, well-structured speeches.

[![GitHub license](https://img.shields.io/github/license/HabtamuFeyera/InfoSpeakAI.svg)](https://github.com/HabtamuFeyera/InfoSpeakAI.git)

---

## ✨ Key Features

- **Automated Data Retrieval:**  
  Seamlessly fetch detailed company profiles and country overviews from external APIs, capturing everything from industry trends to cultural insights.

- **Advanced Text Processing & Embedding:**  
  Convert raw JSON data into meaningful chunks with our recursive text splitter, then generate high-quality embeddings to index and retrieve key information quickly using Pinecone.

- **LLM-Powered Speech Generation:**  
  Leverage the capabilities of GPT-4 to draft, review, and refine speeches. The AI not only generates an initial draft but also iterates based on constructive feedback, ensuring a polished final output.

- **Dynamic Workflow Orchestration:**  
  Our state graph-driven system, powered by LangGraph, manages every step – from data ingestion to final speech output – with built-in checkpointing and revision loops for maximum flexibility.


## 🛠️ How It Works

1. **Data Ingestion & Preprocessing:**  
   InfoSpeakAI loads raw JSON files containing company and country details, combines relevant text fields, and splits the content into digestible chunks.

2. **Embedding Generation & Indexing:**  
   Using OpenAI’s embedding model, it transforms text chunks into vector representations, which are then indexed in Pinecone for rapid retrieval during speech generation.

3. **Intelligent Speech Generation:**  
   With a wealth of curated data at its fingertips, the AI agent leverages GPT-4 to craft an initial speech draft. It then enters an iterative review loop, refining the content based on feedback until it meets the desired quality.

4. **Workflow Orchestration:**  
   The entire process is managed by a state graph, ensuring that every step – from fetching data to formatting the final speech – is executed in a logical, seamless sequence.

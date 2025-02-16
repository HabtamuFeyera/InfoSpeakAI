# Autonomous AI Agent for Information Retrieval & Speech Generation

## Overview

This project develops an autonomous AI agent that generates engaging speeches by automatically retrieving and processing detailed company and country information. The system combines data from pre-prepared JSON documents (extracted from Wikipedia) with live internet data from the Tavily API to create high-quality, up-to-date speeches—all within an interactive Jupyter Notebook environment.

## Features

- **Automated Data Retrieval:**
  - Extracts data from JSON files containing company and country information.
  - Fetches real-time data using the Tavily API.
- **Advanced Text Generation:**
  - Utilizes GPT-4 to generate, review, and refine speech drafts.
- **Efficient Data Handling:**
  - Employs Pinecone for vector-based similarity searches on pre-prepared documents.
- **Iterative Workflow:**
  - Uses a directed workflow graph (via LangGraph) to manage planning, content generation, review, and iterative revisions.
- **Modular Architecture:**
  - Designed for scalability and integration with additional expert agents and future enhancements.

## Setup & Usage in Jupyter Notebook

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/HabtamuFeyera/InfoSpeakAI.git
   cd InfoSpeakAI
   ```

2. **Install Dependencies:**
   Install the required packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configuration:**
   - Set up your API keys for GPT-4, Pinecone, and Tavily.

4. **Launch Jupyter Notebook:**
   Start the notebook server:
   ```bash
   jupyter notebook
   ```
   Open the main notebook file in your browser.

5. **Run the Notebook:**
   - Execute the notebook cells sequentially to initialize the environment, process data, and generate the speech.
   - The notebook is structured to guide you through data extraction, content generation, and iterative refinements.

## Future Enhancements

- Expand multi-agent collaboration with additional expert agents.
- Integrate dynamic user feedback for continuous improvement.
- Enhance data processing algorithms for even more accurate information retrieval.

## References

- [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
- [RAG Agents - Stanford CS230](https://cs230.stanford.edu/syllabus/fall_2024/rag_agents.pdf)
- [Improving LLM Performance](https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/)

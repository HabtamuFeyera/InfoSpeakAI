# AI Agent for Information Retrieval & Speech Generation

This project implements an AI agent that autonomously gathers company and country information and generates a well-structured, engaging speech for an invited guest. It leverages several third-party APIs and libraries to perform tasks such as retrieving data from external sources, generating embeddings, indexing information for fast retrieval, and using large language models (LLMs) for content generation and revision.

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Workflow Details](#workflow-details)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Features

- **Company & Country Data Retrieval**:  
  Retrieves detailed company information (background, industry, achievements, impact) and country information (economic landscape, culture, business environment) using external APIs.

- **Text Processing & Embedding Generation**:  
  Loads JSON files containing raw data, combines relevant text columns, splits documents into manageable chunks, and generates text embeddings.

- **Vector Store Integration (Pinecone)**:  
  Indexes document chunks with their embeddings for rapid retrieval using Pinecone’s vector database.

- **Speech Generation & Review**:  
  Uses an LLM (GPT-4 via OpenAI’s API) to generate an initial draft of a speech and subsequently provide a review with constructive feedback.

- **Workflow Orchestration**:  
  Implements a state graph workflow using LangGraph to manage the sequential steps (fetching info, generating speech, revising) with checkpointing capabilities.

## Architecture Overview

1. **Data Loading & Preprocessing**:  
   - Loads company and country information from JSON files.
   - Combines multiple text columns into a single descriptive field.
   - Splits text into smaller chunks using a recursive text splitter.

2. **Embedding Generation & Indexing**:  
   - Uses OpenAI’s embeddings model to generate vector representations.
   - Upserts the vectorized chunks into a Pinecone index to allow retrieval later in the workflow.

3. **Data Retrieval & Speech Generation**:  
   - Fetches additional information using the Tavily client.
   - Retrieves contextually related data from the Pinecone vector store.
   - Uses an LLM to generate and review a draft speech based on the aggregated information.

4. **Workflow Orchestration**:  
   - Manages all of the above steps through a state graph defined with LangGraph.
   - Allows for iterative speech revisions up to a predefined maximum number of revisions.

## Prerequisites

- **Python 3.8+**  
- Required environment variables (API keys):
  - `OPENAI_API_KEY`
  - `PINECONE_API_KEY`
  - `TAVILY_API_KEY`

- **Third-Party Libraries**:
  - `numpy`
  - `pandas`
  - `python-dotenv`
  - `langchain`
  - `pinecone-client`
  - `tavily`
  - `langgraph`

> Make sure to install all dependencies before running the project.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/ai-agent-speech-generator.git
   cd ai-agent-speech-generator
   ```

2. **Create and Activate a Virtual Environment (Optional but Recommended):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   > **Note:** If a `requirements.txt` file is not provided, ensure you install all the necessary libraries manually.

## Configuration

1. **Environment Variables:**

   Create a `.env` file in the root directory of your project and include the following variables:

   ```dotenv
   OPENAI_API_KEY=your_openai_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

2. **JSON Data Files:**

   Ensure you have the following JSON files in the specified paths (or adjust the file paths in the code):
   - `/content/files/all_countries_info.json`
   - `/content/company_info.json`

   These files should contain the raw data for country and company information respectively.

## Usage

Run the main entry point of the project:

```bash
python your_script_name.py
```

When executed, the agent will:

1. Load and process the JSON data files.
2. Create or verify the existence of a Pinecone index.
3. Generate text embeddings and upsert document chunks into Pinecone.
4. Retrieve company and country information via the Tavily client.
5. Invoke a workflow that generates and iteratively reviews a speech draft based on the collected data.
6. Print and log the final output.

## Workflow Details

The workflow is defined as a state graph with the following nodes:

1. **Fetch Company Info**:  
   Uses the Tavily client to gather comprehensive company data.

2. **Fetch Country Info**:  
   Retrieves key country data including economic and cultural insights.

3. **Retrieve External Info**:  
   Searches the Pinecone vector store for additional context to enrich the information.

4. **Generate Speech**:  
   Constructs an initial draft speech using GPT-4, combining the company and country information.

5. **Review Speech**:  
   Generates constructive feedback on the draft speech.

6. **Revision Decision**:  
   Determines whether further revisions are necessary based on a maximum revision count.

The workflow leverages LangGraph for state management and checkpointing (using `MemorySaver`), enabling persistent state during iterative revisions.

## Project Structure

```

## Troubleshooting

- **Missing API Keys:**  
  If the script raises an error regarding missing API keys, ensure your `.env` file is correctly set up and located in the project root.

- **File Not Found Errors:**  
  Verify that the JSON data files exist at the specified paths. Update the file paths in the code if necessary.

- **Pinecone Index Issues:**  
  Ensure your Pinecone API key is valid and that you have the necessary permissions. The script automatically creates the index if it does not exist.

- **Dependency Errors:**  
  Double-check that all required libraries are installed. Use `pip install <library-name>` to install any missing dependencies.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

This README provides a comprehensive guide to understanding, setting up, and running the AI agent. Adjust paths, API keys, and additional settings as needed for your environment. Enjoy building and iterating on your speech generation agent!

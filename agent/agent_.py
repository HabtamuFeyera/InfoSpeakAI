import os
import re
import logging
import concurrent.futures
import statistics
from typing import List, TypedDict, Dict, Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone

from pinecone import Pinecone as PineconeClient, ServerlessSpec
from tavily import TavilyClient

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ------------------------------
# Environment and Logging Setup
# ------------------------------
load_dotenv()  
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_env_param(key: str, default: str) -> str:
    return os.getenv(key, default)

CHUNK_SIZE = int(get_env_param("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(get_env_param("CHUNK_OVERLAP", "50"))
MAX_REVISIONS = int(get_env_param("MAX_REVISIONS", "3"))

# Required API keys 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not all([OPENAI_API_KEY, PINECONE_API_KEY, TAVILY_API_KEY]):
    raise ValueError("Missing one or more required API keys. Check your environment variables.")

# ------------------------------
# Initialize External Clients and Models
# ------------------------------
embedding_function = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
llm_model = ChatOpenAI(model="gpt-4", temperature=0)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# ------------------------------
# Pinecone Initialization and Index Management
# ------------------------------
INDEX_NAME = "promptwithrag"
pc = PineconeClient(api_key=PINECONE_API_KEY)

def manage_pinecone_index() -> None:
    try:
        existing_indexes = pc.list_indexes().names()
        if INDEX_NAME in existing_indexes:
            logging.info(f"Pinecone index '{INDEX_NAME}' already exists.")
        else:
            logging.info(f"Creating Pinecone index '{INDEX_NAME}'...")
            spec = ServerlessSpec(cloud="aws", region="us-east-1")
            pc.create_index(name=INDEX_NAME, dimension=1536, metric="cosine", spec=spec)
            logging.info(f"Pinecone index '{INDEX_NAME}' created successfully.")
    except Exception as e:
        logging.exception("Error managing Pinecone index: %s", e)

manage_pinecone_index()
index = pc.Index(INDEX_NAME)

# ------------------------------
# Data Loading and Preprocessing
# ------------------------------
def load_json_file(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        logging.error("File not found: %s", file_path)
        return pd.DataFrame()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            df = pd.read_json(f)
        logging.info("Loaded %d records from %s", len(df), file_path)
        return df
    except Exception as e:
        logging.exception("Error loading file %s: %s", file_path, e)
        return pd.DataFrame()

def combine_text_columns(df: pd.DataFrame, expected_columns: List[str], new_col_name: str) -> pd.DataFrame:
    present_columns = [col for col in expected_columns if col in df.columns]
    if not present_columns:
        present_columns = df.select_dtypes(include="object").columns.tolist()
        logging.warning("No expected columns for '%s'; using all text columns: %s", new_col_name, present_columns)
    df[new_col_name] = df[present_columns].fillna("").agg(" ".join, axis=1)
    return df

def chunk_documents(df: pd.DataFrame, text_column: str, source: str) -> List[Dict[str, Any]]:
    result = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    for doc_idx, row in df.iterrows():
        text = row.get(text_column, "")
        if not text or len(text) < 100:
            continue
        for i, chunk in enumerate(splitter.split_text(text)):
            result.append({
                "doc_id": f"{source}_{doc_idx}",
                "chunk_id": i,
                "text": chunk,
                "source": source
            })
    return result

def upsert_embeddings(df_chunks: pd.DataFrame) -> None:
    chunk_texts = df_chunks["text"].tolist()
    try:
        chunk_embeddings = embedding_function.embed_documents(chunk_texts)
    except Exception as e:
        logging.exception("Error generating embeddings: %s", e)
        return

    def process_batch(batch: List[tuple]) -> bool:
        try:
            index.upsert(vectors=batch)
            return True
        except Exception as e:
            logging.exception("Error upserting batch: %s", e)
            return False

    upsert_data = [
        (
            f"{row['doc_id']}_{row['chunk_id']}",
            chunk_embeddings[idx],
            {"doc_id": row["doc_id"], "chunk_id": row["chunk_id"], "text": row["text"], "source": row["source"]}
        )
        for idx, row in df_chunks.iterrows()
    ]
    batch_size = 50
    batches = [upsert_data[i:i + batch_size] for i in range(0, len(upsert_data), batch_size)]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_batch, batches))
    successful_batches = sum(results)
    logging.info("Upsert completed: %d out of %d batches succeeded.", successful_batches, len(batches))

# Load and process data
countries_df = load_json_file("/home/habte/data/all_countries_info.json")
companies_df = load_json_file("/home/habte/data/company_info.json")
countries_df = combine_text_columns(countries_df, ["Summary", "Economy", "Culture"], "country_description")
companies_df = combine_text_columns(companies_df, ["Background", "Industry", "Achievements", "Impact"], "company_description")
chunks = []
chunks.extend(chunk_documents(countries_df, "country_description", "country"))
chunks.extend(chunk_documents(companies_df, "company_description", "company"))
df_chunks = pd.DataFrame(chunks)
upsert_embeddings(df_chunks)
vectorstore = Pinecone.from_existing_index(index_name=INDEX_NAME, embedding=embedding_function)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ------------------------------
# Agent State and Prompt Templates
# ------------------------------
class AgentState(TypedDict):
    company_name: str
    country_name: str
    company_info: str
    country_info: str
    retrieved_info: List[str]
    event_context: str
    audience_profile: str
    plan: str
    content_draft: str
    review_feedback: str
    collaboration_draft: str
    formatted_speech: str
    reflection: str
    readability_score: float
    draft_speech: str
    revision_number: int
    max_revisions: int
    user_feedback: str

PLAN_CREATION_PROMPT = """
You are a strategic planner. Given the event "{event_context}" and the audience "{audience_profile}",
create a detailed step-by-step plan to write an engaging speech that incorporates company and country insights.
Include steps like: gathering information, outlining, drafting, reviewing, and finalizing.
"""

CONTENT_AGENT_PROMPT = """
You are the Content Agent. Using the following information, produce an initial draft of the speech.
Company Information:
{company_info}

Country Information:
{country_info}

Plan:
{plan}

Generate a speech draft with a clear introduction, body, and conclusion.
"""

REVIEW_AGENT_PROMPT = """
You are the Review Agent. Critically review the following speech draft:
{content_draft}
Provide detailed feedback on structure, clarity, engagement, and suggestions for improvement.
"""

COLLABORATION_PROMPT = """
The Content Agent produced the following draft:
{content_draft}

The Review Agent provided the following feedback:
{review_feedback}

Collaboratively generate an improved version of the speech draft that incorporates the feedback.
"""

FORMAT_SPEECH_PROMPT = """
Organize the following draft into a well-structured speech with these sections:
1. Introduction
2. Body
3. Conclusion

Draft:
{collaboration_draft}
"""

REFLECTION_PROMPT = """
Reflect on the following speech:
{formatted_speech}

Describe its strengths and weaknesses and propose ways to improve the process in future iterations.
"""

USER_FEEDBACK_PROMPT = """
Incorporate the following user feedback into the speech to improve its quality.

User Feedback:
{user_feedback}

Current Speech:
{formatted_speech}
"""

def llm_call(prompt: str, additional_messages: List[Dict[str, str]] = None) -> str:
    messages = [{"role": "system", "content": prompt}]
    if additional_messages:
        messages.extend(additional_messages)
    try:
        response = llm_model(messages)
        return response.content
    except Exception as e:
        logging.error("LLM call error: %s", e)
        return f"Error: {str(e)}"

def compute_enhanced_readability(text: str) -> Dict[str, float]:
    try:
        words = re.findall(r'\w+', text)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s for s in sentences if s.strip()]
        avg_word_length = statistics.mean([len(word) for word in words]) if words else 0.0
        avg_sentence_length = statistics.mean([len(re.findall(r'\w+', s)) for s in sentences]) if sentences else 0.0
        sentence_count = len(sentences)
        return {
            "avg_word_length": avg_word_length,
            "avg_sentence_length": avg_sentence_length,
            "sentence_count": float(sentence_count)
        }
    except Exception as e:
        logging.error("Error computing readability metrics: %s", e)
        return {"avg_word_length": 0.0, "avg_sentence_length": 0.0, "sentence_count": 0.0}

def validate_state(state: AgentState) -> None:
    required_fields = ["company_name", "country_name", "event_context", "audience_profile"]
    for field in required_fields:
        if not state.get(field) or not isinstance(state.get(field), str) or not state[field].strip():
            raise ValueError(f"State missing or invalid required field: {field}")
    if state.get("max_revisions", 0) <= 0:
        raise ValueError("max_revisions must be a positive integer.")

# ------------------------------
# Agent Node Functions
# ------------------------------
def retrieve_company_info(state: AgentState) -> Dict[str, Any]:
    query = f"{state['company_name']} company overview, achievements, industry, and impact"
    try:
        response = tavily_client.search(query=query, max_results=3)
        if "results" in response:
            info_list = [
                f"{item.get('content', '')} (Source: {item.get('source', 'Unknown')})"
                for item in response["results"]
            ]
            info = "\n".join(info_list)
        else:
            info = "Company information not retrieved."
    except Exception as e:
        logging.error("Error retrieving company info: %s", e)
        info = "Error during company info retrieval."
    logging.info("Step 1: Retrieved Company Info:\n%s", info)
    return {"company_info": info}

def retrieve_country_info(state: AgentState) -> Dict[str, Any]:
    query = f"{state['country_name']} economic landscape, culture, and business environment"
    try:
        response = tavily_client.search(query=query, max_results=3)
        if "results" in response:
            info_list = [
                f"{item.get('content', '')} (Source: {item.get('source', 'Unknown')})"
                for item in response["results"]
            ]
            info = "\n".join(info_list)
        else:
            info = "Country information not retrieved."
    except Exception as e:
        logging.error("Error retrieving country info: %s", e)
        info = "Error during country info retrieval."
    logging.info("Step 2: Retrieved Country Info:\n%s", info)
    return {"country_info": info}

def retrieve_external_info(state: AgentState) -> Dict[str, Any]:
    try:
        company_retrieval = retriever.invoke(f"{state['company_name']} company background")
        country_retrieval = retriever.invoke(f"{state['country_name']} country economic landscape")
        combined_results = company_retrieval + country_retrieval
        annotated_results = []
        for doc in combined_results:
            if isinstance(doc, dict):
                text = doc.get("page_content", doc.get("text", ""))
                metadata = doc.get("metadata", {})
                source = metadata.get("source", "Unknown")
                annotated_results.append(f"{text} (Source: {source})")
            else:
                annotated_results.append(str(doc))
        retrieved_info = annotated_results
    except Exception as e:
        logging.error("Error retrieving external info: %s", e)
        retrieved_info = []
    logging.info("Step 3: Retrieved External Info:\n%s", retrieved_info)
    return {"retrieved_info": retrieved_info}

def plan_creation(state: AgentState) -> Dict[str, Any]:
    prompt = PLAN_CREATION_PROMPT.format(
        event_context=state["event_context"],
        audience_profile=state["audience_profile"]
    )
    plan = llm_call(prompt, [{"role": "user", "content": "Create the plan."}])
    logging.info("Step 4: Generated Plan:\n%s", plan)
    return {"plan": plan}

def content_agent(state: AgentState) -> Dict[str, Any]:
    prompt = CONTENT_AGENT_PROMPT.format(
        company_info=state["company_info"],
        country_info=state["country_info"],
        plan=state["plan"]
    )
    content_draft = llm_call(prompt, [{"role": "user", "content": "Generate the speech draft."}])
    logging.info("Step 5: Content Agent Draft:\n%s", content_draft)
    return {"content_draft": content_draft, "draft_speech": content_draft}

def review_agent(state: AgentState) -> Dict[str, Any]:
    prompt = REVIEW_AGENT_PROMPT.format(content_draft=state["content_draft"])
    review_feedback = llm_call(prompt, [{"role": "user", "content": "Provide detailed feedback."}])
    logging.info("Step 6: Review Agent Feedback:\n%s", review_feedback)
    return {"review_feedback": review_feedback}

def collaboration_agent(state: AgentState) -> Dict[str, Any]:
    prompt = COLLABORATION_PROMPT.format(
        content_draft=state["content_draft"],
        review_feedback=state["review_feedback"]
    )
    collaboration_draft = llm_call(prompt, [{"role": "user", "content": "Collaborate and produce an improved draft."}])
    logging.info("Step 7: Collaboration Agent Draft:\n%s", collaboration_draft)
    return {"collaboration_draft": collaboration_draft}

def format_speech(state: AgentState) -> Dict[str, Any]:
    prompt = FORMAT_SPEECH_PROMPT.format(collaboration_draft=state["collaboration_draft"])
    formatted_speech = llm_call(prompt, [{"role": "user", "content": "Format the speech."}])
    logging.info("Step 8: Formatted Speech:\n%s", formatted_speech)
    return {"formatted_speech": formatted_speech}

def compute_readability(state: AgentState) -> Dict[str, Any]:
    metrics = compute_enhanced_readability(state["formatted_speech"])
    logging.info("Step 9: Readability Metrics:\n%s", metrics)
    # Here we simply return the average word length as the readability score
    return {"readability_score": metrics["avg_word_length"]}

def reflect(state: AgentState) -> Dict[str, Any]:
    prompt = REFLECTION_PROMPT.format(formatted_speech=state["formatted_speech"])
    reflection = llm_call(prompt, [{"role": "user", "content": "Reflect on the process and suggest improvements."}])
    logging.info("Step 10: Reflection:\n%s", reflection)
    return {"reflection": reflection}

def incorporate_user_feedback(state: AgentState) -> Dict[str, Any]:
    if not state.get("user_feedback", "").strip():
        return {}
    prompt = USER_FEEDBACK_PROMPT.format(
        user_feedback=state["user_feedback"],
        formatted_speech=state["formatted_speech"]
    )
    updated_speech = llm_call(prompt, [{"role": "user", "content": "Incorporate this feedback."}])
    logging.info("Incorporated User Feedback:\n%s", updated_speech)
    return {"formatted_speech": updated_speech}

def should_continue(state: AgentState) -> Dict[str, Any]:
    # Decide the next step based on revision count and presence of user feedback
    if state["revision_number"] < state["max_revisions"]:
        state["revision_number"] += 1
        if state.get("user_feedback", "").strip():
            return {"next": "incorporate_user_feedback"}
        else:
            return {"next": "content_agent"}
    else:
        return {"next": "finalize"}

def finalize(state: AgentState) -> Dict[str, Any]:
    logging.info("Final Output Reached. Final Speech:\n%s", state["formatted_speech"])
    return state

# ------------------------------
# Workflow Graph Assembly
# ------------------------------
def build_workflow_graph() -> StateGraph:
    builder = StateGraph(AgentState)
    
    # Data retrieval nodes
    builder.add_node("fetch_company_info", retrieve_company_info)
    builder.add_node("fetch_country_info", retrieve_country_info)
    builder.add_node("retrieve_external_info", retrieve_external_info)
    
    # Planning and collaboration nodes
    builder.add_node("plan_creation", plan_creation)
    builder.add_node("content_agent", content_agent)
    builder.add_node("review_agent", review_agent)
    builder.add_node("collaboration_agent", collaboration_agent)
    builder.add_node("format_speech", format_speech)
    
    # Tool use and reflection nodes
    builder.add_node("compute_readability", compute_readability)
    builder.add_node("reflect", reflect)
    builder.add_node("incorporate_user_feedback", incorporate_user_feedback)
    
    # Revision controller node and finalization
    builder.add_node("should_continue", should_continue)
    builder.add_node("finalize", finalize)
    
    # Define workflow flow (edges)
    builder.set_entry_point("fetch_company_info")
    builder.add_edge("fetch_company_info", "fetch_country_info")
    builder.add_edge("fetch_country_info", "retrieve_external_info")
    builder.add_edge("retrieve_external_info", "plan_creation")
    builder.add_edge("plan_creation", "content_agent")
    builder.add_edge("content_agent", "review_agent")
    builder.add_edge("review_agent", "collaboration_agent")
    builder.add_edge("collaboration_agent", "format_speech")
    builder.add_edge("format_speech", "compute_readability")
    builder.add_edge("compute_readability", "reflect")
    builder.add_edge("reflect", "should_continue")
    
    # The "should_continue" node returns a dict with key "next" that determines the branch.
    # We define all possible transitions:
    builder.add_edge("should_continue", "incorporate_user_feedback", condition=lambda state, output: output.get("next") == "incorporate_user_feedback")
    builder.add_edge("should_continue", "content_agent", condition=lambda state, output: output.get("next") == "content_agent")
    builder.add_edge("should_continue", "finalize", condition=lambda state, output: output.get("next") == "finalize")
    
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)

# ------------------------------
# Main Execution
# ------------------------------
def main() -> None:
    initial_state: AgentState = {
        "company_name": "Ethio Telecom",
        "country_name": "Ethiopia",
        "company_info": "",
        "country_info": "",
        "retrieved_info": [],
        "event_context": "Annual Conference 2025",
        "audience_profile": "leaders",
        "plan": "",
        "content_draft": "",
        "review_feedback": "",
        "collaboration_draft": "",
        "formatted_speech": "",
        "reflection": "",
        "readability_score": 0.0,
        "draft_speech": "",
        "revision_number": 1,
        "max_revisions": MAX_REVISIONS,
        "user_feedback": ""  
    }
    
    try:
        validate_state(initial_state)
    except ValueError as ve:
        logging.error("Initial state validation error: %s", ve)
        return
    
    workflow_graph = build_workflow_graph()
    try:
        # Set a recursion limit for workflow execution (e.g., 10)
        config = {
            "configurable": {"thread_id": "session_text_only_v2"},
            "recursion_limit": 10
        }
        output = workflow_graph.invoke(initial_state, config=config)
        logging.info("Final Agent Output:\n%s", output)
        print("\n=== Final Agent Output ===\n", output)
    except Exception as e:
        logging.error("Error during workflow execution: %s", e)

if __name__ == '__main__':
    main()

import logging
from tavily import TavilyClient
from agent.config import TAVILY_API_KEY

# Initialize Tavily client
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

def retrieve_company_info(state: dict) -> dict:
    """
    Retrieve detailed company information using the Tavily client.
    """
    query = f"{state['company_name']} company overview, achievements, industry, and impact"
    try:
        response = tavily_client.search(query=query, max_results=3)
        if "results" in response:
            info = "\n".join(item["content"] for item in response["results"])
        else:
            info = "Company information could not be retrieved."
    except Exception as e:
        logging.error(f"Error retrieving company info: {e}")
        info = "Error occurred while retrieving company information."
    return {"company_info": info}

def retrieve_country_info(state: dict) -> dict:
    """
    Retrieve key country information using the Tavily client.
    """
    query = f"{state['country_name']} economic landscape, culture, and business environment"
    try:
        response = tavily_client.search(query=query, max_results=3)
        if "results" in response:
            info = "\n".join(item["content"] for item in response["results"])
        else:
            info = "Country information could not be retrieved."
    except Exception as e:
        logging.error(f"Error retrieving country info: {e}")
        info = "Error occurred while retrieving country information."
    return {"country_info": info}

def retrieve_external_info(state: dict, retriever) -> dict:
    """
    Retrieve additional context information from the Pinecone vector store.
    """
    try:
        company_retrieval = retriever.invoke(f"{state['company_name']} company background")
        country_retrieval = retriever.invoke(f"{state['country_name']} country economic landscape")
        retrieved_info = company_retrieval + country_retrieval
    except Exception as e:
        logging.error(f"Error retrieving external info: {e}")
        retrieved_info = []
    return {"retrieved_info": retrieved_info}

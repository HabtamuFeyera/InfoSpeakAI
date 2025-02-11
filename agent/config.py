import os
from dotenv import load_dotenv

load_dotenv()

# API Keys and other settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
INDEX_NAME = "promptwithrag"

if not all([OPENAI_API_KEY, PINECONE_API_KEY, TAVILY_API_KEY]):
    raise ValueError("Missing one or more required API keys. Please check your environment variables.")

import logging
from langchain_openai import ChatOpenAI
from agent.config import OPENAI_API_KEY

llm_model = ChatOpenAI(model="gpt-4", temperature=0)

SPEECH_GENERATION_PROMPT = """
Create a compelling and well-structured speech using the following information:

Company Information:
{company_info}

Country Information:
{country_info}

The speech should be engaging, informative, and suitable for an invited guest.
"""

SPEECH_REVIEW_PROMPT = """
Review the following speech and provide constructive feedback for improvements.
"""

def generate_speech(state: dict) -> dict:
    """
    Generate a draft speech using the LLM based on collected company and country information.
    """
    prompt = SPEECH_GENERATION_PROMPT.format(
        company_info=state["company_info"],
        country_info=state["country_info"]
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Generate the speech."},
    ]
    try:
        response = llm_model(messages)
        draft_speech = response.content
    except Exception as e:
        logging.error(f"Error generating speech: {e}")
        draft_speech = "Error occurred during speech generation."
    return {"draft_speech": draft_speech, "revision_number": state["revision_number"] + 1}

def review_speech(state: dict) -> dict:
    """
    Review the generated speech and provide constructive feedback.
    """
    messages = [
        {"role": "system", "content": SPEECH_REVIEW_PROMPT},
        {"role": "user", "content": state["draft_speech"]},
    ]
    try:
        response = llm_model(messages)
        speech_review = response.content
    except Exception as e:
        logging.error(f"Error reviewing speech: {e}")
        speech_review = "Error occurred during speech review."
    return {"speech_review": speech_review}

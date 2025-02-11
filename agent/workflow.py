import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from agent.speech import generate_speech, review_speech
from agent.retrieval import retrieve_company_info, retrieve_country_info, retrieve_external_info

def should_continue(state: dict) -> dict:
    """
    Determine whether to continue with another revision of the speech.
    """
    if state["revision_number"] > state["max_revisions"]:
        return {"next": END}
    else:
        return {"next": "generate_speech"}

def build_workflow_graph(agent_state_type) -> StateGraph:
    """
    Build and compile the workflow graph for the AI agent.
    """
    builder = StateGraph(agent_state_type)
    builder.add_node("fetch_company_info", retrieve_company_info)
    builder.add_node("fetch_country_info", retrieve_country_info)
    builder.add_node("retrieve_external_info", retrieve_external_info)
    builder.add_node("generate_speech", generate_speech)
    builder.add_node("review_speech", review_speech)
    builder.add_node("should_continue", should_continue)

    # Define the workflow sequence.
    builder.set_entry_point("fetch_company_info")
    builder.add_edge("fetch_company_info", "fetch_country_info")
    builder.add_edge("fetch_country_info", "retrieve_external_info")
    builder.add_edge("retrieve_external_info", "generate_speech")
    builder.add_edge("generate_speech", "review_speech")
    builder.add_edge("review_speech", "should_continue")

    # Enable checkpointing for state persistence.
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    return graph

import streamlit as st
import logging
from agent.workflow import build_workflow_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

st.title("AI Agent: Information Retrieval & Speech Generation")
st.markdown(
    """
    This application autonomously gathers company and country information and then generates an engaging, informative speech for an invited guest.
    """
)

# User inputs
company_name = st.text_input("Enter Company Name", "Microsoft Corporation")
country_name = st.text_input("Enter Country Name", "United States of America")
max_revisions = st.number_input("Maximum Revisions", min_value=1, max_value=10, value=3, step=1)

if st.button("Generate Speech"):
    with st.spinner("Generating speech..."):
        # Define the initial state for the agent.
        initial_state = {
            "company_name": company_name,
            "country_name": country_name,
            "company_info": "",
            "country_info": "",
            "draft_speech": "",
            "revision_number": 1,
            "max_revisions": max_revisions,
            "retrieved_info": []
        }
        try:
            # Build the workflow graph; here we pass `dict` as the type for the agent state.
            workflow_graph = build_workflow_graph(dict)
            output = workflow_graph.invoke(initial_state, config={"configurable": {"thread_id": "streamlit_session"}})
            st.success("Speech generation completed!")
            st.subheader("Generated Speech")
            st.write(output.get("draft_speech", "No speech generated."))
        except Exception as e:
            st.error(f"An error occurred: {e}")

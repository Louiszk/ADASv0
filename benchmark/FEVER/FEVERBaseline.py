# Imports
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List, Any, TypedDict
from agentic_system.large_language_model import LargeLanguageModel
import re


def build_system():
    # Define state attributes for the system
    class AgentState(TypedDict):
        messages: List[Any]
        claim: str
        prediction: str

    # Initialize graph with state
    graph = StateGraph(AgentState)

    # ===== Node Definitions =====
    def agent_node(state):
        llm = LargeLanguageModel(temperature=0.2)
        system_prompt = """
            You will evaluate factual claims.
            
            For your analysis, please classify the given claim into one of these categories:
            - SUPPORTS: The claim is supported by factual evidence
            - REFUTES: The claim contradicts factual evidence
            - NOT ENOUGH INFO: There is insufficient evidence to determine if the claim is supported or refuted
            
            Write your final prediction in the last line using exactly one of these three labels: SUPPORTS, REFUTES, or NOT ENOUGH INFO.
        """

        full_messages = [SystemMessage(content=system_prompt), HumanMessage(content=state["claim"])]
        response = llm.invoke(full_messages)
        
        response_text = response.content
        print(response_text[-50:])
        
        # Try to extract the prediction from the response
        labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
        final_answer = "NOT ENOUGH INFO" 
        
        last_line = response_text[-20:]
        
        for label in labels:
            if label in last_line:
                final_answer = label
                break
        
        # If not found in the last line, search the entire response
        if final_answer == "NOT ENOUGH INFO" and "NOT ENOUGH INFO" not in last_line:
            final_answer = "NO ANSWER"
        
        new_state = state.copy()
        new_state["prediction"] = final_answer
        
        return new_state

    graph.add_node("FEVERAgent", agent_node)

    # ===== Entry/Exit Configuration =====
    graph.set_entry_point("FEVERAgent")
    graph.set_finish_point("FEVERAgent")

    # ===== Compilation =====
    workflow = graph.compile()
    return workflow, {}
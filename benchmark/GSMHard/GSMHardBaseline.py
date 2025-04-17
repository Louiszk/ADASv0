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
        problem: str
        solution: str

    # Initialize graph with state
    graph = StateGraph(AgentState)

    # ===== Node Definitions =====
    def agent_node(state):
        llm = LargeLanguageModel(temperature=0.2)
        system_prompt = """
            You will solve math problems.
            
            Write your final answer in the last line without thousands separator.
        """

        full_messages = [SystemMessage(content=system_prompt), HumanMessage(content=state["problem"])]
        response = llm.invoke(full_messages)
        
        response_text = response.content
        print(response_text[-20:])
        
        # Try to find an explicit answer line first
        numbers = re.findall(r"\s*(-?\d+(?:\.\d+)?)", response_text)
        final_answer = numbers[-1] if numbers else "No answer found"
        
        new_state = state.copy()
        new_state["solution"] = final_answer
        
        return new_state

    graph.add_node("GSMHardAgent", agent_node)

    # ===== Entry/Exit Configuration =====
    graph.set_entry_point("GSMHardAgent")
    graph.set_finish_point("GSMHardAgent")

    # ===== Compilation =====
    workflow = graph.compile()
    return workflow, {}
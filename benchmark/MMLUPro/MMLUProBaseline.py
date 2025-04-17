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
        question: str
        options: List[str]
        solution: str

    # Initialize graph with state
    graph = StateGraph(AgentState)

    # ===== Node Definitions =====
    def agent_node(state):
        llm = LargeLanguageModel(temperature=0.2)
        system_prompt = """
            You will solve multiple-choice questions in computer science.
            
            Each question comes with a list of options. Select the best option that answers the question.
            
            Write your final answer only as a single letter (A, B, C, ..., J) on the last line.
            e.g. The answer is X
        """

        question = state["question"]
        options = state["options"]
        
        # Create a formatted problem text with options
        formatted_options = ""
        for i, option in enumerate(options):
            if option:  # Check if option is not empty
                option_letter = chr(65 + i)  # Convert 0 -> A, 1 -> B, etc.
                formatted_options += f"{option_letter}. {option}\n"
        
        problem_text = f"{question}\n\n{formatted_options}"
        
        full_messages = [SystemMessage(content=system_prompt), HumanMessage(content=problem_text)]
        response = llm.invoke(full_messages)
        
        response_text = response.content
        print(response_text[-20:])
        
        # Extract the answer (should be a single letter A-J)
        matches = list(re.finditer(r'(?<![A-Za-z])([A-J])(?![A-Za-z])', response_text.strip()[-40:]))
        final_answer = matches[-1].group(1) if matches else "No answer found"
        
        new_state = state.copy()
        new_state["solution"] = final_answer
        
        return new_state

    graph.add_node("MMLUProAgent", agent_node)

    # ===== Entry/Exit Configuration =====
    graph.set_entry_point("MMLUProAgent")
    graph.set_finish_point("MMLUProAgent")

    # ===== Compilation =====
    workflow = graph.compile()
    return workflow, {}
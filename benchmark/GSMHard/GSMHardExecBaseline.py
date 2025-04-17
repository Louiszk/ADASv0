# Imports
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List, Any, TypedDict
from agentic_system.large_language_model import LargeLanguageModel, execute_tool_calls
import re
import sys
from io import StringIO
from langchain_core.tools import tool

@tool
def execute_python(code: str) -> str:
    """Executes Python code and returns the value of the 'result' variable."""
    local_env = {}
    original_stdout = sys.stdout
    captured_output = StringIO()
    sys.stdout = captured_output
   
    try:
        exec(code, {"__builtins__": __builtins__}, local_env)
        
        if 'result' in local_env:
            return str(local_env['result'])
        else:
            return f"No result variable found. Output: {captured_output.getvalue().strip()}"
            
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        sys.stdout = original_stdout

def build_system():
    # Register the Python execution tool
    tools = {"execute_python": execute_python}
    LargeLanguageModel.register_available_tools(tools)
   
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
        llm.bind_tools(["execute_python"])
       
        system_prompt = """
            You will solve math problems with Python code.
           
            Use the execute_python tool to run your solution.
            Your code MUST define a variable named 'result' with the final answer as a float.
            The solution must be efficient and correct. Do not import external libraries.
        """
       
        full_messages = [SystemMessage(content=system_prompt), HumanMessage(content=state["problem"])]
        response = llm.invoke(full_messages)
       
        # Execute any tool calls in the response
        tool_messages, tool_results = execute_tool_calls(response)
       
        # Use the tool result if available
        if "execute_python" in tool_results:
            solution = tool_results["execute_python"]
        else:
            solution = "No tool call was made"
       
        new_state = state.copy()
        new_state["solution"] = solution
       
        return new_state
   
    graph.add_node("GSMHardExecAgent", agent_node)
   
    # ===== Entry/Exit Configuration =====
    graph.set_entry_point("GSMHardExecAgent")
    graph.set_finish_point("GSMHardExecAgent")
   
    # ===== Compilation =====
    workflow = graph.compile()
   
    return workflow, tools
# Imports
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from typing import Dict, List, Any, Callable, Optional, Union, TypeVar, Generic, Tuple, Set, TypedDict
from agentic_system.large_language_model import LargeLanguageModel, execute_tool_calls

def build_system():
    # Define state attributes for the system
    class AgentState(TypedDict):
        messages: List[Any]

    # Initialize graph with state
    graph = StateGraph(AgentState)

    tools = {}
    # ===== Tool Definitions =====
    # No tools defined yet

    # Register all tools with LargeLanguageModel class
    LargeLanguageModel.register_available_tools(tools)

    # ===== Node Definitions =====
    # No nodes defined yet

    # ===== Edge Definitions =====
    # No edges or conditional edges defined yet

    # ===== Compilation =====
    workflow = graph.compile()
    return workflow, tools
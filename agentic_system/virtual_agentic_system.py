import re

class VirtualAgenticSystem:
    """
    A virtual representation of an agentic system with nodes, tools, and edges.
    This class provides a way to define the structure of an agentic system
    without actually compiling it.
    """
    
    def __init__(self, system_name='Default'):
        self.system_name = system_name
        
        self.nodes = {}  # node_name -> description
        self.node_functions = {}  # node_name -> function implementation
        self.tools = {}  # tool_name -> description
        self.tool_functions = {}  # tool_name -> function implementation
        
        self.edges = []  # list of (source, target) tuples
        self.conditional_edges = {}  # source_node -> {condition: func, path_map: map}
        
        self.entry_point = None
        self.finish_point = None
        self.imports = [
            "from langchain_core.tools import tool",
            "from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage",
            "from typing import Dict, List, Any, Callable, Optional, Union, TypeVar, Generic, Tuple, Set, TypedDict",
            "from agentic_system.large_language_model import LargeLanguageModel, execute_tool_calls"
            ]
        
        self.state_attributes = {"messages": "List[Any]"}
        
    def set_state_attributes(self, attrs):
        self.state_attributes = {"messages": "List[Any]"}
        for name, type_annotation in attrs.items():
            self.state_attributes[name] = type_annotation
        return True
        
    def add_imports(self, import_statement):
        if import_statement not in self.imports:
            self.imports.append(import_statement)

    def set_entry_point(self, entry_node):
        if entry_node not in self.nodes:
            raise ValueError(f"Invalid entry point: Node '{entry_node}' does not exist")
        self.entry_point = entry_node

    def set_finish_point(self, finish_node):
        if finish_node not in self.nodes:
            raise ValueError(f"Invalid finish point: Node {finish_node} does not exist")
        self.finish_point = finish_node

    def create_node(self, name, func, description="", source_code=None):
        if func.__doc__ is None or func.__doc__.strip() == "":
            func.__doc__ = description
        if source_code:
            func._source_code = source_code
            
        self.nodes[name] = description
        self.node_functions[name] = func

        return True

    def edit_node(self, name, func=None, description=None, source_code=None):
        if name not in self.nodes:
            return False
            
        if description:
            self.nodes[name] = description
                
        if func:
            self.node_functions[name] = func
        
        if source_code:
            self.node_functions[name]._source_code = source_code
                
        return True

    def create_tool(self, name, description, func, source_code=None):
        """Create a tool function that can be used by nodes."""
        if func.__doc__ is None or func.__doc__.strip() == "":
            raise ValueError("Tool function must contain a docstring.")
        if source_code:
            func._source_code = source_code
            
        self.tools[name] = description
        self.tool_functions[name] = func

        return True
    
    def edit_tool(self, name, new_function=None, new_description=None, source_code=None):
        """Edit tool properties."""
        if name not in self.tools:
            return False
            
        if new_description:
            self.tools[name] = new_description
                
        if new_function:
            self.tool_functions[name] = new_function
        
        if source_code:
            self.tool_functions[name]._source_code = source_code
                
        return True

    def create_edge(self, source, target):
        """Create a standard edge between nodes."""
        if source not in self.nodes:
            raise ValueError(f"Invalid source node: '{source}' does not exist")
            
        if target not in self.nodes:
            raise ValueError(f"Invalid target node: '{target}' does not exist")
        
        if any(edge_source == source for edge_source, _ in self.edges):
            raise ValueError(f"Source node '{source}' already has an outgoing edge. Parallel processing is disabled.")
        
        self.edges.append((source, target))
        return True

    def create_conditional_edge(self, source, condition, condition_code=None, path_map=None):
        """Create a conditional edge with a router function."""
        if source not in self.nodes:
            raise ValueError(f"Invalid source node: '{source}' does not exist")
        
        if condition_code:
            condition._source_code = condition_code
        
        edge_info = {"condition": condition}
        
        if path_map is not None:
            for target in path_map.values():
                if target not in self.nodes:
                    raise ValueError(f"Invalid target node in path_map: '{target}' does not exist")
            
            edge_info["path_map"] = path_map.copy()
        
        self.conditional_edges[source] = edge_info
        
        return True

    def delete_node(self, name):
        """Delete a node and all associated edges."""
        if name not in self.nodes:
            return False
        
        del self.nodes[name]
        if name in self.node_functions:
            del self.node_functions[name]
        
        if name in self.conditional_edges:
            del self.conditional_edges[name]
        
        self.edges = [(s, t) for s, t in self.edges if s != name and t != name]
        
        if self.entry_point == name:
            self.entry_point = None
        if self.finish_point == name:
            self.finish_point = None
        
        return True

    def delete_edge(self, source, target):
        """Delete a standard edge."""
        edge = (source, target)
        if edge in self.edges:
            self.edges.remove(edge)
            return True
        
        return False

    def delete_conditional_edge(self, source):
        """Delete a conditional edge."""
        if source in self.conditional_edges:
            del self.conditional_edges[source]
            return True
        
        return False
    
    def get_function(self, function_code):
            # if "placeholder" in function_code.lower():
            #     raise ValueError("Do not use placeholder logic. Always implement the full logic.")
            
            match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', function_code)
            if not match:
                return "Error: Could not identify function name in the provided code"
            
            function_name = match.group(1)
            completed_function_code = "\n".join(self.imports) + "\n" + function_code    
            local_vars = {}
            exec(completed_function_code, {"__builtins__": __builtins__}, local_vars)
            
            if function_name in local_vars and callable(local_vars[function_name]):
                new_function = local_vars[function_name]
                return new_function
            else:
                return f"Error: Function '{function_name}' not found after execution"
            
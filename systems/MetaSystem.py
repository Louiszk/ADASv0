# MetaSystem System Configuration
# Total nodes: 2
# Total tools: 14

from langgraph.graph import StateGraph
import os
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from typing import Dict, List, Any, Callable, Optional, Union, TypeVar, Generic, Tuple, Set, TypedDict
from agentic_system.large_language_model import LargeLanguageModel, execute_tool_calls
import json
from tqdm import tqdm
import traceback
import dill as pickle
import re
import io
import contextlib
import sys
import subprocess
from systems import system_prompts
from agentic_system.materialize import materialize_system
target_system = None


def build_system():
    # Define state attributes for the system
    class AgentState(TypedDict):
        messages: List[Any]
        design_completed: bool

    # Initialize graph with state
    graph = StateGraph(AgentState)

    # Tool definitions
    # ===== Tool Definitions =====
    tools = {}
    # Tool: PipInstall
    # Description: Securely installs a Python package using pip
    def pip_install(package_name: str) -> str:
        """
            Securely installs a Python package using pip.
                package_name: Name of the package to install e.g. "langgraph==0.3.5"
        """
    
        # Validate package name to prevent command injection
        valid_pattern = r'^[a-zA-Z0-9._-]+(\s*[=<>!]=\s*[0-9a-zA-Z.]+)?$'
    
        if not re.match(valid_pattern, package_name):
            print(f"Error: Invalid package name format. Package name '{package_name}' contains invalid characters.")
            return None
    
        try:
            process = subprocess.run(
                [sys.executable, "-m", "pip", "install", package_name], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True,
                shell=False
            )
    
            if process.returncode == 0:
                print(f"Successfully installed {package_name}")
            else:
                print(f"Error installing {package_name}:\n{process.stdout}")
    
        except Exception as e:
            print(f"Installation failed: {str(e)}")
    

    tools["PipInstall"] = tool(runnable=pip_install, name_or_callable="PipInstall")

    # Tool: AddImports
    # Description: Adds custom import statements to the target system
    def add_imports(import_statement: str) -> str:
        """
            Adds custom imports to the target system.
                import_statement: A string containing import statements e.g. "from x import y"
        """
        try:
            target_system.add_imports(import_statement.strip())
            print(f"Import statement '{import_statement}' added to target system.")
        except Exception as e:
            print(f"Error adding import: {repr(e)}")
    

    tools["AddImports"] = tool(runnable=add_imports, name_or_callable="AddImports")

    # Tool: SetStateAttributes
    # Description: Sets state attributes with type annotations for the target system
    def set_state_attributes(attributes: Dict[str, str]) -> str:
        """
            Defines state attributes accessible throughout the system. Only defines the type annotations, not the values.
                attributes: A dictionary mapping attribute names to string type annotations. 
                {'messages': 'List[Any]'} is the default and will be set automatically.
        """
        try:
            target_system.set_state_attributes(attributes)
            print(f"State attributes set successfully: {attributes}")
        except Exception as e:
            print(f"Error setting state attributes: {repr(e)}")
    

    tools["SetStateAttributes"] = tool(runnable=set_state_attributes, name_or_callable="SetStateAttributes")

    # Tool: CreateNode
    # Description: Creates a node in the target system with custom function implementation
    def add_node(name: str, description: str, function_code: str) -> str:
        """
            Creates a node in the target system.
                function_code: Python code defining the node's processing function
        """
        try:
            node_function = target_system.get_function(function_code)
    
            target_system.create_node(name, node_function, description, function_code)
            print(f"Node '{name}' created successfully")
        except Exception as e:
            print(f"Error creating node: {repr(e)}")
    

    tools["CreateNode"] = tool(runnable=add_node, name_or_callable="CreateNode")

    # Tool: CreateTool
    # Description: Creates a tool in the target system that can be used by nodes
    def add_tool(name: str, description: str, function_code: str) -> str:
        """
            Creates a tool in the target system that can be bound to agents and invoked by functions.
                function_code: Python code defining the tool's function including type annotations and a clear docstring
        """
        try:
            tool_function = target_system.get_function(function_code)
    
            target_system.create_tool(name, description, tool_function, function_code)
            print(f"Tool '{name}' created successfully")
        except Exception as e:
            print(f"Error creating tool: {repr(e)}")
    

    tools["CreateTool"] = tool(runnable=add_tool, name_or_callable="CreateTool")

    # Tool: EditComponent
    # Description: Edits a node or tool's implementation
    def edit_component(component_type: str, name: str, new_function_code: str, new_description: Optional[str] = None) -> str:
        """
            Modifies an existing node or tool's implementation by providing a new_function_code. This does not allow renaming.
                component_type: Type of component to edit ('node' or 'tool')
                name: Name of the component to edit
                new_function_code: New Python code for the component's function
        """
        try:
            if component_type.lower() not in ["node", "tool"]:
                print(f"Error: Invalid component type '{component_type}'. Must be 'node' or 'tool'.")
                return None
    
            if name not in target_system.nodes and name not in target_system.tools:
                print(f"Error: '{name}' not found")
                return None
    
            new_function = target_system.get_function(new_function_code)
    
            if component_type.lower() == "node":
                if name not in target_system.nodes:
                    print(f"Error: Node '{name}' not found")
                    return None
    
                target_system.edit_node(name, new_function, new_description, new_function_code)
                print(f"Node '{name}' updated successfully")
    
            else:
                if name not in target_system.tools:
                    print(f"Error: Tool '{name}' not found")
                    return None
    
                target_system.edit_tool(name, new_function, new_description, new_function_code)
                print(f"Tool '{name}' updated successfully")
    
        except Exception as e:
            print(f"Error editing {component_type}: {repr(e)}")
    

    tools["EditComponent"] = tool(runnable=edit_component, name_or_callable="EditComponent")

    # Tool: AddEdge
    # Description: Adds an edge between nodes in the target system
    def add_edge(source: str, target: str) -> str:
        """
            Adds an edge between nodes in the target system.
                source: Name of the source node
                target: Name of the target node
        """
        try:
            target_system.create_edge(source, target)
            print(f"Edge from '{source}' to '{target}' added successfully")
        except Exception as e:
            print(f"Error adding edge: {repr(e)}")
    

    tools["AddEdge"] = tool(runnable=add_edge, name_or_callable="AddEdge")

    # Tool: AddConditionalEdge
    # Description: Adds a conditional edge in the target system.
    def add_conditional_edge(source: str, condition_code: str) -> str:
        """
            Adds a conditional edge from a source node.
                source: Name of the source node
                condition_code: Python code for the condition function that returns the target node
        """
        try:
            condition_function = target_system.get_function(condition_code)
    
            # Extract potential node names from string literals in the code (better visualization)
            string_pattern = r"['\"]([^'\"]*)['\"]"
            potential_nodes = set(re.findall(string_pattern, condition_code))
    
            path_map = None
            auto_path_map = {}
            for node_name in potential_nodes:
                if node_name in target_system.nodes:
                    auto_path_map[node_name] = node_name
    
            if auto_path_map:
                path_map = auto_path_map
    
            target_system.create_conditional_edge(
                source = source, 
                condition = condition_function,
                condition_code = condition_code,
                path_map = path_map
            )
    
            result = f"Conditional edge from '{source}' added successfully"
            if path_map:
                result += f" with path map to {list(path_map.values())}"
    
            print(result)
        except Exception as e:
            print(f"Error adding conditional edge: {repr(e)}")
    

    tools["AddConditionalEdge"] = tool(runnable=add_conditional_edge, name_or_callable="AddConditionalEdge")

    # Tool: SetEndpoints
    # Description: Sets the entry point and/or finish point of the workflow
    def set_endpoints(entry_point: str = None, finish_point: str = None) -> str:
        """
            Sets the entry point (start node) and/or finish point (end node) of the workflow.
                entry_point: Name of the node to set as entry point
                finish_point: Name of the node to set as finish point
        """
        results = []
    
        if entry_point is not None:
            try:
                target_system.set_entry_point(entry_point)
                results.append(f"Entry point set to '{entry_point}' successfully")
            except Exception as e:
                results.append(f"Error setting entry point: {repr(e)}")
    
        if finish_point is not None:
            try:
                target_system.set_finish_point(finish_point)
                results.append(f"Finish point set to '{finish_point}' successfully")
            except Exception as e:
                results.append(f"Error setting finish point: {repr(e)}")
    
        if not results:
            print("No endpoints were specified. Please provide entry_point and/or finish_point.")
            return None
    
        print("\n".join(results))
    

    tools["SetEndpoints"] = tool(runnable=set_endpoints, name_or_callable="SetEndpoints")

    # Tool: TestSystem
    # Description: Tests the target system with a given state
    def test_system(state: Dict[str, Any]) -> str:
        """
            Executes the current system with a test input state to validate functionality.
                state: A python dictionary with state attributes e.g. {'messages': ['Test Input'], 'attr2': [3, 5]}
        """
        all_outputs = []
        error_message = ""
    
        try:
            if not (target_system.entry_point and target_system.finish_point):
                raise Exception("You must set an entry point and finish point before testing")
    
            source_code = materialize_system(target_system, None)
            namespace = {}
            exec(source_code, namespace, namespace)
    
            if 'build_system' not in namespace:
                raise Exception("Could not find build_system function in generated code")
    
            target_workflow, _ = namespace['build_system']()
            pbar = tqdm(desc="Testing the System")
    
            for output in target_workflow.stream(state, config={"recursion_limit": 20}):
                cleaned_messages = []
                if "messages" in output:
                    for message in output["messages"]:
                        cleaned_message = getattr(message, 'type', 'Unknown') + ": "
                        if hasattr(message, 'content') and message.content:
                            cleaned_message += message.content
                        if hasattr(message, 'tool_calls') and message.tool_calls:
                            cleaned_message += str(message.tool_calls)
                        if not hasattr(message, 'content') and not hasattr(message, 'tool_calls'):
                            cleaned_message += str(message)
                        cleaned_messages.append(cleaned_message)
                    output["messages"] = cleaned_messages
                all_outputs.append(output)
                pbar.update(1)
    
            pbar.close()
    
        except Exception as e:
            error_message = f"\n\n Error while testing the system:\n{str(e)}"
    
        result = all_outputs if all_outputs else {}
    
        test_result = f"Test completed.\n <TestResults>\n{result}\n</TestResults>"
    
        print(test_result)
    
        if error_message:
            raise Exception(error_message)
    

    tools["TestSystem"] = tool(runnable=test_system, name_or_callable="TestSystem")

    # Tool: DeleteNode
    # Description: Deletes a node and all its associated edges from the target system
    def delete_node(node_name: str) -> str:
        """
            Deletes a node and all its associated edges.
                node_name: Name of the node to delete
        """
        try:
            result = target_system.delete_node(node_name)
            print(f"Node '{node_name}' deleted successfully" if result else f"Failed to delete node '{node_name}'")
        except Exception as e:
            print(f"Error deleting node: {repr(e)}")
    

    tools["DeleteNode"] = tool(runnable=delete_node, name_or_callable="DeleteNode")

    # Tool: DeleteEdge
    # Description: Deletes an edge between nodes in the target system
    def delete_edge(source: str, target: str) -> str:
        """
            Deletes an edge between nodes.
                source: Name of the source node
                target: Name of the target node
        """
        try:
            result = target_system.delete_edge(source, target)
            print(f"Edge from '{source}' to '{target}' deleted successfully" if result else f"No such edge from '{source}' to '{target}'")
        except Exception as e:
            print(f"Error deleting edge: {repr(e)}")
    

    tools["DeleteEdge"] = tool(runnable=delete_edge, name_or_callable="DeleteEdge")

    # Tool: DeleteConditionalEdge
    # Description: Deletes a conditional edge from a source node
    def delete_conditional_edge(source: str) -> str:
        """
            Deletes a conditional edge from a source node.
                source: Name of the source node
        """
        try:
            result = target_system.delete_conditional_edge(source)
            print(f"Conditional edge from '{source}' deleted successfully" if result else f"No conditional edge found from '{source}'")
        except Exception as e:
            print(f"Error deleting conditional edge: {repr(e)}")
    

    tools["DeleteConditionalEdge"] = tool(runnable=delete_conditional_edge, name_or_callable="DeleteConditionalEdge")

    # Tool: EndDesign
    # Description: Finalizes the system design process
    def end_design() -> str:
        """
            Finalizes the system design process.
        """
        try:
            if not (target_system.entry_point and target_system.finish_point):
                print("Error finalizing system: You must set an entry point and finish point before finalizing")
                return None
    
            code_dir = "sandbox/workspace/automated_systems"
            materialize_system(target_system, code_dir)
            print(f"System code materialized to {code_dir}")
    
            pickle_name = target_system.system_name.replace("/", "_").replace("\\", "_").replace(":", "_") + ".pkl"
            pickle_path = os.path.join(code_dir, pickle_name)
            with open(pickle_path, 'wb') as f:
                pickle.dump(target_system, f)
            print(f"System pickled to {pickle_path}")
    
            print("Design process completed successfully.")
        except Exception as e:
            print(f"Error finalizing system: {repr(e)}")
    

    tools["EndDesign"] = tool(runnable=end_design, name_or_callable="EndDesign")

    # Register tools with LargeLanguageModel class
    LargeLanguageModel.register_available_tools(tools)
    # ===== Node Definitions =====
    # Node: MetaAgent
    # Description: Meta Agent
    def meta_agent_function(state: Dict[str, Any]) -> Dict[str, Any]:  
        llm = LargeLanguageModel(temperature=0.2, wrapper="blablador", model_name="alias-fast-experimental")
        context_length = 8*2 # even
        messages = state.get("messages", [])
        iteration = len([msg for msg in messages if isinstance(msg, AIMessage)])
        initial_messages, current_messages = messages[:2], messages[2:]
        last_messages = current_messages[-context_length:] if len(current_messages) >= context_length else current_messages
    
        code_message = f"(Iteration {iteration}) Current Code:\n" + materialize_system(target_system, output_dir=None)
    
        full_messages = [SystemMessage(content=system_prompts.meta_agent)] + initial_messages + last_messages + [HumanMessage(content=code_message)]
        response = llm.invoke(full_messages)
    
        # Extract tool calls
        response_content = response.content
    
        # Check for tool calls and execute them
        design_completed = False
        tool_results = []
    
        # Find all tool calls
        tool_calls_pattern = r"```tool_calls\n(.*?)```end"
        tool_calls_matches = re.findall(tool_calls_pattern, response_content, re.DOTALL)
    
        # Define the available tools in a namespace
        tools_namespace = {
            "set_state_attributes": set_state_attributes,
            "pip_install": pip_install,
            "add_imports": add_imports,
            "add_node": add_node,
            "add_tool": add_tool,
            "edit_component": edit_component,
            "add_edge": add_edge,
            "add_conditional_edge": add_conditional_edge,
            "delete_conditional_edge": delete_conditional_edge,
            "set_endpoints": set_endpoints,
            "test_system": test_system,
            "delete_node": delete_node,
            "delete_edge": delete_edge,
            "end_design": end_design
        }
    
        for tool_call in tool_calls_matches:
            try:
                # Capture stdout to get tool execution results
                string_io = io.StringIO()
                with contextlib.redirect_stdout(string_io):
                    local_namespace = dict(tools_namespace)
    
                    exec(tool_call, globals(), local_namespace)
    
                output = string_io.getvalue().strip()
    
                if "Design process completed" in output:
                    design_completed = True
    
                tool_results.append(output or "Tool call executed successfully.")
            except Exception as e:
                output = string_io.getvalue().strip()
                error_message = f"\nError executing tool call: {repr(e)}"
                tool_results.append(output + error_message)
                break
    
        if tool_results:
            tool_output = "\n\n".join(tool_results)
            tool_response = f"\n\nTool Execution Results:\n{tool_output}"
        else: 
            tool_response = "You made no tool calls. Maybe you forget to wrap the tool calls inside ```tool_calls\n```end"
    
        tool_message = HumanMessage(content=tool_response)
        updated_messages = messages + [response, tool_message]
    
        new_state = {"messages": updated_messages, "design_completed": design_completed}
        return new_state
    

    graph.add_node("MetaAgent", meta_agent_function)

    # Node: EndDesign
    # Description: Terminal node for workflow completion
    def end_design_node(state: Dict[str, Any]) -> Dict[str, Any]:
        return state
    

    graph.add_node("EndDesign", end_design_node)

    # ===== Conditional Edges =====
    # Conditional Router from: MetaAgent
    def design_completed_router(state: Dict[str, Any]) -> str:
        """Routes to EndDesign if design is completed, otherwise back to MetaAgent."""
        if state.get("design_completed", False):
            return "EndDesign"
        return "MetaAgent"
    

    graph.add_conditional_edges("MetaAgent", design_completed_router, {'MetaAgent': 'MetaAgent', 'EndDesign': 'EndDesign'})

    # ===== Entry/Exit Configuration =====
    graph.set_entry_point("MetaAgent")
    graph.set_finish_point("EndDesign")

    # ===== Compilation =====
    workflow = graph.compile()
    return workflow, tools

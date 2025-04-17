# MetaSystem System Configuration
# Total nodes: 2
# Total tools: 14

from langgraph.graph import StateGraph
import os
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from typing import Dict, List, Any, Callable, Optional, Union, TypeVar, Generic, Tuple, Set, TypedDict
from agentic_system.large_language_model import LargeLanguageModel, execute_decorator_tool_calls
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
from langchain_core.messages import trim_messages
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
            return f"Error: Invalid package name format. Package name '{package_name}' contains invalid characters."
    
        try:
            process = subprocess.run(
                [sys.executable, "-m", "pip", "install", package_name], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True,
                shell=False
            )
    
            if process.returncode == 0:
                return f"Successfully installed {package_name}"
            else:
                return f"Error installing {package_name}:\n{process.stdout}"
    
        except Exception as e:
            return f"Installation failed: {str(e)}"
    

    tools["PipInstall"] = tool(runnable=pip_install, name_or_callable="PipInstall")

    # Tool: SetImports
    # Description: Sets the list of necessary import statements for the target system, replacing existing custom imports.
    def set_imports(import_statements: List[str]) -> str:
        """
            Sets the list of import statements for the target system. This replaces any existing imports.
                import_statements: A list of strings, where each string is a complete import statement (e.g., ['import os', 'from typing import List']).
        """
    
        try:
            # Basic validation for each statement
            for stmt in import_statements:
                if not isinstance(stmt, str) or not (stmt.startswith("import ") or stmt.startswith("from ")):
                    return f"Error: Invalid import statement format: '{stmt}'. Must start with 'import' or 'from'."
    
            # Always keep the mandatory base imports
            base_imports = [
                "from langchain_core.tools import tool",
                "from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage",
                "from typing import Dict, List, Any, Callable, Optional, Union, TypeVar, Generic, Tuple, Set, TypedDict",
                "from agentic_system.large_language_model import LargeLanguageModel, execute_tool_calls"
            ]
            # Use a set to avoid duplicates and preserve order for non-base imports
            final_imports = base_imports + sorted(list(set(stmt.strip() for stmt in import_statements if stmt.strip() not in base_imports)))
    
            target_system.imports = final_imports
            return f"Import statements set successfully for target system. Total imports: {len(target_system.imports)}."
        except Exception as e:
            return f"Error setting imports: {repr(e)}"
    

    tools["SetImports"] = tool(runnable=set_imports, name_or_callable="SetImports")

    # Tool: SetStateAttributes
    # Description: Sets state attributes with type annotations for the target system
    def set_state_attributes(attributes: Dict[str, str]) -> str:
        """
            Defines state attributes accessible throughout the system. Only defines the type annotations, not the values.
                attributes: A python dictionary mapping attribute names to string type annotations. 
                {"messages": "List[Any]"} is the default and will be set automatically.
        """
        try:
            target_system.set_state_attributes(attributes)
            return f"State attributes set successfully: {attributes}"
        except Exception as e:
            return f"Error setting state attributes: {repr(e)}"
    

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
            return f"Node '{name}' created successfully"
        except Exception as e:
            return f"Error creating node: {repr(e)}"
    

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
            return f"Tool '{name}' created successfully"
        except Exception as e:
            return f"Error creating tool: {repr(e)}"
    

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
                return f"Error: Invalid component type '{component_type}'. Must be 'node' or 'tool'."
    
            if name not in target_system.nodes and name not in target_system.tools:
                return f"Error: '{name}' not found"
    
            new_function = target_system.get_function(new_function_code)
    
            if component_type.lower() == "node":
                if name not in target_system.nodes:
                    return f"Error: Node '{name}' not found"
    
                target_system.edit_node(name, new_function, new_description, new_function_code)
                return f"Node '{name}' updated successfully"
    
            else:
                if name not in target_system.tools:
                    return f"Error: Tool '{name}' not found"
    
                target_system.edit_tool(name, new_function, new_description, new_function_code)
                return f"Tool '{name}' updated successfully"
    
        except Exception as e:
            return f"Error editing {component_type}: {repr(e)}"
    

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
            return f"Edge from '{source}' to '{target}' added successfully"
        except Exception as e:
            return f"Error adding edge: {repr(e)}"
    

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
    
            return result
        except Exception as e:
            return f"Error adding conditional edge: {repr(e)}"
    

    tools["AddConditionalEdge"] = tool(runnable=add_conditional_edge, name_or_callable="AddConditionalEdge")

    # Tool: SetEndpoints
    # Description: Sets the entry point and/or finish point of the workflow
    def set_endpoints(entry_point: str = None, finish_point: str = None) -> str:
        """
            Sets the entry point (start node) and/or finish point (end node) of the workflow.
                entry_point: Name of the node to set as entry point; equivalent to graph.add_edge(START, entry_point)
                finish_point: Name of the node to set as finish point; equivalent to graph.add_edge(finish_point, END)
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
            return "No endpoints were specified. Please provide entry_point and/or finish_point."
    
        return "\n".join(results)
    

    tools["SetEndpoints"] = tool(runnable=set_endpoints, name_or_callable="SetEndpoints")

    # Tool: TestSystem
    # Description: Tests the target system with a given state
    def test_system(state: Dict[str, Any]) -> str:
        """
            Executes the current system with a test input state to validate functionality.
                state: A python dictionary with state attributes e.g. {"messages": ["Test Input"], "attr2": [3, 5]}
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
    
        result = "\n".join([f"State {i}: " + str(out) for i, out in enumerate(all_outputs)]) if all_outputs else {}
    
        test_result = f"Test completed.\n <TestResults>\n{result}\n</TestResults>"
    
        if error_message:
            raise Exception(error_message)
        else:
            return test_result
    

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
            return f"Node '{node_name}' deleted successfully" if result else f"Failed to delete node '{node_name}'"
        except Exception as e:
            return f"Error deleting node: {repr(e)}"
    

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
            return f"Edge from '{source}' to '{target}' deleted successfully" if result else f"No such edge from '{source}' to '{target}'"
        except Exception as e:
            return f"Error deleting edge: {repr(e)}"
    

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
            return f"Conditional edge from '{source}' deleted successfully" if result else f"No conditional edge found from '{source}'"
        except Exception as e:
            return f"Error deleting conditional edge: {repr(e)}"
    

    tools["DeleteConditionalEdge"] = tool(runnable=delete_conditional_edge, name_or_callable="DeleteConditionalEdge")

    # Tool: EndDesign
    # Description: Finalizes the system design process
    def end_design() -> str:
        """
            Finalizes the system design process.
        """
        try:
            if not (target_system.entry_point and target_system.finish_point):
                return "Error finalizing system: You must set an entry point and finish point before finalizing"
    
            code_dir = "sandbox/workspace/automated_systems"
            materialize_system(target_system, code_dir)
            print(f"System code materialized to {code_dir}")
    
            pickle_name = target_system.system_name.replace("/", "_").replace("\\", "_").replace(":", "_") + ".pkl"
            pickle_path = os.path.join(code_dir, pickle_name)
            with open(pickle_path, 'wb') as f:
                pickle.dump(target_system, f)
            print(f"System pickled to {pickle_path}")
    
            return "Ending the design process..."
        except Exception as e:
            error_msg = f"Error finalizing system: {repr(e)}"
            print(error_msg)
            return error_msg
    

    tools["EndDesign"] = tool(runnable=end_design, name_or_callable="EndDesign")

    # Register tools with LargeLanguageModel class
    LargeLanguageModel.register_available_tools(tools)
    # ===== Node Definitions =====
    # Node: MetaAgent
    # Description: Meta Agent
    def meta_agent_function(state: Dict[str, Any]) -> Dict[str, Any]:  
        llm = LargeLanguageModel(temperature=0.2, wrapper="google", model_name="gemini-2.0-flash")

        context_length = 8*2 # even
        messages = state.get("messages", [])
        iteration = len([msg for msg in messages if isinstance(msg, AIMessage)])
        initial_message, current_messages = messages[0], messages[1:]
        try:
            trimmed_messages = trim_messages(
                current_messages,
                max_tokens=context_length,
                strategy="last",
                token_counter=len,
                allow_partial=False
            )
        except Exception as e:
            print(f"Error during message trimming: {e}")

        code_message = f"(Iteration {iteration}) Current Code:\n" + materialize_system(target_system, output_dir=None)
    
        full_messages = [SystemMessage(content=system_prompts.meta_agent), initial_message] + trimmed_messages + [HumanMessage(content=code_message)]
        print([getattr(last_msg, 'type', 'Unknown') for last_msg in full_messages])
        response = llm.invoke(full_messages)

        if not hasattr(response, 'content') or not response.content:
            response.content = "I will call the necessary tools."
        
        # Check for decorator-style tool calls
        tool_messages, tool_results, _ = execute_decorator_tool_calls(response.content)

        updated_messages = messages + [response]
        if tool_messages:
            updated_messages.extend(tool_messages)
        else:
            updated_messages.append(HumanMessage(content="You made no valid function calls. Remember to use the @@decorator_name() syntax."))

            
        # Ending the design if the last test ran without errors (this does not check accuracy)
        design_completed = False
        if tool_results and 'EndDesign' in tool_results and "Ending the design process" in str(tool_results['EndDesign']):
            test_passed_recently = False
            search_start_index = max(0, len(messages) - 6)
            for msg in reversed(updated_messages[search_start_index:]):
                if isinstance(msg, HumanMessage) and hasattr(msg, 'content'):
                    if "Test completed." in msg.content:
                        test_passed_recently = True
                        break
                    elif "Error while testing the system" in msg.content:
                        test_passed_recently = False
                        break

            if test_passed_recently or iteration >= 58:
                design_completed = True
            else:
                for i, tm in enumerate(tool_messages):
                    if tm.name == 'EndDesign':
                        tm.content += "Error: Cannot finalize design. Please run successful tests using TestSystem first."
    
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

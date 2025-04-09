import re
import os
import sys
import dill as pickle
import inspect
from tqdm import tqdm
import subprocess
import io
import json
import contextlib
from typing import Dict, List, Any, Optional, Union
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage, trim_messages
from agentic_system.large_language_model import LargeLanguageModel, execute_tool_calls
from agentic_system.virtual_agentic_system import VirtualAgenticSystem
from agentic_system.materialize import materialize_system
from systems import system_prompts

def create_meta_system():
    print(f"\n----- Creating Meta System -----\n")
    
    meta_system = VirtualAgenticSystem("MetaSystem")
    
    # another approach would be to have the target_system in the state, but we might run into serialization issues
    target_system = None

    meta_system.set_state_attributes({"design_completed": "bool"})
    meta_system.add_imports("import json")
    meta_system.add_imports("from tqdm import tqdm")
    meta_system.add_imports("import traceback")
    meta_system.add_imports("import dill as pickle")
    meta_system.add_imports("import re")
    meta_system.add_imports("import io")
    meta_system.add_imports("import contextlib")
    meta_system.add_imports("import sys")
    meta_system.add_imports("import subprocess")
    meta_system.add_imports("from systems import system_prompts")
    meta_system.add_imports("from langchain_core.messages import trim_messages")
    meta_system.add_imports("from agentic_system.materialize import materialize_system")
    meta_system.add_imports("target_system = None") # global variable
    
    # --- Tool Functions ---
    
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
    
    meta_system.create_tool(
        "PipInstall",
        "Securely installs a Python package using pip",
        pip_install
    )
    
    def add_imports(import_statement: str) -> str:
        """
            Adds custom imports to the target system.
                import_statement: A string containing import statements e.g. "from x import y"
        """
        try:
            target_system.add_imports(import_statement.strip())
            return f"Import statement '{import_statement}' added to target system."
        except Exception as e:
            return f"Error adding import: {repr(e)}"
    
    meta_system.create_tool(
        "AddImports",
        "Adds custom import statements to the target system",
        add_imports
    )
    
    def set_state_attributes(attributes: str) -> str:
        """
            Defines state attributes accessible throughout the system. Only defines the type annotations, not the values.
                attributes: A json string mapping attribute names to string type annotations. 
                '{"messages": "List[Any]"}' is the default and will be set automatically.
        """
        try:
            attributes = json.loads(attributes)
            target_system.set_state_attributes(attributes)
            return f"State attributes set successfully: {attributes}"
        except Exception as e:
            return f"Error setting state attributes: {repr(e)}"
    
    meta_system.create_tool(
        "SetStateAttributes",
        "Sets state attributes with type annotations for the target system",
        set_state_attributes
    )
    
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
    
    meta_system.create_tool(
        "CreateNode",
        "Creates a node in the target system with custom function implementation",
        add_node
    )
    
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
    
    meta_system.create_tool(
        "CreateTool",
        "Creates a tool in the target system that can be used by nodes",
        add_tool
    )
    
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
        
    meta_system.create_tool(
        "EditComponent",
        "Edits a node or tool's implementation",
        edit_component
    )

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
    
    meta_system.create_tool(
        "AddEdge",
        "Adds an edge between nodes in the target system",
        add_edge
    )
    
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
    
    meta_system.create_tool(
        "AddConditionalEdge",
        "Adds a conditional edge in the target system.",
        add_conditional_edge
    )

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
            return "No endpoints were specified. Please provide entry_point and/or finish_point."
        
        return "\n".join(results)

    meta_system.create_tool(
        "SetEndpoints",
        "Sets the entry point and/or finish point of the workflow",
        set_endpoints
    )
    
    
    def test_system(state: str) -> str:
        """
            Executes the current system with a test input state to validate functionality.
                state: A json string with state attributes e.g. '{"messages": ["Test Input"], "attr2": [3, 5]}'
        """
        all_outputs = []
        error_message = ""
        state = json.loads(state)

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

        if error_message:
            raise Exception(error_message)
        else:
            return test_result

    meta_system.create_tool(
        "TestSystem",
        "Tests the target system with a given state",
        test_system
    )
    
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
    
    meta_system.create_tool(
        "DeleteNode",
        "Deletes a node and all its associated edges from the target system",
        delete_node
    )
    
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
    
    meta_system.create_tool(
        "DeleteEdge",
        "Deletes an edge between nodes in the target system",
        delete_edge
    )

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
    
    meta_system.create_tool(
        "DeleteConditionalEdge",
        "Deletes a conditional edge from a source node",
        delete_conditional_edge
    )
    
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
            
            return "Design process completed successfully."
        except Exception as e:
            error_msg = f"Error finalizing system: {repr(e)}"
            print(error_msg)
            return error_msg
    
    meta_system.create_tool(
        "EndDesign",
        "Finalizes the system design process",
        end_design
    )

    tools = {}
    
    def meta_agent_function(state: Dict[str, Any]) -> Dict[str, Any]:  
        llm = LargeLanguageModel(temperature=0.2, wrapper="google", model_name="gemini-2.0-flash")
        llm.bind_tools(list(tools.keys()))

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

        tool_messages, tool_results = execute_tool_calls(response)
        
        design_completed = False
        if tool_results and 'EndDesign' in tool_results and "Design process completed successfully" in str(tool_results['EndDesign']):
            design_completed = True

        updated_messages = messages + [response]
        if tool_messages:
            updated_messages.extend(tool_messages)
        else:
            updated_messages.append(HumanMessage(content="You made no tool calls."))
        
        new_state = {"messages": updated_messages, "design_completed": design_completed}
        return new_state
    
    def end_design_node(state: Dict[str, Any]) -> Dict[str, Any]:
        return state

    meta_system.create_node("MetaAgent", meta_agent_function, "Meta Agent")
    meta_system.create_node("EndDesign", end_design_node, "Terminal node for workflow completion")
    
    meta_system.set_entry_point("MetaAgent")
    meta_system.set_finish_point("EndDesign")
    
    def design_completed_router(state: Dict[str, Any]) -> str:
        """Routes to EndDesign if design is completed, otherwise back to MetaAgent."""
        if state.get("design_completed", False):
            return "EndDesign"
        return "MetaAgent"
    
    meta_system.create_conditional_edge(
        source="MetaAgent",
        condition=design_completed_router,
        condition_code=None,
        path_map={
            "MetaAgent": "MetaAgent",
            "EndDesign": "EndDesign"
        }
    )
    
    materialize_system(meta_system)
    print("----- Materialized Meta System -----")

if __name__ == "__main__":
    create_meta_system()

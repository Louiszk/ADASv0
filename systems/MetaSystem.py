# MetaSystem System Configuration
# Total nodes: 1
# Total tools: 4

from langgraph.graph import StateGraph
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, trim_messages
from typing import Dict, List, Any, Callable, Optional, Union, TypeVar, Generic, Tuple, Set, TypedDict
from agentic_system.large_language_model import LargeLanguageModel, execute_decorator_tool_calls
from tqdm import tqdm
import traceback
import re
import io
import json
import contextlib
import sys
import subprocess
from systems import system_prompts

# Target system file path
target_system_file = "/sandbox/workspace/automated_systems/target_system.py"
target_system_name = "DefaultSystem"

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

    # Tool: TestSystem
    # Description: Tests the target system with a given state
    def test_system(state: Dict[str, Any]) -> str:
        """
            Executes the current system with a test input state to validate functionality.
                state: A python dictionary with state attributes e.g. '{"messages": ["Test Input"], "attr2": [3, 5]}'
        """
        all_outputs = []
        error_message = None
    
        try:
            with open(target_system_file, 'r') as f:
                source_code = f.read()
    
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

    # Tool: ChangeCode
    # Description: Updates the target system file with the provided content
    def change_code(file_content: str) -> str:
        """
            Updates the target system file with the provided content.
                file_content: The complete content to write to the target system file.
        """
        try:
            with open(target_system_file, 'w') as f:
                f.write(file_content)
            print("Successfully updated the system file.")
            return "Successfully updated the system file."
        except Exception as e:
            error_msg = f"Error updating system file: {repr(e)}"
            print(error_msg)
            return error_msg

    tools["ChangeCode"] = tool(runnable=change_code, name_or_callable="ChangeCode")

    # Tool: EndDesign
    # Description: Finalizes the system design process
    def end_design() -> str:
        """
            Finalizes the system design process.
        """
        return "Ending the design process..."

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

        # Read the current content of the target system file
        with open(target_system_file, 'r') as f:
            code_content = f.read()

        code_message = f"(Iteration {iteration}) Current Code:\n" + code_content

        full_messages = [SystemMessage(content=system_prompts.meta_agent), initial_message] + trimmed_messages + [HumanMessage(content=code_message)]
        print([getattr(last_msg, 'type', 'Unknown') for last_msg in full_messages])
        response = llm.invoke(full_messages)

        if not hasattr(response, 'content') or not response.content:
            response.content = "I will call the necessary tools."

        tool_messages, tool_results = [], {}
        
        decorator_tool_messages, decorator_tool_results, _ = execute_decorator_tool_calls(response.content)

        if decorator_tool_messages:
            tool_messages.extend(decorator_tool_messages)
            tool_results.update(decorator_tool_results)

        # Update messages with the response and tool messages
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

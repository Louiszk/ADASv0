# MetaSystem System Configuration
# Total nodes: 1
# Total tools: 4

from langgraph.graph import StateGraph
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from typing import Dict, List, Any, Callable, Optional, Union, TypeVar, Generic, Tuple, Set, TypedDict
from agentic_system.large_language_model import LargeLanguageModel, execute_tool_calls
from tqdm import tqdm
import traceback
import re
import io
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
            with open(target_system_file, 'r') as f:
                source_code = f.read()
            if "set_entry_point" not in source_code or "set_finish_point" not in source_code:
                raise Exception("You must set an entry point and finish point before testing")
    
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

    # Tool: ChangeCode
    # Description: Modifies the target system file using a diff
    def change_code(diff: str) -> str:
        """
            Modifies the target system file using a unified diff.
                diff: A unified diff string representing the changes to make to the target system file.
        """
        try:
            from agentic_system.udiff import find_diffs, do_replace, hunk_to_before_after, no_match_error, SearchTextNotUnique

            with open(target_system_file, 'r') as f:
                content = f.read()

            edits = find_diffs(diff)

            if not edits:
                print(no_match_error)
                return no_match_error

            success = False
            failed_hunks = []

            for _, hunk in edits:
                try:
                    # Apply the diff
                    new_content = do_replace(target_system_file, content, hunk)
                    if new_content is not None:
                        content = new_content
                        success = True
                    else:
                        # failed hunks for debugging
                        before_text, _ = hunk_to_before_after(hunk)
                        failed_hunks.append({
                            "before_text": before_text[:150] + ("..." if len(before_text) > 150 else ""),
                            "hunk_lines": len(hunk),
                            "error": "no_match"
                        })
                except SearchTextNotUnique:
                    before_text, _ = hunk_to_before_after(hunk)
                    failed_hunks.append({
                        "before_text": before_text[:150] + ("..." if len(before_text) > 150 else ""),
                        "hunk_lines": len(hunk),
                        "error": "not_unique"
                    })

            if not success:
                error_msg = f"Error: Failed to apply diffs to the system.\n"

                for i, failed in enumerate(failed_hunks):
                    if failed.get("error") == "not_unique":
                        error_msg += f"Hunk #{i+1} matched multiple locations:\n```\n{failed['before_text']}\n```\n"
                    else:
                        error_msg += f"Hunk #{i+1} failed to match:\n```\n{failed['before_text']}\n```\n"

                print(error_msg)
                return (error_msg)

            with open(target_system_file, 'w') as f:
                f.write(content)

            print("Successfully applied diff to the system.")
        except Exception as e:
            print(f"Error applying diff: {repr(e)}")

    tools["ChangeCode"] = tool(runnable=change_code, name_or_callable="ChangeCode")

    # Tool: EndDesign
    # Description: Finalizes the system design process
    def end_design() -> str:
        """
            Finalizes the system design process.
        """
        try:
            with open(target_system_file, 'r') as f:
                content = f.read()

            if "set_entry_point" not in content or "set_finish_point" not in content:
                print("Error finalizing system: You must set an entry point and finish point before finalizing")
                return None

            # We could test the system here again

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
    
        # Read the current content of the target system file
        with open(target_system_file, 'r') as f:
            code_content = f.read()

        code_message = f"(Iteration {iteration}) Current Code:\n" + code_content
    
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
            "pip_install": pip_install,
            "test_system": test_system,
            "change_code": change_code,
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
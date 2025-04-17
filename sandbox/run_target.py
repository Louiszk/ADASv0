import os
import sys
import json
import time
import importlib
import argparse

sys.path.append('/sandbox/workspace')

def main():
    parser = argparse.ArgumentParser(description="Run an agentic system")
    parser.add_argument("--system_name", help="Name of the target system to run")
    parser.add_argument("--state", default="{}", help="JSON string defining the initial state")
    args = parser.parse_args()
    
    system_name = args.system_name
    print(f"Running target system: {system_name}")
    
    try:
        # Import the target system module dynamically
        module_path = f"automated_systems.{system_name}"
        target_module = importlib.import_module(module_path)
        
        workflow, tools = target_module.build_system()
        
        try:
            filename = f"automated_systems/{system_name}.png"
            workflow.get_graph().draw_mermaid_png(output_file_path=filename)
            print(f"System graph visualization saved to {filename}")
        except Exception as e:
            print(f"Failed to visualize graph: {str(e)}")
        
        try:
            state_dict = json.loads(args.state)
        except json.JSONDecodeError:
            print("Invalid JSON for state, using empty state")
            state_dict = {}
        
        print(f"Initial state: {state_dict}")
        
        print("Streaming system execution...")
        for output in workflow.stream(state_dict, config={"recursion_limit": 80}):
            print(list(output.keys()))
           
            for out in output.values():
                if "messages" in out:
                    messages = out["messages"]
                    if messages:
                        last_msg = messages[-1]
                        msg_type = getattr(last_msg, 'type', 'Unknown')
                        content = getattr(last_msg, 'content', '')
                        tool_calls = getattr(last_msg, 'tool_calls', '')
                        print(f"\n[{msg_type}]: {content}\n {tool_calls}")
                del out["messages"]
                print(out)
            time.sleep(2)
    
    except ModuleNotFoundError:
        print(f"Error: System '{system_name}' not found.")
    except Exception:
        import traceback
        error = traceback.format_exc()
        print(f"Error running system: {error}")

if __name__ == "__main__":
    main()
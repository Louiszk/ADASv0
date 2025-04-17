import os
import sys
import json
import dill as pickle
import time
import datetime
from langchain_core.messages import HumanMessage

sys.path.append('/sandbox/workspace')
from agentic_system.virtual_agentic_system import VirtualAgenticSystem
from systems import MetaSystem

def main():
    start_time = time.time()
    metrics = {
        "system_name": "",
        "start_time": datetime.datetime.now().isoformat(),
        "end_time": None,
        "duration_seconds": 0,
        "iterations": 0,
        "token_usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        },
        "status": "started",
        "error": None,
        "stream_content": ""
    }
    
    problem_statement = "Create me a simple system that can produce eggs."
    if len(sys.argv) >= 2:
        problem_statement = sys.argv[1]
   
    system_name = "Eggs"
    if len(sys.argv) >= 3:
        system_name = sys.argv[2]
    
    metrics["system_name"] = system_name
    metrics["problem_statement"] = problem_statement
    
    optimize_from_file = None
    if len(sys.argv) >= 4:
        optimize_from_file = sys.argv[3]
        metrics["optimize_from_file"] = optimize_from_file
   
    print(f"Running meta system for '{system_name}'...")
   
    try:
        if optimize_from_file:
            path = "/sandbox/workspace/automated_systems/" + optimize_from_file.replace("/", "_").replace("\\", "_").replace(":", "_")
            try:
                with open(path + '.pkl', 'rb') as f:
                    MetaSystem.target_system = pickle.load(f)

                MetaSystem.target_system.system_name = system_name
                print("Code initialized")
            except Exception as e:
                print(f"Error initializing: {e}")
                MetaSystem.target_system = VirtualAgenticSystem(system_name)
        else:
            MetaSystem.target_system = VirtualAgenticSystem(system_name)
       
        workflow, tools = MetaSystem.build_system()
        inputs = {"messages": [HumanMessage(content=problem_statement)]}
        processed_msg_ids = set()
        processed_msg_count = 0

        print("Streaming meta system execution...")
        for output in workflow.stream(inputs, config={"recursion_limit": 80}):
            metrics["iterations"] += 1
            
            for out in output.values():
                if "messages" in out:
                    messages = out["messages"]
                    if messages:
                        # Only process new messages
                        new_messages = messages[processed_msg_count:]
                        for msg in new_messages:
                            msg_type = getattr(msg, 'type', 'Unknown')
                            content = getattr(msg, 'content', '')
                            stream_content = f"\n[{msg_type}]: {content}\n"
                            metrics["stream_content"] += stream_content.replace('"', '\"')
                            print(stream_content)
                            
                            if "MALFORMED_FUNCTION_CALL" in str(msg):
                                print("MALFORMED_FUNCTION_CALL")
                            
                            # Extract token usage from AI messages (avoid double counting)
                            msg_id = getattr(msg, 'id', None)
                            if (msg_id and msg_id not in processed_msg_ids and
                                hasattr(msg, 'usage_metadata') and msg.usage_metadata):
                                token_usage = msg.usage_metadata
                                metrics["token_usage"]["input_tokens"] += token_usage.get('input_tokens', 0)
                                metrics["token_usage"]["output_tokens"] += token_usage.get('output_tokens', 0)
                                metrics["token_usage"]["total_tokens"] += token_usage.get('total_tokens', 0)
                                processed_msg_ids.add(msg_id)
                        
                        # Update count
                        processed_msg_count = len(messages)
                
                if "design_completed" in out and out["design_completed"]:
                    print("Design completed.")
                    metrics["status"] = "completed"
            
            time.sleep(2)

        metrics["status"] = "completed"
       
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(error_traceback)
        print(f"Error running meta system: {str(e)}")
        
        metrics["status"] = "error"
        metrics["error"] = {
            "message": repr(e),
            "traceback": error_traceback
        }
    
    finally:
        # Finalize metrics
        end_time = time.time()
        metrics["end_time"] = datetime.datetime.now().isoformat()
        metrics["duration_seconds"] = end_time - start_time
        
        metrics_dir = "/sandbox/workspace/automated_systems/metrics"
        os.makedirs(metrics_dir, exist_ok=True)
        
        metrics_file = f"{metrics_dir}/" + system_name.replace('/', '_').replace('\\', '_').replace(':', '_') + ".json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Metrics saved to {metrics_file}")

if __name__ == "__main__":
    main()

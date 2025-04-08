import os
import sys
import json
import time
import shutil
import datetime
from langchain_core.messages import HumanMessage

sys.path.append('/sandbox/workspace')
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
    
    MetaSystem.target_system_name = system_name
    MetaSystem.target_system_file = "/sandbox/workspace/automated_systems/" + system_name.replace("/", "_").replace("\\", "_").replace(":", "_") + ".py"
    
    try:
        # Initialize target system file
        source_path = "/sandbox/workspace/agentic_system/target_system_template.py"
        initial_code = ""
        
        if optimize_from_file:
            source_path = "/sandbox/workspace/automated_systems/" + optimize_from_file.replace("/", "_").replace("\\", "_").replace(":", "_") + ".py"

        if os.path.exists(source_path):
            with open(source_path, 'r') as f:
                initial_code = f.read()
                
            with open(MetaSystem.target_system_file, 'w') as f:
                f.write(initial_code)
                
            print(f"Initialized target system from {source_path}")
        else:
            print("Target system file path does not exist.")
                
        workflow, tools = MetaSystem.build_system()
        inputs = {"messages": [HumanMessage(content=problem_statement)]}
        processed_msg_ids = set()
       
        print("Streaming meta system execution...")
        for output in workflow.stream(inputs, config={"recursion_limit": 80}):
            metrics["iterations"] += 1
            
            for out in output.values():
                if "messages" in out:
                    messages = out["messages"]
                    if messages:
                        last_msgs = messages[-2:]
                        for last_msg in last_msgs:
                            msg_type = getattr(last_msg, 'type', 'Unknown')
                            content = getattr(last_msg, 'content', '')
                            stream_content = f"\n[{msg_type}]: {content}\n"
                            metrics["stream_content"] += stream_content.replace('"', '\"')
                            print(stream_content)
                            
                            # Extract token usage from AI messages (avoid double counting)
                            msg_id = getattr(last_msg, 'id', None)
                            if (msg_id and msg_id not in processed_msg_ids and 
                                hasattr(last_msg, 'usage_metadata') and last_msg.usage_metadata):
                                token_usage = last_msg.usage_metadata
                                metrics["token_usage"]["input_tokens"] += token_usage.get('input_tokens', 0)
                                metrics["token_usage"]["output_tokens"] += token_usage.get('output_tokens', 0)
                                metrics["token_usage"]["total_tokens"] += token_usage.get('total_tokens', 0)
                                processed_msg_ids.add(msg_id)
                
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
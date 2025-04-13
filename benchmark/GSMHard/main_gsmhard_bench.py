import os
import argparse
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from sandbox.sandbox import StreamingSandboxSession, setup_sandbox_environment, check_docker_running

def run_gsmhard_benchmark_in_sandbox(session, system_name):
    print(f"Running GSM-Hard benchmark for system: {system_name}")
    
    base_path = "benchmark/GSMHard"
    system_path = system_name.replace(".", "/") + ".py"
    os.makedirs(base_path, exist_ok=True)
    
    # Ensure the benchmark directory exists in the sandbox
    session.execute_command(f"mkdir -p /sandbox/workspace/{base_path}")
    session.execute_command(f"mkdir -p /sandbox/workspace/{base_path}/results")
    session.execute_command(f"mkdir -p /sandbox/workspace/automated_systems/GSMHard")
    
    # Copy benchmark files to the sandbox
    session.copy_to_runtime(f"{base_path}/run_gsmhard_bench.py", f"/sandbox/workspace/{base_path}/run_gsmhard_bench.py")
    session.copy_to_runtime(f"{base_path}/GSMHardBaseline.py", f"/sandbox/workspace/{base_path}/GSMHardBaseline.py")
    session.copy_to_runtime(f"{base_path}/GSMHardExecBaseline.py", f"/sandbox/workspace/{base_path}/GSMHardExecBaseline.py")
    session.copy_to_runtime(system_path, f"/sandbox/workspace/{system_path}")
    
    if "not found" in str(session.execute_command("pip show datasets")):
        session.execute_command("pip install datasets")
    
    if "not found" in str(session.execute_command("pip show sympy")):
        session.execute_command("pip install sympy")
    
    # Run the benchmark
    command = f"python3 /sandbox/workspace/{base_path}/run_gsmhard_bench.py --system=\"{system_name}\""
    print(f"Executing command: {command}")
    
    # Run the benchmark and stream the output
    for chunk in session.execute_command_streaming(command):
        print(chunk, end="", flush=True)
    
    print("\nBenchmark execution completed!")
    
    # Copy the results back to the host
    os.makedirs(f"{base_path}/results", exist_ok=True)
    results_file = f"benchmark_results_{system_name}.json"
    if results_file in str(session.execute_command(f"ls -la /sandbox/workspace/{base_path}/results")):
        session.copy_from_runtime(
            f"/sandbox/workspace/{base_path}/results/{results_file}", 
            f"{base_path}/results/{results_file}"
        )
        print(f"Copied benchmark results back to host as {results_file}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Run GSM-Hard benchmark in a sandboxed environment")
    parser.add_argument("--system", required=True, help="Name of the system to benchmark (e.g., 'GSMHardBaseline')")
    parser.add_argument("--no-keep-template", dest="keep_template", action="store_false", help="Don't keep the Docker image after the session is closed")
    parser.add_argument("--reinstall", action="store_true", help="Reinstall dependencies")
    
    args = parser.parse_args()
    
    session = StreamingSandboxSession(
        image="python:3.11-slim",
        keep_template=args.keep_template,
        verbose=True
    )
    
    try:
        session.open()
        print("Sandbox session opened")
        
        if setup_sandbox_environment(session, args.reinstall):
            run_gsmhard_benchmark_in_sandbox(
                session, 
                args.system
            )
            print("Benchmark finished successfully!")
        else:
            print("Failed to set up sandbox environment")

    except Exception as e:
        print(f"Error during benchmark execution: {str(e)}")
    finally:
        print("Closing session...")
        session.close()

if __name__ == "__main__":
    main()
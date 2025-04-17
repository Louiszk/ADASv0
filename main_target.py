import json
import argparse
from sandbox.sandbox import StreamingSandboxSession, setup_sandbox_environment, check_docker_running, check_podman_running

def run_target_system(session, system_name, state=None):
    cmd_parts = [f"python3 /sandbox/workspace/run_target.py --system_name=\"{system_name}\""]
    
    if state:
        state = json.dumps(state)
        quoted_state = state.replace('"', '\\"')
        cmd_parts.append(f'--state="{quoted_state}"')
    
    command = " ".join(cmd_parts)
    print(f"Executing: {command}")
    
    for chunk in session.execute_command_streaming(command):
        print(chunk, end="", flush=True)
    
    print("\nTarget system execution completed")

def main():
    parser = argparse.ArgumentParser(description="Run target agentic systems in a sandboxed environment")
    parser.add_argument("--system_name", required=True, help="Name of the target system to run")
    parser.add_argument("--state", default="{\"messages\": [\"Hello\"]}", 
                        help="JSON string defining the initial state")
    parser.add_argument("--container", choices=["auto", "docker", "podman"], default="auto", 
                        help="Container runtime to use (auto will try Docker first, then Podman)")
    
    args = parser.parse_args()
    
    # Determine container type
    container_type = None
    if args.container == "docker":
        if not check_docker_running():
            print("Docker is not running or not available. Please start Docker and try again.")
            return
        container_type = "docker"
    elif args.container == "podman":
        if not check_podman_running():
            print("Podman is not running or not available. Please install/start Podman and try again.")
            return
        container_type = "podman"
    else:  # auto
        if not check_docker_running() and not check_podman_running():
            print("Neither Docker nor Podman are available. Please install and start one of them.")
            return
        # container_type remains None to let StreamingSandboxSession auto-detect
    
    try:
        state_dict = json.loads(args.state)
    except json.JSONDecodeError:
        print("Invalid JSON for state, using default state")
        state_dict = {"messages": ["Hello"]}
    
    session = StreamingSandboxSession(
        image="python:3.11-slim",
        keep_template=True,
        verbose=True,
        container_type=container_type
    )
    
    try:
        session.open()
        if setup_sandbox_environment(session):
            run_target_system(session, args.system_name, state_dict)
        else:
            print("Failed to set up sandbox environment")
    finally:
        print("Closing session...")
        session.close()

if __name__ == "__main__":
    main()
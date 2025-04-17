import os
import argparse
from sandbox.sandbox import StreamingSandboxSession, setup_sandbox_environment, check_docker_running, check_podman_running

def run_meta_system_in_sandbox(session, problem_statement, target_name, optimize_system=None):
    quoted_problem = problem_statement.replace('"', '\\"')
    command = f"python3 /sandbox/workspace/run_meta.py \"{quoted_problem}\" \"{target_name}\" " + (
        f"\"{optimize_system}\"" if optimize_system else "")
    
    for chunk in session.execute_command_streaming(command):
        print(chunk, end="", flush=True)
    
    print("\nMeta system execution completed!")
    
    # Copy any generated systems and metrics back to the host
    if "automated_systems" in str(session.execute_command("ls -la /sandbox/workspace")):
        print("Copying generated systems and metrics back to host...")
        os.makedirs("automated_systems", exist_ok=True)
        target_file_name = target_name.replace("/", "_").replace("\\", "_").replace(":", "_") + ".py"
        
        if target_file_name in str(session.execute_command("ls -la /sandbox/workspace/automated_systems")):
            session.copy_from_runtime(
                f"/sandbox/workspace/automated_systems/{target_file_name}", 
                f"automated_systems/{target_file_name}"
            )
            print(f"Copied {target_file_name} back to host")
        
        if "metrics" in str(session.execute_command("ls -la /sandbox/workspace/automated_systems")):
            metrics_file = target_name.replace("/", "_").replace("\\", "_").replace(":", "_") + ".json"
            
            if metrics_file in str(session.execute_command("ls -la /sandbox/workspace/automated_systems/metrics")):
                os.makedirs("automated_systems/metrics", exist_ok=True)
                session.copy_from_runtime(
                    f"/sandbox/workspace/automated_systems/metrics/{metrics_file}", 
                    f"automated_systems/metrics/{metrics_file}"
                )
                print(f"Copied metrics file {metrics_file} back to host")
    
    return True

def main():
    prompt = """
    Design a system to solve 'Project Euler' tasks.
    Project Euler challenges participants to solve complex mathematical and computational problems
    using programming skills and mathematical insights.
    
    The system should consist of just one agent and one tool.
    The tool should allow to execute python code, so that the agent can solve any problem.
    The state must contain the attribute "solution" : "str", where only the final solution is saved.
    """

    target_name = "SimpleEulerSolver"

    parser = argparse.ArgumentParser(description="Run agentic systems in a sandboxed environment")
    parser.add_argument("--no-keep-template", dest="keep_template", action="store_false", help="Don't keep the Docker image after the session is closed")
    parser.add_argument("--reinstall", action="store_true", help="Reinstall dependencies.")
    parser.add_argument("--problem", default=prompt, help="Problem statement to solve")
    parser.add_argument("--name", default=target_name, help="Target system name")
    parser.add_argument("--optimize-system", default=None, help="Specify target system name to optimize or change")
    parser.add_argument("--container", choices=["auto", "docker", "podman"], default="auto", 
                        help="Container runtime to use (auto will try Docker first, then Podman)")
    
    args = parser.parse_args()
    print(args)
    
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
    
    session = StreamingSandboxSession(
        # dockerfile="Dockerfile",
        image="python:3.11-slim",
        keep_template=args.keep_template,
        verbose=True,
        container_type=container_type
    )
    
    try:
        session.open()
        if setup_sandbox_environment(session, args.reinstall):
            run_meta_system_in_sandbox(session, args.problem, args.name, args.optimize_system)
            print("Finished successfully!")
        else:
            print("Failed to set up sandbox environment")
    except Exception as e:
        print(repr(e))
    finally:
        print("Session closed.")
        session.close()

if __name__ == "__main__":
    main()
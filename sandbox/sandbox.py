from llm_sandbox.docker import SandboxDockerSession
from llm_sandbox.podman import SandboxPodmanSession
import os
import subprocess
import json

class StreamingSandboxSession:
    def __init__(self, image=None, dockerfile=None, keep_template=False, 
                 stream=True, verbose=True, runtime_configs=None, 
                 container_type=None, **kwargs):
        self.verbose = verbose
        self.session = None
        
        # Determine which container technology to use
        if container_type:
            # If explicitly specified, use that
            self._initialize_session(container_type, image, dockerfile, keep_template, 
                                     verbose, runtime_configs, **kwargs)
        else:
            # Try Docker first, then Podman
            if check_docker_running():
                self._initialize_session('docker', image, dockerfile, keep_template, 
                                         verbose, runtime_configs, **kwargs)
            elif check_podman_running():
                self._initialize_session('podman', image, dockerfile, keep_template, 
                                         verbose, runtime_configs, **kwargs)
            else:
                raise RuntimeError("Neither Docker nor Podman are available. Please install and start one of them.")
    
    def _initialize_session(self, container_type, image, dockerfile, keep_template, 
                           verbose, runtime_configs, **kwargs):
        if self.verbose:
            print(f"Using {container_type} as container runtime")
            
        if container_type == 'docker':
            self.session = SandboxDockerSession(
                image=image,
                dockerfile=dockerfile,
                keep_template=keep_template,
                verbose=verbose,
                runtime_configs=runtime_configs,
                **kwargs
            )
        elif container_type == 'podman':
            # For Podman, use direct image ID lookup
            podman_image = image
            if image and not dockerfile and '/' not in image:
                # Use fully qualified image name for podman
                podman_image = f"docker.io/library/{image}"
                if verbose:
                    print(f"Using fully qualified image name for Podman: {podman_image}")
                
                # Get image ID directly using CLI for more reliable operation
                image_id = get_podman_image_id(podman_image, verbose)
                if image_id:
                    if verbose:
                        print(f"Using image ID directly: {image_id}")
                    podman_image = image_id
                elif verbose:
                    print("Image ID not found, will try pulling during session initialization")
            
            self.session = SandboxPodmanSession(
                image=podman_image,
                dockerfile=dockerfile,
                keep_template=keep_template,
                verbose=verbose,
                runtime_configs=runtime_configs,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown container type: {container_type}")

    def open(self):
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        try:
            return self.session.open()
        except Exception as e:
            if hasattr(self.session, 'image') and isinstance(self.session, SandboxPodmanSession):
                # Handle podman-specific errors
                if "image not known" in str(e) and isinstance(self.session.image, str):
                    if self.verbose:
                        print(f"Image not found. Attempting to pull: {self.session.image}")
                    try:
                        # Try pulling with CLI
                        result = subprocess.run(
                            ["podman", "pull", self.session.image],
                            capture_output=True, text=True, check=True
                        )
                        if self.verbose:
                            print(result.stdout)
                        
                        # Try again after pull
                        return self.session.open()
                    except Exception as pull_error:
                        print(f"Failed to pull image: {pull_error}")
                        raise e
            raise e
    
    def close(self):
        return self.session.close() if self.session else None
    
    def execute_command(self, command, workdir=None):
        return self.session.execute_command(command, workdir)
    
    def copy_to_runtime(self, src, dest):
        return self.session.copy_to_runtime(src, dest)
    
    def copy_from_runtime(self, src, dest):
        return self.session.copy_from_runtime(src, dest)
    
    def execute_command_streaming(self, command, workdir=None):
        if not self.session:
            raise RuntimeError("Session is not open")
        
        kwargs = {"stream": True, "tty": True}
        if workdir:
            kwargs["workdir"] = workdir
            
        _, output_stream = self.session.container.exec_run(command, **kwargs)
        
        # podman buffers the stream
        for chunk in output_stream:
            yield chunk.decode("utf-8")

def get_podman_image_id(image_name, verbose=False):
    """Get Podman image ID using CLI."""
    try:
        result = subprocess.run(
            ["podman", "images", "--format", "{{.ID}}", image_name],
            capture_output=True, text=True, check=False
        )
        image_id = result.stdout.strip()
        
        if not image_id and verbose:
            print(f"Image {image_name} not found with exact name, checking repositories...")
            
            # Check all images
            result = subprocess.run(
                ["podman", "images", "--format", "json"],
                capture_output=True, text=True, check=False
            )
            
            try:
                images = json.loads(result.stdout)
                for img in images:
                    # Check for name match in different formats
                    if image_name in str(img.get('names', [])):
                        image_id = img.get('id', '')
                        if verbose:
                            print(f"Found image with ID: {image_id}")
                        break
            except json.JSONDecodeError:
                if verbose:
                    print("Failed to parse image list JSON")
        
        return image_id
    except Exception as e:
        if verbose:
            print(f"Error getting image ID: {e}")
        return None

def check_docker_running():
    """Check if Docker is running and available."""
    try:
        import docker
        client = docker.from_env()
        client.ping()
        return True
    except (ImportError, docker.errors.DockerException) as e:
        if isinstance(e, ImportError):
            print("Docker Python package not installed. Unable to use Docker.")
        else:
            print("Could not connect to Docker daemon. Is Docker Desktop running?")
            print("\nDetailed error:", str(e))
        return False

def check_podman_running():
    """Check if Podman is running and available."""
    try:
        # Check podman availability using CLI first (more reliable)
        result = subprocess.run(
            ["podman", "info", "--format", "{{.Host.RemoteSocket.Path}}"],
            capture_output=True, text=True, check=False
        )
        
        if result.returncode != 0:
            # Try basic command
            result = subprocess.run(
                ["podman", "info"],
                capture_output=True, text=True, check=False
            )
            
        if result.returncode != 0:
            print("Podman command failed. Is podman installed?")
            return False
        
        # Now try with Python API
        from podman import PodmanClient
        client = PodmanClient()
        client.info()
        return True
    except (ImportError, Exception) as e:
        if isinstance(e, ImportError):
            print("Podman Python package not installed. Unable to use Podman.")
        else:
            print("Could not connect to Podman. Is Podman installed and running?")
            print("\nDetailed error:", str(e))
        return False
    
def setup_sandbox_environment(session, reinstall=False):
    """Set up the sandbox environment with required files and dependencies."""
    print("Setting up sandbox environment...")
    
    session.execute_command("mkdir -p /sandbox/workspace/systems")
    session.execute_command("mkdir -p /sandbox/workspace/agentic_system")
    session.execute_command("mkdir -p /sandbox/workspace/automated_systems")
    
    required_files = [
        ("agentic_system/virtual_agentic_system.py", "/sandbox/workspace/agentic_system/virtual_agentic_system.py"),
        ("agentic_system/large_language_model.py", "/sandbox/workspace/agentic_system/large_language_model.py"),
        ("agentic_system/materialize.py", "/sandbox/workspace/agentic_system/materialize.py"),
        ("agentic_system/udiff.py", "/sandbox/workspace/agentic_system/udiff.py"),
        ("agentic_system/target_system_template.py", "/sandbox/workspace/agentic_system/target_system_template.py"),
        ("systems/system_prompts.py", "/sandbox/workspace/systems/system_prompts.py"),
        ("systems/MetaSystem.py", "/sandbox/workspace/systems/MetaSystem.py"),
        ("sandbox/run_meta.py", "/sandbox/workspace/run_meta.py"),
        ("sandbox/run_target.py", "/sandbox/workspace/run_target.py"),
        (".env", "/sandbox/workspace/.env")
    ]
    
    for src_path, dest_path in required_files:
        if os.path.exists(src_path):
            session.copy_to_runtime(src_path, dest_path)
        else:
            print(f"Warning: Required file {src_path} not found")
    
    if reinstall:
        print("Installing dependencies in sandbox...")
        dependencies = [
            "langgraph==0.3.5", 
            "langchain_openai==0.3.8",
            "langchain_google_genai==2.0.11",
            "python-dotenv==1.0.1",
            "dill==0.3.9"
        ]
        session.execute_command(f"pip install {' '.join(dependencies)}")
    
    print("Sandbox environment set up successfully!")
    return True
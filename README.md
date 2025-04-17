# Automated Design of Agentic Systems

Third version with multiple tools to design the system

## Quick Setup

### Environment Setup

1. Clone the repository
2. Copy the example environment file:
   ```bash
   cp .env_copy .env
   ```
3. Edit `.env` with your API keys:
   ```
   OPENAI_API_KEY=sk-...
   GOOGLE_API_KEY=...
   # Optional:
   # PERPLEXITY_API_KEY=...
   # HELMHOLTZ_API_KEY=...
   ```

### Virtual Environment (recommended)

```bash
# Create virtual environment
python -m venv adasvenv

# Activate on Linux/Mac
source adasvenv/bin/activate
# OR on Windows
# adasvenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Docker Setup (for sandbox execution)

The system uses Docker to create a sandbox environment for execution. Make sure you have Docker installed and running:

```bash
docker --version
```

## Running the System

### Creating and Running the Meta System

The meta system is an agentic system that can design other agentic systems.

First Start
```bash
python main_meta.py --reinstall
```

After the first start it is advisable to set no-keep-template.

Options:
- `--name`: Target system name
- `--problem`: Problem statement to solve
- `--reinstall`: Reinstall dependencies (should be set for the first start)
- `--optimize-system`: Name of existing system to optimize
- `--no-keep-template`: Do not keep the docker image template

- Some examples are in system_ideas.txt

### Running a Target System

```bash
python main_target.py --system_name "SimpleEulerSolver"
```

You can set the initial state in the main_target.py file.
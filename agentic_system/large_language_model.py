import os
import re
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import ToolMessage
from dotenv import load_dotenv

def extract_code_blocks(text):
    """Extract all Python code blocks from markdown text."""
    pattern = r'```\s*python\s*([\s\S]*?)```'
    matches = re.findall(pattern, text, re.IGNORECASE)
    return [match.strip() for match in matches]

def extract_function_info(code_block):
    """Extract function name and type (node, tool, router) based on naming conventions."""
    # Find function definition
    match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code_block)
    if not match:
        return None, None
    
    function_name = match.group(1)
    
    # Determine function type based on naming convention
    if function_name.endswith('_node'):
        return function_name, "node"
    elif 'tool' in function_name:
        return function_name, "tool"
    elif function_name.endswith('_router'):
        return function_name, "router"
    else:
        return function_name, "unknown"

def extract_router_source(code_block):
    """Extract the source node for a router from function comments or docstring."""
    # Look for a comment or docstring indicating the source node
    source_match = re.search(r'#\s*source[:\s]+([a-zA-Z_][a-zA-Z0-9_]*)', code_block, re.IGNORECASE)
    if source_match:
        return source_match.group(1)
    
    docstring_match = re.search(r'"""[\s\S]*?source[:\s]+([a-zA-Z_][a-zA-Z0-9_]*)', code_block, re.IGNORECASE)
    if docstring_match:
        return docstring_match.group(1)
    
    # Default: try to guess from function name
    router_match = re.search(r'def\s+([a-zA-Z_]+)_router', code_block)
    if router_match:
        source_guess = router_match.group(1) + "_node"
        return source_guess
    
    return None

def extract_path_map(code_block, all_nodes):
    # Extract potential node names from string literals
    string_pattern = r"['\"]([^'\"]*)['\"]"
    potential_nodes = set(re.findall(string_pattern, code_block))
    
    path_map = None
    auto_path_map = {}
    for node_name in potential_nodes:
        if node_name in all_nodes:
            auto_path_map[node_name] = node_name
    
    if auto_path_map:
        path_map = auto_path_map
    return path_map

def execute_tool_calls(response):
    """Execute any tool calls in the llm response."""
    if not hasattr(response, "tool_calls") or not response.tool_calls:
        return [], {}
        
    tool_messages = []
    tool_results = {}
    for tool_call in response.tool_calls:
        print("Tool Call:", tool_call)
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        tool_id = tool_call['id']
        
        if tool_name in LargeLanguageModel.available_tools:
            try:
                result = LargeLanguageModel.available_tools[tool_name].invoke(tool_args)
                tool_messages.append(ToolMessage(
                    content=str(result) if result else f"Tool {tool_name} executed successfully.",
                    tool_call_id=tool_id,
                    name=tool_name
                ))
                tool_results[tool_name] = result
            except Exception as e:
                error_message = f"Error executing tool {tool_name}: {repr(e)}"
                tool_messages.append(ToolMessage(
                    content=error_message,
                    tool_call_id=tool_id,
                    name=tool_name
                ))
                tool_results[tool_name] = error_message
                
    return tool_messages, tool_results

load_dotenv()
def get_model(wrapper, model_name, temperature):
    api_keys = {
        "google": "GOOGLE_API_KEY",
        "openai": "OPENAI_API_KEY",
        "blablador": "HELMHOLTZ_API_KEY",
        "scads": "SCADS_API_KEY",
        "perplexity": "PERPLEXITY_API_KEY"
    }
    
    if wrapper not in api_keys:
        raise ValueError(f"Invalid wrapper: '{wrapper}'. Supported: {', '.join(api_keys.keys())}")
    
    key_name = api_keys[wrapper]
    api_key = os.getenv(key_name)
    if not api_key:
        raise ValueError(f"Missing environment variable: {key_name} required for {wrapper}")
    
    try:
        model_wrapper = {
            "google": ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=temperature,
                    google_api_key=api_key
                ),
            "openai": ChatOpenAI(
                    model=model_name,
                    temperature=temperature,
                    api_key=api_key
                ),
            "blablador": ChatOpenAI(
                    model=model_name,
                    temperature=temperature,
                    api_key=api_key,
                    cache=False,
                    max_retries=2,
                    base_url="https://api.helmholtz-blablador.fz-juelich.de/v1/"
                ),
            "scads": ChatOpenAI(
                    model=model_name,
                    temperature=temperature,
                    api_key=api_key,
                    cache=False,
                    max_retries=2,
                    base_url="https://llm.scads.ai/v1"
                ),
            "perplexity": ChatOpenAI(
                    model="sonar",
                    temperature=temperature,
                    api_key=api_key,
                    base_url="https://api.perplexity.ai"
            )
        }
        return model_wrapper[wrapper]
    except Exception as e:
        raise RuntimeError(f"Failed to initialize {wrapper} model: {str(e)}") from e

class LargeLanguageModel:
    available_tools = {}

    def __init__(self, temperature=0.4, wrapper = "openai", model_name="gpt-4o-mini"):
        self.model = get_model(wrapper, model_name, temperature)
        self.wrapper = wrapper

    def bind_tools(self, tool_names, parallel_tool_calls=True):
        if tool_names:
            tool_objects = [LargeLanguageModel.available_tools[tool_name] for tool_name in tool_names if tool_name in LargeLanguageModel.available_tools]
            if tool_objects:
                if self.wrapper=="google":
                    self.model = self.model.bind_tools(tool_objects)
                else:
                    self.model = self.model.bind_tools(tool_objects, parallel_tool_calls=parallel_tool_calls)
        return self.model

    def invoke(self, input):
        return self.model.invoke(input)
    
    @classmethod
    def register_available_tools(cls, tools_dict):
        cls.available_tools.update(tools_dict)

import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
import re
import uuid

def parse_decorator_tool_calls(text):
    """Parse decorator-style tool calls from text."""
    tool_calls = []
    
    def camelfy(snake_case):
        parts = snake_case.split("_")
        return "".join([part.capitalize() for part in parts])
    
    # Code-related tools that need special handling
    code_related_tools = {
        'create_node': 'function_code',
        'create_tool': 'function_code',
        'edit_component': 'new_function_code',
        'add_conditional_edge': 'condition_code'
    }
    
    # Extract code blocks
    code_blocks = re.findall(r'^```(.*?)\n```', text, re.DOTALL | re.MULTILINE)
    if not code_blocks:
        print("No code blocks found!")
    
    for block in code_blocks:
        lines = block.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('@@'):
                call_match = re.match(r'@@([a-zA-Z_][a-zA-Z0-9_]*)', line)
                if call_match:
                    decorator_name = call_match.group(1)
                    args_str = line[line.index('(')+1:line.rindex(')')]
                    tool_id = str(uuid.uuid4())[:32]
                    
                    # Map decorator name to tool name if possible
                    tool_name = camelfy(decorator_name)
                    
                    # Special handling for code-related tools
                    if decorator_name in code_related_tools:
                        start_idx = i + 1
                        end_idx = len(lines)
                        
                        for j in range(start_idx, len(lines)):
                            if lines[j].strip().startswith('@@'):
                                end_idx = j
                                break
                        
                        content = '\n'.join(lines[start_idx:end_idx])
                        
                        # Parse the regular arguments
                        args = {}
                        if args_str:
                            try:
                                exec_str = f"def parsing_function(**kwargs): return kwargs\nargs = parsing_function({args_str})"
                                local_vars = {"HumanMessage": HumanMessage, "SystemMessage": SystemMessage}
                                exec(exec_str, {}, local_vars)
                                args = local_vars.get('args', {})
                            except Exception as e:
                                print(f"Error parsing arguments: {e}")
                        
                        # Add the code content to the appropriate parameter
                        param_name = code_related_tools[decorator_name]
                        args[param_name] = content
                        
                        tool_calls.append({
                            'name': tool_name,
                            'args': args,
                            'id': f'call_{tool_id}'
                        })
                        
                        i = end_idx - 1
                    else:
                        # Parse regular arguments
                        args = {}
                        
                        if args_str:
                            try:
                                exec_str = f"def parsing_function(**kwargs): return kwargs\nargs = parsing_function({args_str})"
                                local_vars = {}
                                exec(exec_str, {}, local_vars)
                                args = local_vars.get('args', {})
                            except Exception as e:
                                print(f"Error parsing arguments: {e}")
                        
                        tool_calls.append({
                            'name': tool_name,
                            'args': args,
                            'id': f'call_{tool_id}'
                        })
            i += 1
    
    return tool_calls

def execute_decorator_tool_calls(text):
    """Execute decorator-style tool calls found in the text."""
    tool_calls = parse_decorator_tool_calls(text)
    if not tool_calls:
        return [], {}, []
        
    tool_messages = []
    tool_results = {}
    
    for tool_call in tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        tool_id = tool_call['id']
        
        if tool_name in LargeLanguageModel.available_tools:
            try:
                result = LargeLanguageModel.available_tools[tool_name].invoke(tool_args)
                
                tool_messages.append(HumanMessage(
                    content=str(result) if result else f"Tool {tool_name} executed successfully.",
                    tool_call_id=tool_id,
                    name=tool_name
                ))
                tool_results[tool_name] = result
            except Exception as e:
                error_message = f"Error executing tool {tool_name}: {repr(e)}"
                tool_messages.append(HumanMessage(
                    content=error_message,
                    tool_call_id=tool_id,
                    name=tool_name
                ))
                tool_results[tool_name] = error_message
        else:
            error_message = f"Tool {tool_name} not found"
            tool_messages.append(HumanMessage(
                content=error_message,
                tool_call_id=tool_id,
                name="error"
            ))
                
    return tool_messages, tool_results, tool_calls

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

    def __init__(self, temperature=0.2, wrapper = "openai", model_name="gpt-4.1-nano"):
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

import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import ToolMessage
from dotenv import load_dotenv

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
                    content=str(result),
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

    def bind_tools(self, tool_names, parallel_tool_calls=True):
        if tool_names:
            tool_objects = [LargeLanguageModel.available_tools[tool_name] for tool_name in tool_names if tool_name in LargeLanguageModel.available_tools]
            if tool_objects:
                self.model = self.model.bind_tools(tool_objects, parallel_tool_calls=parallel_tool_calls)
        return self.model

    def invoke(self, input):
        return self.model.invoke(input)
    
    @classmethod
    def register_available_tools(cls, tools_dict):
        cls.available_tools.update(tools_dict)

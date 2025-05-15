from typing import List, Callable, Dict, Any
import openai
from langchain_core.utils.function_calling import convert_to_openai_function
from .tools import get_weather, get_news
import json
from secret_keys import OPENROUTER_API_KEY

class Agent:
    def __init__(self, name: str, model_name: str, system_prompt: str, tools: List[str]):
        self.name = name
        self.model_name = model_name
        self.model = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY
        )

        self.setup_tools(tools)
        self.system_prompt = system_prompt
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]

    def manage_ctx_window(self):
        if len(self.messages) > 10:
            self.messages.pop(1)

    def setup_tools(self, tools: List[str]):
        self.tool_schemas = [convert_to_openai_function(i) for i in tools]
        self.tool_schemas = [{"type": "function", "function": i} for i in self.tool_schemas]
        self.tools = {i.__name__: i for i in tools}

    def get_llm_response(self):
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
            tools=self.tool_schemas
        )
        if response.choices[0].finish_reason == "stop":
            return response.choices[0].message.content, []
        elif response.choices[0].finish_reason == "tool_calls":
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = tool_call.function.arguments
                    print(f"Tool call: {tool_name} with args: {tool_args}")
            return response.choices[0].message.content, tool_calls
        return response.choices[0].message.content, []

    def handle_tool_calls(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            if tool_name == "get_weather":
                result = get_weather(**tool_args)
            elif tool_name == "get_news":
                result = get_news(**tool_args)
            else:
                result = f"Unknown tool: {tool_name}"
            results.append(result)
        return results
            

    def run(self,):
        while True:
            user_input = input("You: ")
            if user_input == "exit":
                break
            self.messages.append({"role": "user", "content": user_input})
            
            response, tool_calls = self.get_llm_response()
            self.messages.append({"role": "assistant", "content": response})
            print(f"Agent {self.name}: {response}")
            
            if tool_calls:
                print(f"Tool calls: {tool_calls}")
                results = self.handle_tool_calls(tool_calls)
                for result in results:
                    self.messages.append({"role": "tool", "content": result})
                    print(f"Tool result: {result}")

if __name__ == "__main__":
    agent = Agent(
        name="test",
        model_name="Qwen/Qwen3-4B",
        tools=[get_weather, get_news],
        system_prompt="You are a helpful assistant who has access to tools."
    )
    agent.run()
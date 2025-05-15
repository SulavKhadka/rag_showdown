# RAG Showdown

A project for testing and comparing different RAG (Retrieval-Augmented Generation) approaches.

## Agent Module

The `agent` module provides a simple agent implementation with tool-calling capabilities. It includes:

- `Agent` class: A core agent implementation that can process user inputs and call tools
- Tool definitions: Simple example tools like `get_weather` and `get_news`
- Prompt templates: Predefined prompts for tool usage

### Usage

```python
from agent import Agent, get_weather, get_news

# Create an agent instance
agent = Agent(
    name="my_agent",
    model_name="Qwen/Qwen3-4B",  # Or any other model
    tools=[get_weather, get_news],
    system_prompt="You are a helpful assistant who has access to tools."
)

# Run the agent in interactive mode
agent.run()
```

## Installation

```bash
# Install in development mode
pip install -e .
```
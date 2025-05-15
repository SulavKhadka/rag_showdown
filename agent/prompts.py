tools_prompt = """# Tools
You may call one or more functions to assist with the user query.

You are provided with function signatures below:
<tools>
{tools}
</tools>

For each function call, you MUST return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>

Remember to output the tool call within <tool_call></tool_call> tags.
"""
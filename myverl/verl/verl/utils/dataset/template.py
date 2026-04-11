def fc_sys_2(data, prompt_key, tokenizer):
    prompt_template = """In this environment you have access to a set of tools you can use to assist with the user query. \
You may perform multiple rounds of function calls. \
In each round, you can call one or more functions. \

Here are available functions in JSONSchema format: \n```json\n{func_schemas}\n```

In your response, you need to first think about the reasoning process in the mind and then conduct function calling to get the information or perform the actions if needed. \
The reasoning process and function calling are enclosed within <think> </think> and <tool_call> </tool_call> tags. \
The results of the function calls will be given back to you after execution, \
and you can continue to call functions until you get the final answer for the user's question. \
Finally, if you have got the answer, enclose it within \\boxed{{}} with latex format and do not continue to call functions, \
i.e., <think> Based on the response from the function call, I get the weather information. </think> The weather in Beijing on 2025-04-01 is \\[ \\boxed{{20C}} \\].

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""
    return tokenizer.apply_chat_template([
        {'role': 'system', 'content': prompt_template.format(func_schemas=data['extra_info']['func_schemas'])}, 
        {'role': 'user', 'content': data[prompt_key]}
    ], add_generation_prompt=True, tokenize=False) + "<think>"

def dpsk_zero_template(data, prompt_key, tokenizer):
    return """A conversation between User and Assistant. \
The user asks a question, and the Assistant solves it. \
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. \
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, \
respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. \
User: {prompt}. Assistant:""".format(prompt=data[prompt_key])

def dpsk_zero_with_box_template(data, prompt_key, tokenizer):
    return """A conversation between User and Assistant. \
The user asks a question, and the Assistant solves it. \
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. \
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, \
respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. \
In the last part of the answer, the final exact answer is enclosed within \\boxed{{}} with latex format, \
i.e., 'The final answer is \\[ \\boxed{{answer here}} \\]'. \
User: {prompt}. Assistant:""".format(prompt=data[prompt_key])


def dapo_template(data, prompt_key, tokenizer):
    prompt_template = '''Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n{prompt}\n\nRemember to put your answer on its own line after "Answer:".'''
    user_input = data[prompt_key]
    if isinstance(user_input, list):
        for c in user_input:
            if c['role'] == 'user':
                c['content'] = prompt_template.format(prompt=c['content'])
    elif isinstance(user_input, str):
        user_input = [{"role": "user", "content": prompt_template.format(prompt=user_input)}]
    else:
        raise ValueError(f"Invalid chat type: {type(user_input)}")
    return tokenizer.apply_chat_template(user_input, add_generation_prompt=True, tokenize=False)

prompt_template_dict = {}
prompt_template_dict['fc_sys_2'] = fc_sys_2
prompt_template_dict['dpsk_zero'] = dpsk_zero_template
prompt_template_dict['dpsk_zero_box'] = dpsk_zero_with_box_template
prompt_template_dict['dapo_template'] = dapo_template

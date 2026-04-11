import ujson
import json

_SYSTEM_ENV_FEEDBACK_PROMPT = """
Based on the task description and the reference agent-environment interaction, please generate the environment feedback for the agent's action and determine whether the current action has reached the final goal. If the agent's action has reached the final goal, please output "Task Completed!"; 
"""

_SYSTEM_ENV_FEEDBACK_PROMPT_NEW = """
Based on the task description and the reference agent-environment interaction, please generate the environment feedback for the agent's action and determine whether the current action has reached the final goal. If the agent's action has reached the final goal, please output "Task Completed!"; 
else, 
the feedback should in the following format:
"Observation: ..."

<instruction>
{instruction}
</instruction>

<previous_observation>
{observation}
</previous_observation>

<agent_action>
{agent_action}
</agent_action>

Environment Feedback:
"""

# for the frontier model
_ENV_FEEDBACK_PROMPT = {
    "alfworld": """
else, 
the feedback should in the following format:
"Observation: ..."

---
For instance, given the instruction "You are in the middle of a room. Looking quickly around you, you see a bathtubbasin 1, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 1, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.\n\nYour task is to: put a toiletpaper in toiletpaperhanger."
The agent action is "Thought: To solve the task, I need to find a toiletpaper and put it in the toiletpaperhanger. First, I need to find a toiletpaper. A toiletpaper is more likely to appear in the toilet (1), cabinet (1-2), countertop (1), garbagecan (1), sinkbasin (1), bathtubbasin (1). I can check one by one, starting with the toiletpaperhanger.\nAction: go to toiletpaperhanger 1."
The environment feedback could be:
"Observation: On the toiletpaperhanger 1, you see nothing."
---

<instruction>
{instruction}
</instruction>

<previous_observation>
{observation}
</previous_observation>

<agent_action>
{agent_action}
</agent_action>

Environment Feedback:
"""
}

_SYSTEM_RE_PLANNING_PROMPT = """
Based on the task description and previous observation of agent interactions with the environment, please generate all possible step-by-step global plans of the remaining execution steps, which serve as high-level, natural guidance to assist in planning.
"""


# include re-planning
_RE_PLANNING_PROMPT = {
    "alfworld": """
For the house holding task, the action list you can take:
1. go to recep
2. task obj from recep
3. put obj in/on recep
4. open recep
5. close recep
6. toggle obj recep
7. clean obj with recep
8. heat obj with recep
9. cool obj with recep
where obj and recep correspond to objects and receptacles.

The generated global plans should be written in the following format:
```json
["
Step 1: ...
Step 2: ...
...
", ...]
```

---
For instance, given the instruction "look at the CD under the desklamp" and the index of already executed steps is 0, the possible global plans could be:
```json
["
Step 1: Go to where the CD may be placed.
Step 2: Take the CD from where you found it.
Step 3: Go to where the desklamp is located.
Step 4: Use the CD to look at the desklamp.
", ...]
```
---

<instruction>
{instruction}
</instruction>

<previous_observation>
{observation}
</previous_observation>

All possible global plans of the remaining execution steps:
""",
}


# for agent stage 2
# for data construction
_AGENT_STAGE2_GENERAL_PROMPT = {
    "alfworld": """
<dataset>alfworld</dataset>Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish.

---
Here is an example.
{example}
---

Now, it's your turn and here is the instruction.
{instruction}
"""
}

# dynamic global planning stage
# generate the most suitable global plan based on the previous observation and the current instruction
# thought-action: _DYNAMIC_GLOBAL_PLAN_SELECTION_PROMPT + _DYNAMIC_GLOBAL_PLAN_SELECTION_PROMPT
_AGENT_STAGE2_GLOBAL_PLAN_GEN_PROMPT = {
    "alfworld": """
Based on the task description, the previous global plan, and accumalated observation of agent interactions with the environment, please generate the most suitable step-by-step global plan of the remaining execution steps, which serves as high-level, natural guidance to assist in planning.
Maintain the plan for all steps preceding the index of execution step index, while selectively modifying the plan for steps following the execution step index.

For the house holding task, the action list you can take:
1. go to recep
2. task obj from recep
3. put obj in/on recep
4. open recep
5. close recep
6. toggle obj recep
7. clean obj with recep
8. heat obj with recep
9. cool obj with recep
where obj and recep correspond to objects and receptacles.

The generated global plan should be written in the following format:
<global_plan>
Step 1: ...
Step 2: ...
...
</global_plan>

---
For instance, given the instruction "look at the CD under the desklamp" and the index of already executed steps is 0, the most suitable global plan could be:
<global_plan>
Step 1: Go to where the CD may be placed.
Step 2: Take the CD from where you found it.
Step 3: Go to where the desklamp is located.
Step 4: Use the CD to look at the desklamp.
</global_plan>
---

<previous_global_plan>
{global_plan}
</previous_global_plan>

<execution_step_index>
{execution_step_index}
</execution_step_index>

<previous_observation>
{observation}
</previous_observation>

The most suitable global plan for the current situation:
"""
}

# The most suitable global plan of the remaining execution steps:

# for agent stage 1
_GLOBAL_PLAN_SELECTION_PROMPT = {
    "alfworld": """
Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish.
For each of your turn, you will be given the observation of the last turn. You should choose from two actions: "Thought" or "Action". If you choose "Thought", you should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format: "Thought: your thoughts.\n Action: your next action"; If you choose "Action", you should directly output the action in this turn. Your output must strictly follow this format: "Action: your next action".
You are given several global plans serving as high-level, natural guidance to assist in planning. 
Please select the most suitable global plan from all available global plans to accomplish the task, and take action according to its instructions.

The available actions are:
1. go to recep
2. take obj from recep
3. put obj in/on recep
4. open recep
5. close recep
6. toggle obj recep
7. clean obj with recep
8. heat obj with recep
9. cool obj with recep
where obj and recep correspond to objects and receptacles.

After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the environment output "Nothing happened", that means the previous action is invalid and you should try more options.

Reminder:
1. The action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal.
2. Think when necessary, try to act directly more in the process.

---
Here is an example.
{example}
---

Here is the instruction.
<instruction>
{instruction}
</instruction>
""",

    "textcraft": """
You are given a few useful crafting recipes to craft items in Minecraft. Crafting commands are of the format "craft [target object] using [input ingredients]".
For each of your turn, you will be given the observation of the last turn. You should choose from two actions: "Thought" or "Action". If you choose "Thought", you should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format: "Thought: your thoughts.\n Action: your next action"; If you choose "Action", you should directly output the action in this turn. Your output must strictly follow this format: "Action: your next action". 
As for your action, you can "get an object (ingredients) from the inventory or the environment, look up the game "inventory" by inventory, or "craft" (target) using any of the crafting commands. You can use ONLY these crafting commands provided, do not use your own crafting commands. However, if the crafting command uses a generic ingredient like "planks", you can use special types of the same ingredient e.g. dark oak "planks" in the command instead. 
You are given several global plans serving as high-level, natural guidance to assist in planning. 
Please select the most suitable global plan from all available global plans to accomplish the task, and take action according to its instructions.

The available actions are:
1. go to recep
2. take obj from recep
3. put obj in/on recep
4. open recep
5. close recep
6. toggle obj recep
7. clean obj with recep
8. heat obj with recep
9. cool obj with recep
where obj and recep correspond to objects and receptacles.

After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the environment output "Nothing happened", that means the previous action is invalid and you should try more options.

Reminder:
1. The action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal.
2. Think when necessary, try to act directly more in the process.

---
Here is a demo of how to fetch and craft objects.
Crafting commands:
craft 3 dark oak sign using 6 dark oak planks, 1 stick
craft 4 dark oak planks using 1 dark oak log
craft 1 stick using 1 planks
craft 4 stick using 2 bamboo
craft 4 oak planks using 1 oak log
craft 1 dark oak fence using 2 stick, 4 dark oak planks
craft 1 warped stairs using 6 warped planks
craft 3 oak sign using 6 oak planks, 1 stick

Goal: craft dark oak sign
> Thought: I should check if I can fetch dark oak sign directly from the environment or the inventory.
OK.
> inventory
Inventory: [stick] (1) [dark oak planks] (8)
> get dark oak sign
Could not find dark oak sign
> Thought: I cannot get dark oak sign directly, I need to craft it. From the crafting commands, I can use: craft dark oak sign using 6 dark oak planks, 1 stick. Ingredients needed: 6 dark oak planks, 1 stick. Input assumption: I have all the necessary ingredients in my inventory. Let me verify this first.
OK.
> inventory
Inventory: [stick] (1) [dark oak planks] (8)
> Thought: I found my ingredients: 6 dark oak planks, 1 stick in my inventory. My assumption is true, I can proceed. I will use the crafting command: craft dark oak sign using 6 dark oak planks
OK.
> craft 1 dark oak sign using 6 dark oak planks, 1 stick
Crafted 1 minecraft:dark_oak_sign
> inventory
Inventory: [dark oak sign] (1)
> Thought: I now have dark oak sign in my inventory. Task Completed!
OK.
---

Now, it's your turn and here is the instruction.
<instruction>
{instruction}
</instruction>
""",

    "maze": """
You are an expert maze solver. Your objective is to reach the goal in as few steps as possible. At each step you will be given information about where the goal is, your current position, and the walls that surround you. When you move right you increase your y position by 1, when you move down you increase your x position by 1. Your possible actions are "move up", “move down”, “move left”, “move right”. Your output must strictly follow this format: "Thought: your thoughts.\n Action: your next action"; If you choose "Action", you should directly output the action in this turn. Your output must strictly follow this format: "Action: your next action". 
You are given several global plans serving as high-level, natural guidance to assist in planning. 
Please select the most suitable global plan from all available global plans to accomplish the task, and take action according to its instructions.

The available actions are:
1. move up
2. move down
3. move left
4. move right

Reminder:
1. The action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal.
2. Think when necessary, try to act directly more in the process.

---
Here is an example.
environment: Your current position is at position 5, 6. There is a wall above you.
action: move left
environment: Your current position is at position 5, 5. There are walls above you, below you.
action: move left
environment: Your current position is at position 5, 4. There are walls above you, below you.
action: move up
environment: Your current position is at position 5, 4. There are walls above you, below you.
action: move left
environment: Your current position is at position 5, 3. There are walls to your left, below you.
action: move down
environment: Your current position is at position 5, 3. There are walls to your left, below you.
action: move left
environment: Your current position is at position 5, 3. There are walls to your left, below you.
action: move down
environment: Your current position is at position 5, 3. There are walls to your left, below you.
action: move left
environment: Your current position is at position 5, 3. There are walls to your left, below you.
action: move right
environment: Your current position is at position 5, 4. There are walls above you, below you.
action: move down
environment: Your current position is at position 5, 4. There are walls above you, below you.
action: move right
environment: Your current position is at position 5, 5. There are walls above you, below you.
action: move right
environment: Your current position is at position 5, 6. There is a wall above you.
action: move down
environment: Your current position is at position 6, 6. There are walls to your right, to your left.
action: move down
environment: Your current position is at position 7, 6. There are walls to your right, to your left.
action: move right
environment: Your current position is at position 7, 6. There are walls to your right, to your left.
action: move down
environment: Success.
---

Now, it's your turn and here is the instruction.
<instruction>
{instruction}
</instruction>
""",
}

_DYNAMIC_GLOBAL_PLAN_SELECTION_PROMPT="""

Previous observation:
{observation}

This available global plans maybe helpful for you to complete the task:
{global_plans}

Your next action:
"""

_DYNAMIC_GLOBAL_PLAN_SELECTION_STAGE2_PROMPT={
    "alfworld": """
In this turn, you will be given the observation of the previous turns, where the environment provides you with feedback based on your previous action. If the environment output "Nothing happened", that means the previous action is invalid and you should try more options.
You should choose from two actions: "Thought" or "Action". If you choose "Thought", you should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format: "Thought: your thoughts.\n Action: your next action"; If you choose "Action", you should directly output the action in this turn. Your output must strictly follow this format: "Action: your next action".
You are given a global plan serving as high-level, natural guidance to assist in planning to accomplish the task, and take action according to its instructions.

The available actions are:
1. go to recep
2. take obj from recep
3. put obj in/on recep
4. open recep
5. close recep
6. toggle obj recep
7. clean obj with recep
8. heat obj with recep
9. cool obj with recep
where obj and recep correspond to objects and receptacles.

Reminder:
1. The action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal.
2. Think when necessary, try to act directly more in the process.

Previous observation:
{observation}

This available global plans maybe helpful for you to complete the task:
{global_plans}

Your next action:
""",

}


_SYSTEM_INITIAL_DYNAMIC_GLOBAL_PLAN_PROMPT="""
You are a helpful assistant that generates global plans for the following tasks. Your task is to generate all possible step-by-step global plans for the task, which serve as high-level, natural guidance to assist in planning.
"""

_INITIAL_DYNAMIC_GLOBAL_PLAN_PROMPT="""
Please generate all possible step-by-step global plans for the house holding task, which serve as high-level, natural guidance to assist in planning.
The generated global plans should be written in the following format:
```json
["
Step 1: ...
Step 2: ...
...
", ...]
```

All possible global plans:
"""


_GOLBAL_PLANNING_PROMPT = {
    "alfworld": """
Please generate all possible step-by-step global plans for the house holding task, which serve as high-level, natural guidance to assist in planning.
The action list you can take:
1. go to recep
2. task obj from recep
3. put obj in/on recep
4. open recep
5. close recep
6. toggle obj recep
7. clean obj with recep
8. heat obj with recep
9. cool obj with recep
where obj and recep correspond to objects and receptacles.

The generated global plans should be written in the following format:
```json
["
Step 1: ...
Step 2: ...
...
", ...]
```

---
For instance, given the instruction "look at the CD under the desklamp", the possible global plans could be:
```json
["
Step 1: Go to where the CD may be placed.
Step 2: Take the CD from where you found it.
Step 3: Go to where the desklamp is located.
Step 4: Use the CD to look at the desklamp.
", ...]
```
---

<task>
{task}
</task>

All possible global plans:
""",

"textcraft": """
Please generate all possible step-by-step global plans for the item crafting task, which serve as high-level, natural guidance to assist in planning. 
You are given a few useful crafting recipes to craft items in Minecraft. Craft command can be understood as follows: craft [target] using [ingredients], where target is item/object generated by the craft command as output and ingredient are the inputs. You are given an agent that can "craft" or "fetch" objects. You can take the help of crafting commands below to create new objects. Each global plan can use at most ONE of the provided crafting commands.

The generated global plans should be written in the following format:
```json
["
Step 1: ...
Step 2: ...
...
", ...]
```

---
For instance, given the crafting commands:
craft 3 dark oak sign using 6 dark oak planks, 1 stick
craft 4 dark oak planks using 1 dark oak log
craft 1 stick using 1 planks
craft 4 stick using 2 bamboo
craft 4 oak planks using 1 oak log
craft 1 dark oak fence using 2 stick, 4 dark oak planks
craft 1 warped stairs using 6 warped planks
craft 3 oak sign using 6 oak planks, 1 stick
Goal: craft dark oak sign.

The possible global plans could be:
```json
["
Step 1: Fetch 6 dark oak planks.
Step 2: Fetch 1 stick.
Step 3: Craft dark oak sign using 6 dark oak planks, 1 stick.
", ...]
```
---

<crafting_commands>
{task}
</crafting_commands>

All possible global plans:
""",

"maze": """
Please generate all possible step-by-step global plans for the maze solving task, which serve as high-level, natural guidance to assist in planning.
Your objective is to reach the goal in as few steps as possible. When you move right you increase your y position by 1, when you move down you increase your x position by 1.

The action list you can take:
1. move up
2. move down
3. move left
4. move right

The generated global plans should be written in the following format:
```json
["
Step 1: ...
Step 2: ...
...
", ...]
```

---
For instance, given the current environment state:
The goal is at position 8, 6. Your current position is at position 1, 1. There are walls to your left, above you, below you. The possible global plans could be:
```json
["
Step 1: move right (from 1, 1 to 1, 2)
Step 2: move right (from 1, 2 to 1, 3)
Step 3: move right (from 1, 3 to 1, 4)
Step 4: move down (from 1, 4 to 2, 4)
Step 5: move down (from 2, 4 to 3, 4)
Step 6: move down (from 3, 4 to 4, 4)
Step 7: move down (from 4, 4 to 5, 4)
Step 8: move down (from 5, 4 to 6, 4)
Step 9: move down (from 6, 4 to 7, 4)
Step 10: move down (from 7, 4 to 8, 4)
Step 11: move right (from 8, 4 to 8, 5)
Step 12: move right (from 8, 5 to 8, 6)
", ...]
```
---

<task>
{task}
</task>

All possible global plans:
""",
# wordle task (不适合 initial global plan, 适合 re-plan)
"wordle": """ 
Please generate all possible step-by-step global plans for the wordle task, which serve as high-level, natural guidance to assist in planning.
Your objective is to guess a hidden 5 letter word. You have 6 attempts to guess it correctly and you should try to guess it in as few attempts as possible. When guessing the word, you should format your word as a space separated sequence of letters, like "s h i r e" for example. After guessing the word, you will receive feedback from the game environment in the form of a sequence of 5 space separated letters like "b y g g b", where each letter indicates some information about the hidden word. The environment will return one of three letters - "b", "g", or "y" – for each letter in the word you guessed. We describe the meaning of each letter below:
"b": If the environment returns a “b”, it means that the letter at that position in your guessed word is not in the hidden word.
"y": If the environment returns a “y”, it means that the letter at that position in your guessed word is in the hidden word but is not in the correct position.
"g": If the environment returns a “g”, it means that the letter at that position in your guessed word is in the hidden word and is in the correct position.
As a note, if you guess an invalid word (e.g. not a 5 letter word or a word not in the vocabulary), the environment will respond with an “invalid word” message. In general though, you should use this information returned by the environment to update your belief about what the hidden word might be and adjust your next guess accordingly.

The generated global plans should be written in the following format:
```json
["
Step 1: ...
Step 2: ...
...
", ...]
```

Now let's start a new game. Remember, the word you guess should be strictly in the vocabulary. You should return your thought and your word strictly in the format mentioned above.
<task>
{task}
</task>

All possible global plans:
""",
}

# based on Qwen2.5-14B-Instruct (verify global plans generated by ds-R1)
# add original golden traj for reference
_GOLDEN_GLOBAL_PLAN_EVALUATION_PROMPT = """
Please act as a professional instruction evaluator and judge the global plan across the following three dimensions:
1. Correctness - Does the global plan accurately fulfill the task requirements?
2. Followability - Is the global plan clear, easy to understand, and are the steps reasonable?
3. Standardization - Does the global plan follow a consistent and standardized format?

For each dimension, please score the global plan on a scale of 1 to 5, where 1 indicates poor performance and 5 indicates excellent performance, and explain the reason.
Please output the result in JSON format, including the following fields:
```json
{{
    "correctness_score": xxx,
    "correctness_reason": "...",
    "followability_score": xxx,
    "followability_reason": "...",
    "standardization_score": xxx,
    "standardization_reason": "..."
}}

### Task Description: {instruction}

Below is the standard and detailed procedure for solving this task:
{conversation}

### Global Plan:
{global_plan}

### Judgment Result:
"""

# for Agent Stage 2, dynamic global plan evaluation
_SYSTEM_DYNAMIC_GOLDEN_GLOBAL_PLAN_EVALUATION_PROMPT = """
Please act as a professional instruction evaluator and judge the global plan across the following three dimensions:
1. Correctness - Based on the environment's feedback on the agent's actions in response to the current global plan guidance, does the global plan accurately fulfill the task requirements?
2. Followability - Based on the agent's adherence to the global plan, is the global plan clear, easy to understand, and are the steps reasonable?
3. Standardization - Does the global plan follow a consistent and standardized format?

For each dimension, please score the global plan on a scale of 1 to 5, where 1 indicates poor performance and 5 indicates excellent performance, and explain the reason.
Please output the result in JSON format, including the following fields:
```json
{{
    "correctness_score": xxx,
    "correctness_reason": "...",
    "followability_score": xxx,
    "followability_reason": "...",
    "standardization_score": xxx,
    "standardization_reason": "..."
}}
```
"""


_USER_DYNAMIC_GOLDEN_GLOBAL_PLAN_EVALUATION_PROMPT = """
### Task Description: 
{instruction}

### Global Plan:
{global_plan}

Agent's current action:
{agent_action}

Environment feedback:
{env_feedback}

### Judgment Result:
"""


_GOLDEN_GLOBAL_PLAN_EVALUATION_NO_TRAJ_PROMPT = """
Please act as a professional instruction evaluator and judge the global plan across the following three dimensions:
1. Correctness - Does the global plan accurately fulfill the task requirements?
2. Followability - Is the global plan clear, easy to understand, and are the steps reasonable?
3. Standardization - Does the global plan follow a consistent and standardized format?

For each dimension, please score the global plan on a scale of 1 to 5, where 1 indicates poor performance and 5 indicates excellent performance, and explain the reason.
Please output the result in JSON format, including the following fields:
```json
{{
    "correctness_score": xxx,
    "correctness_reason": "...",
    "followability_score": xxx,
    "followability_reason": "...",
    "standardization_score": xxx,
    "standardization_reason": "..."
}}

### Task Description: {task}

### Global Plan:
{global_plan}

### Judgment Result:
"""

# agent stage 2
_INTERACTION_PROMPT = {
    "alfworld": """
Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish.
For each of your turn, you will be given the observation of the last turn. You should choose from two actions: "Thought" or "Action". If you choose "Thought", you should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format: "Thought: your thoughts.\n Action: your next action"; If you choose "Action", you should directly output the action in this turn. Your output must strictly follow this format: "Action: your next action".

The available actions are:
1. go to recep
2. take obj from recep
3. put obj in/on recep
4. open recep
5. close recep
6. toggle obj recep
7. clean obj with recep
8. heat obj with recep
9. cool obj with recep
where obj and recep correspond to objects and receptacles.

After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the environment output "Nothing happened", that means the previous action is invalid and you should try more options.

Reminder:
1. The action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal.
2. Think when necessary, try to act directly more in the process.

---
Here is an example.
{example}
---

Now, it's your turn and here is the instruction.
{instruction}

This global plan maybe helpful for you to complete the task:
{global_plan}

Previous observation:
{observation}

Your next action:
""",

"textcraft": """
You are given a few useful crafting recipes to craft items in Minecraft. Crafting commands are of the format "craft [target object] using [input ingredients]". Every round I will give you an observation, you have to respond to an action based on the state and instruction. You can "get an object (ingredients) from the inventory or the environment, look up the game "inventory" by inventory, or "craft" (target) using any of the crafting commands. You can use ONLY these crafting commands provided, do not use your own crafting commands. However, if the crafting command uses a generic ingredient like "planks", you can use special types of the same ingredient e.g. dark oak "planks" in the command instead. For any other natural language or thoughts, use prefix 'think:'.
For each of your turn, you will be given the observation of the last turn.

---
Here is a demo of how to fetch and craft objects.
Crafting commands:
craft 3 dark oak sign using 6 dark oak planks, 1 stick
craft 4 dark oak planks using 1 dark oak log
craft 1 stick using 1 planks
craft 4 stick using 2 bamboo
craft 4 oak planks using 1 oak log
craft 1 dark oak fence using 2 stick, 4 dark oak planks
craft 1 warped stairs using 6 warped planks
craft 3 oak sign using 6 oak planks, 1 stick

Goal: craft dark oak sign
> think: I should check if I can fetch dark oak sign directly from the environment or the inventory.
OK.
> inventory
Inventory: [stick] (1) [dark oak planks] (8)
> get dark oak sign
Could not find dark oak sign
> think: I cannot get dark oak sign directly, I need to craft it. From the crafting commands, I can use: craft dark oak sign using 6 dark oak planks, 1 stick. Ingredients needed: 6 dark oak planks, 1 stick. Input assumption: I have all the necessary ingredients in my inventory. Let me verify this first.
OK.
> inventory
Inventory: [stick] (1) [dark oak planks] (8)
> think: I found my ingredients: 6 dark oak planks, 1 stick in my inventory. My assumption is true, I can proceed. I will use the crafting command: craft dark oak sign using 6 dark oak planks
OK.
> craft 1 dark oak sign using 6 dark oak planks, 1 stick
Crafted 1 minecraft:dark_oak_sign
> inventory
Inventory: [dark oak sign] (1)
> think: I now have dark oak sign in my inventory. Task Completed!
OK.
---

You can use the provided crafting commands to accomplish the goal. When you the desired item in your inventory, think: Task Completed! If you have tried your best but cannot proceed, think: task failed!
Now, it's your turn and here are the crafting commands and the goal.
{task}

This global plan maybe helpful for you to complete the task:
{global_plan}

Previous observation:
{observation}

Your next action:
""",

"maze": """
You are an expert maze solver. Your objective is to reach the goal in as few steps as possible. At each step you will be given information about where the goal is, your current position, and the walls that surround you. When you move right you increase your y position by 1, when you move down you increase your x position by 1. Your possible actions are "move up", “move down”, “move left”, “move right”. Formally, your return should be in this format:

Thought: <Your Thought>
Action: <Your Action>

The available actions are:
1. move up
2. move down
3. move left
4. move right

Reminder:
1. The action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal.
2. Think when necessary, try to act directly more in the process.

---
Here is an example.
environment: Your current position is at position 5, 6. There is a wall above you.
action: move left
environment: Your current position is at position 5, 5. There are walls above you, below you.
action: move left
environment: Your current position is at position 5, 4. There are walls above you, below you.
action: move up
environment: Your current position is at position 5, 4. There are walls above you, below you.
action: move left
environment: Your current position is at position 5, 3. There are walls to your left, below you.
action: move down
environment: Your current position is at position 5, 3. There are walls to your left, below you.
action: move left
environment: Your current position is at position 5, 3. There are walls to your left, below you.
action: move down
environment: Your current position is at position 5, 3. There are walls to your left, below you.
action: move left
environment: Your current position is at position 5, 3. There are walls to your left, below you.
action: move right
environment: Your current position is at position 5, 4. There are walls above you, below you.
action: move down
environment: Your current position is at position 5, 4. There are walls above you, below you.
action: move right
environment: Your current position is at position 5, 5. There are walls above you, below you.
action: move right
environment: Your current position is at position 5, 6. There is a wall above you.
action: move down
environment: Your current position is at position 6, 6. There are walls to your right, to your left.
action: move down
environment: Your current position is at position 7, 6. There are walls to your right, to your left.
action: move right
environment: Your current position is at position 7, 6. There are walls to your right, to your left.
action: move down
environment: Success.
---

Now, it's your turn and here is the task.
{task}

This global plan maybe helpful for you to complete the task:
{global_plan}

Previous observation:
{observation}

Your next action:
""",
 
"wordle": """
You are an expert wordle player. Welcome to the game of Wordle. Your objective is to guess a hidden 5 letter word. You have 6 attempts to guess it correctly and you should try to guess it in as few attempts as possible. When guessing the word, you should format your word as a space separated sequence of letters, like "s h i r e" for example. After guessing the word, you will receive feedback from the game environment in the form of a sequence of 5 space separated letters like "b y g g b", where each letter indicates some information about the hidden word. The environment will return one of three letters - "b", "g", or "y" – for each letter in the word you guessed. We describe the meaning of each letter below:
"b": If the environment returns a “b”, it means that the letter at that position in your guessed word is not in the hidden word.
"y": If the environment returns a “y”, it means that the letter at that position in your guessed word is in the hidden word but is not in the correct position.
"g": If the environment returns a “g”, it means that the letter at that position in your guessed word is in the hidden word and is in the correct position.
As a note, if you guess an invalid word (e.g. not a 5 letter word or a word not in the vocabulary), the environment will respond with an “invalid word” message. In general though, you should use this information returned by the environment to update your belief about what the hidden word might be and adjust your next guess accordingly.

---
Here is the complete list of valid vocabulary words that are accepted by the game:\n```\nelect\nnymph\nsolar\npence\nglade\nulcer\nsolve\ncoupe\nheath\nchirp\nhunch\nbacon\nbaggy\ntacit\nabled\nfried\nrecut\nretry\nivory\nunity\napart\naltar\nslyly\nfudge\nswine\navian\nstole\nsniff\nblush\nbraid\nalgae\nniece\nswill\nclung\nwrist\nnoble\nnorth\nelder\npolka\nspilt\nmedic\nladen\nblade\ntacky\ntrove\ncamel\nstorm\nhello\nindex\nelbow\nidler\nknead\nchaff\ngenie\ncreak\nmamma\ngavel\ntheft\nswish\nperky\nrodeo\ncacao\nlipid\nskate\nsalty\nhedge\nhyena\nrange\nhumor\nspiny\nruddy\nprime\nbluff\nhouse\namber\nrevel\ndrink\nframe\ngaudy\ninner\nretro\nabide\nplied\ntiger\nidiot\nlunch\ndopey\ntwirl\nseven\nflung\nfella\nsmash\nfence\nflush\nfault\nalloy\ngonad\nboard\nexact\nrumba\naloft\nmince\nwryly\nmodel\nclean\nhappy\nknoll\nvigil\nsmall\nequip\nknelt\nacute\nbroom\nproof\nperil\nhatch\nsaucy\ntough\nmoral\nnoose\nother\nnomad\nboney\nlabel\ngusto\nscoff\nspill\ncrimp\nspice\nworth\nrecur\nblare\nvixen\ncedar\ntopic\ncivil\nfugue\nrhino\nspeak\nspawn\nruder\nfiber\nretch\ngrave\ncider\namong\ncheat\ntrial\npixie\nchase\ngeese\nloyal\nhunky\nquota\ncynic\ntract\nbuggy\nminim\nlogic\nimpel\nshoot\nstomp\novary\nadore\nrinse\nspear\nattic\nguard\nkneed\nannoy\nneedy\nmanly\nbully\nshirk\nprank\nshell\ncough\nspurn\ntorso\nsnort\nmoist\nscrum\nraise\nhoney\ntrope\nlocal\nerupt\nlivid\nshied\nfelon\nmecca\nshake\nreach\nstyle\ncovey\ndelve\nunset\njiffy\ncrash\nbarge\ntoast\nallot\naudio\nprint\nbadly\ntrust\nhabit\nhoard\nshard\nazure\nlucky\nblind\nhefty\nrebel\nethos\nfrown\ngodly\nvalor\nslate\naider\nstoop\nfinch\naback\nwomen\nmaker\ngully\nnasty\nprick\nwrath\nfrisk\nspell\nroyal\nlefty\nslept\npetal\nstunk\nsmack\nfluid\nclear\nlodge\nproxy\npanic\nthorn\nshock\nhotly\ngamma\ngypsy\nfarce\nkoala\nbroth\nstink\nmerit\nexcel\nangst\nslurp\ncould\ncreep\npause\nsalsa\ndandy\ntithe\noptic\natoll\nbrute\nwrung\nchurn\nnatal\nensue\nvying\nbrawl\ncloth\nsoggy\nbough\nglaze\nchock\nteary\njetty\nlight\nadobe\nleant\nbrink\nforgo\nasset\nbawdy\ntaste\ngloat\ngoing\nmorph\nswamp\nrobot\nstalk\nrajah\nultra\nliver\nfizzy\nditch\nsushi\ndusty\nsheep\nlimbo\ngroom\nbiome\ndebug\nslain\nethic\nfoist\nwound\nsloop\ngamer\npoint\npesky\ntwist\nvoter\naisle\ntally\nteeth\ntrail\nminer\ncaulk\nsynod\ntouch\nwhack\nguava\ndutch\njuicy\nyouth\nbayou\ndenim\ndisco\nblurt\nrocky\nfrost\ngooey\ndodge\nforte\nsnowy\nshrug\nrelax\ntiara\ndepth\nstuff\nwince\ncopse\nalive\ndecry\nmania\ngrant\nalter\nelegy\nicily\nblunt\nbasil\nbrush\nvoice\ntotal\nchili\nworry\nseedy\nthose\nsmoky\ncream\nroger\nloose\nshire\nallow\ngroin\nchina\npinky\ngrime\nacorn\nuncut\ndonut\npleat\nglass\nwhere\nliken\nnadir\nflume\ntwang\nsugar\nnavel\nbeing\npushy\nfocal\nhymen\nbaler\ntweet\nvideo\nskill\nsonar\naloud\nmount\nspurt\nanode\nshack\nbreak\nsatyr\nbelly\naroma\nawait\nneigh\nskull\nderby\njerky\nhumph\ninter\nsober\nicing\nsound\ntrace\nblame\nplank\naugur\nrobin\nmadly\ncheck\nmirth\ndelta\ncliff\nfacet\nevoke\nclown\novert\nfuror\nstart\nzonal\nblond\ndoing\ncello\ncreme\nmotto\n```\n\nHere is an example. If the current status of the game is given as:\n```\nguess 1: p a n i c\nfeedback 1: b b y b b\nguess 2: f e l o n\nfeedback 2: g b b y g\n```\nBased on the feedback from the environment, you know that the first letter is \"f\", the last letter is \"n\", and there is an \"o\" somewhere in the word, but it is not in the second to last position. You also know that there is not a \"p\", \"a\", \"i\", \"c\", \"e\", or \"l\" in the word. Knowing this, you might guess the next word to be:\nThought:\nI know that the first letter is \"f\", the last letter is \"n\", and there is an \"o\" somewhere in the word, but it is not in the second to last position. I also know that there is not a \"p\", \"a\", \"i\", \"c\", \"e\", or \"l\" in the word. A good word from the vocabulary to try might therefore be \"f r o w n\", since it is in the vocabulary, meets all known letter constraints, and we get to gain more information about the position of \"o\". Therefore this is a good guess to try next.\n\nAction:\nf r o w n\n\nFormally, your return should be in this format:\nThought:\n<Your Thought>\n\nAction:\n<The Word You Guess>\n\nThe guessed word is in the vocabulary, meets all known letter constraints, and we get to gain more information about the position of \"o\", so it is a good guess to try next.\n\nNow let's start a new game. Remember, the word you guess should be strictly in the vocabulary. You should return your thought and your word strictly in the formation mentioned above.
---

Now, it's your turn and here is the task.
{task}

This global plan maybe helpful for you to complete the task:
{global_plan}

Previous observation:
{observation}

Your next action:
""",
}


_TOOL_SIMULATION_PROMPT = {
    "alfworld": """

"""
}



_ANSWER_PROMPT = """Please Answer the following questions:"""
_ANSWER_WITH_THINK_PROMPT_EN = """You are a medical expert.  
Given a question, you should answer it by first thinking about the reasoning process in the mind and then providing the final answer. 
Please answer the question in the format of <think>...</think><answer>...</answer>. 
That is, <think>Here is the reasoning process</think><answer>Here is the answer</answer>. 
You should perform thinking with decomposing, reflecting, brainstorming, verifying, refining, and revising. 
Besides, you can perform searching for uncertain knowledge if necessary with the format of <|begin_of_query|> search query <|end_of_query|> during your thinking process.
Then, the search system will provide you with the retrieval information with the format of <|begin_of_documents|> search results <|end_of_documents|>.
The answer needs to summarize the reasoning process and give the final answer.

Quesion: {question}
Response:
""" 

_ANSWER_WITH_THINK_PROMPT_EN_RESTART = """You are a medical expert.  
Given a question, you should answer it by first thinking about the reasoning process in the mind and then providing the final answer. 
Please answer the question in the format of <think>...</think><answer>...</answer>. 
That is, <think>Here is the reasoning process</think><answer>Here is the answer</answer>. 
You should perform thinking with decomposing, reflecting, brainstorming, verifying, refining, and revising. 
Besides, you can perform searching for uncertain knowledge if necessary with the format of <|begin_of_query|> search query <|end_of_query|> during your thinking process.
Then, the search system will provide you with the retrieval information with the format of <|begin_of_documents|> search results <|end_of_documents|>.
The answer needs to summarize the reasoning process and give the final answer.
You are required to continue your reasoning and response in conjunction with the existing answer and retrieved information, ensuring that the subsequent answers you generate maintain coherence with the previously generated answers.

Question: {question} 
Existing Answer and Retrieved Information: {existing_answer} 
Response:
"""
# Retrieved Information: {retrieved_docs} 

_ANSWER_WITH_THINK_PROMPT_ZH = """你是一名医学专家。
在回答问题时，你应该首先思考这个问题背后的推理过程，然后提供最终答案。
请按<think>...</think><answer>...</answer>标签格式回答问题。
即<think>这里是推理过程</think><answer>这里是答案</answer>。
你应该通过分解、反思、头脑风暴、验证、精炼和修订来进行思考。此外，如果需要查询不确定的知识，可以在思考过程中使用以下格式进行搜索：<|begin_of_query|> 搜索关键词在这里 <|end_of_query|>。
然后，搜索系统会以以下格式提供检索信息：<|begin_of_documents|> 搜索结果 <|end_of_documents|>。
答案需要对推理过程进行总结并给出最终答案。

问题: {question}
回答:
"""

# 后续的 response 需要与更新后的 prompt 组成一个闭环 <think></think><answer></answer>

_ANSWER_WITH_THINK_PROMPT_ZH_RESTART = """你是一名医学专家。
在回答问题时，你应该首先思考这个问题背后的推理过程，然后提供最终答案。
请按<think>...</think><answer>...</answer>标签格式回答问题。
即<think>这里是推理过程</think><answer>这里是答案</answer>。
你应该通过分解、反思、头脑风暴、验证、精炼和修订来进行思考。此外，如果需要查询不确定的知识，可以在思考过程中使用以下格式进行搜索：<|begin_of_query|> 搜索关键词在这里 <|end_of_query|>。
然后，搜索系统会以以下格式提供检索信息：<|begin_of_documents|> 搜索结果 <|end_of_documents|>。
答案需要对推理过程进行总结并给出最终答案。
你需要结合检索到的信息继续你的推理和回答，使得你继续生成的回答与此前已经生成的回答具有连贯性。

问题: {question} 
已经生成的回答及检索信息: {existing_answer} 
回答:
"""
# 检索到的信息: {retrieved_docs} 

_ANSWER_WITH_THINK_PROMPT_ZH_HK = """你是一名醫學專家。
在回答問題時，你應該首先思考這個問題背後的推理過程，然後提供最終答案。
請按<think>...</think><answer>...</answer>標籤格式回答問題。
即<think>這裡是推理過程</think><answer>這裡是答案</answer>。
你應該通過分解、反思、頭腦風暴、驗證、精煉和修訂來進行思考。此外，如果需要查詢不確定的知識，可以在思考過程中使用以下格式進行搜索：<|begin_of_query|> 搜索關鍵詞在這裡 <|end_of_query|>。
然後，搜索系統會以以下格式提供檢索信息：<|begin_of_documents|> 搜索結果 <|end_of_documents|>。
答案需要對推理過程進行總結並給出最終答案。

問題: {question}
回答:
"""

_ANSWER_WITH_THINK_PROMPT_ZH_HK_RESTART = """你是一名醫學專家。
在回答問題時，你應該首先思考這個問題背後的推理過程，然後提供最終答案。
請按<think>...</think><answer>...</answer>標籤格式回答問題。
即<think>這裡是推理過程</think><answer>這裡是答案</answer>。
你應該通過分解、反思、頭腦風暴、驗證、精煉和修訂來進行思考。此外，如果需要查詢不確定的知識，可以在思考過程中使用以下格式進行搜索：<|begin_of_query|> 搜索關鍵詞在這裡 <|end_of_query|>。
然後，搜索系統會以以下格式提供檢索信息：<|begin_of_documents|> 搜索結果 <|end_of_documents|>。
答案需要對推理過程進行總結並給出最終答案。
你需要結合檢索到的資訊繼續你的推理和回答，使得你繼續生成的回答與此前已經生成的回答具有連貫性。

問題：{question}  
已經生成的回答與檢索資訊：{existing_answer}  
回答：
"""
# 檢索到的資訊：{retrieved_docs}  

_ANSWER_DIRECT_PROMPT_EN = """A conversation between User and Assistant.
The user asks a question, and the Assistant solves it.

User: {prompt}
Assistant:
"""
_ANSWER_DIRECT_PROMPT_ZH = """用户和AI助手之间的对话。
用户提出问题，助手解决问题。
助手首先在心中思考推理过程，然后提供答案给用户。

用户: {prompt}
助手:
"""
_ANSWER_DIRECT_PROMPT_ZH_HK = """用戶和AI助手之間的對話。
用戶提出問題，助手解決問題。
助手首先在心中思考推理過程，然後提供答案給用戶。

用戶: {prompt}
助手:
"""
_JUDGE_UNIQUE_ANSWER_PROMPT_ZH = """请判断下面的选择题是否适合用来转换为开放性问题。可转换的问题需要满足下面几个条件
1. 去掉选项后，问题本身依然成立，且答案唯一，且正是该答案
2. 问题和答案都不能存在歧义
3. 问题只能有唯一最优答案，不能是范围，不能是模糊的值
4. 反选类问题，即选出不符合条件的选项，或者"最不可能的是"，也不适合转换
5. 其他情况，请发挥你的逻辑判断力，进行判断

下面会提供一个选择题，以及正确答案，以及其他误导选项

请回复 true / false
true 表示具有唯一答案，false 则相反
请给出判断的理由

### 返回格式
```json
{{
    "unique": true,
    "reason": "因为..."
}}
```

### 问题: 
{question}
### 正确答案:
{answer}
### 误导选项:
{options}

### 判断结果:
"""
_JUDGE_UNIQUE_ANSWER_PROMPT_EN = """Please judge whether the following multiple-choice question is suitable for conversion into an open-ended question.
The question to be converted must meet the following conditions:

1. After removing the options, the question itself remains valid, and the answer is unique and correct.
2. There is no ambiguity in the question and answer.
3. The question must have a unique optimal answer, not a range or a vague value.
4. Questions with negative options, such as selecting the option that does not meet the conditions or the "least likely" option, are not suitable for conversion.
5. In other cases, please use your logical judgment to make a decision.

### Return Format
```json
{{
    "unique": true,
    "reason": "Because..."
}}
```

### Question:
{question}
### Correct Answer:
{answer}
### Misleading Options:
{options}

### Judgment Result:
"""

_VERIFY_PROMPT_ZH = """下面会给你一个问题，并给出一名学生的回答，以及给出该问题的正确且唯一答案，以及其他误导选项。
请你对比该学生的回答与正确答案，然后分析该学生的回答是否正确。
学生的回答中，<think>...</think>标签包裹，表示思考过程，<answer>...</answer>标签包裹，表示最终回答。
在模型的思考过程中，对于不确定的部分，它会使用 <|begin_of_query|> 搜索关键词 <|end_of_query|> 标签进行包裹，以便后续查询相关信息。当查询到相关内容后，模型将基于检索到的信息继续其推理过程。
需要注意的是，<|begin_of_documents|> 搜索结果 <|end_of_documents|> 中的内容是外部检索的结果，并非模型自己生成的。因此，这些内容不应被用于衡量模型本身的逻辑推理能力，而仅作为评估其处理和整合外部信息的能力的依据。

请基于给出的正确答案，对学生的思考过程与回答进行判断，评为1~5分，并说明理由。
5分答案标准：
1. 思考过程逻辑严丝无缝，推理过程具体且清晰
2. 最终回答与正确答案意思一致，允许是正确答案的同义词、缩写等，但不能够包含错误选项
3~4分答案标准：
1. 思考过程逻辑合理
2. 最终回答与正确答案意思一致，允许有一定补充，只要不与正确答案冲突，且正确答案占主要地位，而不是其他误导选项
1~2分答案标准：
1. 思考过程逻辑不清晰
2. 最终回答与正确答案不一致，或者包含错误选项
3. 包含乱码、格式错误、乱序、多余无关信息

请按照下面的json格式返回
```json
{{
    "score": xxx,
    "reason": "..."
}}

### 问题:
{question}
### 学生回答:
{answer}
### 正确答案:
{correct_answer}
### 误导选项:
{misleading_options}

### 判断结果:
"""

_VERIFY_PROMPT_RAREARENA_ZH = """下面会给你一个问题，并给出一名学生的回答，以及给出该问题的正确且唯一答案。
请你对比该学生的回答与正确答案，然后分析该学生的回答是否正确。
学生的回答中，<think>...</think>标签包裹，表示思考过程，<answer>...</answer>标签包裹，表示最终回答。
在模型的思考过程中，对于不确定的部分，它会使用 <|begin_of_query|> 搜索关键词 <|end_of_query|> 标签进行包裹，以便后续查询相关信息。当查询到相关内容后，模型将基于检索到的信息继续其推理过程。
需要注意的是，<|begin_of_documents|> 搜索结果 <|end_of_documents|> 中的内容是外部检索的结果，并非模型自己生成的。因此，这些内容不应被用于衡量模型本身的逻辑推理能力，而仅作为评估其处理和整合外部信息的能力的依据。

请基于给出的正确答案，对学生的思考过程与回答进行判断，评为1~5分，并说明理由。
5分答案标准：
1. 思考过程逻辑严丝无缝，推理过程具体且清晰
2. 最终回答与正确答案意思一致，允许是正确答案的同义词、缩写等
3~4分答案标准：
1. 思考过程逻辑合理
2. 最终回答与正确答案意思一致，允许有一定补充，只要不与正确答案冲突，且正确答案占主要地位
1~2分答案标准：
1. 思考过程逻辑不清晰
2. 最终回答与正确答案不一致
3. 包含乱码、格式错误、乱序、多余无关信息

请按照下面的json格式返回
```json
{{
    "score": xxx,
    "reason": "..."
}}

### 问题:
{question}
### 学生回答:
{answer}
### 正确答案:
{correct_answer}

### 判断结果:
"""

_VERIFY_PROMPT_EN = """You will be given a question, a student's answer, the correct and unique answer to the question, and other misleading options.
Please compare the student's answer with the correct answer and analyze whether the student's answer is correct.
In the student's answer, the <think>...</think> tag wraps the thinking process, and the <answer>...</answer> tag wraps the final answer.
During the model's reasoning process, uncertain parts are encapsulated using the <|begin_of_query|> search query <|end_of_query|> tags to facilitate subsequent information retrieval. Once relevant content is retrieved, the model continues its reasoning based on the retrieved information.
It is important to note that the content within <|begin_of_documents|> search results <|end_of_documents|> represents externally retrieved information and is not generated by the model itself. Therefore, this content should not be used to assess the model's logical reasoning ability. Instead, it should be used solely to evaluate the model's capability to process and integrate external information.

Please judge the student's thinking process and answer based on the correct answer, rate it from 1 to 5 points, and explain the reason.
5-point answer criteria:
1. The thinking process is logical and seamless, and the reasoning process is specific and clear.
2. The final answer is consistent with the correct answer, allowing synonyms, abbreviations, etc. of the correct answer, but cannot contain incorrect options.
3-4 point answer criteria:
1. The thinking process is reasonable.
2. The final answer is consistent with the correct answer, allowing some supplements, as long as they do not conflict with the correct answer, and the correct answer is the main one, not other misleading options.
1-2 point answer criteria:
1. The thinking process is not clear.
2. The final answer is inconsistent with the correct answer or contains incorrect options.
3. Contains garbled characters, format errors, disorder, and irrelevant information.

Please return in the following json format
```json
{{
    "score": xxx,
    "reason": "..."
}}

### Question:
{question}
### Student's Answer:
{answer}
### Correct Answer:
{correct_answer}
### Misleading Options:
{misleading_options}

### Judgment Result:
"""

_VERIFY_PROMPT_RAREARENA_EN = """You will be given a question, a student's answer, the correct and unique answer to the question.
Please compare the student's answer with the correct answer and analyze whether the student's answer is correct.
In the student's answer, the <think>...</think> tag wraps the thinking process, and the <answer>...</answer> tag wraps the final answer.
During the model's reasoning process, uncertain parts are encapsulated using the <|begin_of_query|> search query <|end_of_query|> tags to facilitate subsequent information retrieval. Once relevant content is retrieved, the model continues its reasoning based on the retrieved information.
It is important to note that the content within <|begin_of_documents|> search results <|end_of_documents|> represents externally retrieved information and is not generated by the model itself. Therefore, this content should not be used to assess the model's logical reasoning ability. Instead, it should be used solely to evaluate the model's capability to process and integrate external information.

Please judge the student's thinking process and answer based on the correct answer, rate it from 1 to 5 points, and explain the reason.
5-point answer criteria:
1. The thinking process is logical and seamless, and the reasoning process is specific and clear.
2. The final answer is consistent with the correct answer, allowing synonyms, abbreviations, etc. of the correct answer.
3-4 point answer criteria:
1. The thinking process is reasonable.
2. The final answer is consistent with the correct answer, allowing some supplements, as long as they do not conflict with the correct answer, and the correct answer is the main one.
1-2 point answer criteria:
1. The thinking process is not clear.
2. The final answer is inconsistent with the correct answer.
3. Contains garbled characters, format errors, disorder, and irrelevant information.

Please return in the following json format
```json
{{
    "score": xxx,
    "reason": "..."
}}

### Question:
{question}
### Student's Answer:
{answer}
### Correct Answer:
{correct_answer}

### Judgment Result:
"""

_REFORMAT_PROMPT_ZH = """请你将下面的选择题转换为开放性问题。
请尽可能保留问题前面部分的内容不变，将最后的问句修改为开放性询问的方式，即将原本"下列哪一项是"，"最可能的选项是"等等方式进行修改。
修改后的问题也要保持语义通畅无歧义，问题答案应该与原问题的正确答案一致。
修改后的问题使用的语言要和原问题保持一致。

请按下面的 json 格式返回
```json
{{
    "question": "修改后的问题"
}}

### 原问题:
{question}
### 答案:
{answer}

### 修改后的问题:
"""

_REFORMAT_PROMPT_EN = """Please convert the following multiple-choice question into an open-ended question.
Please try to keep the content of the question unchanged as much as possible, and modify the last question into an open-ended inquiry, that is, modify the original "Which of the following is", "The most likely option is", etc.
The modified question should also be semantically smooth and unambiguous, and the answer to the question should be consistent with the correct answer to the original question.
The language used in the modified question should be consistent with the original question.

Please return in the following json format
```json
{{
    "question": "The modified question"
}}

### Original Question:
{question}
### Answer:
{answer}

### Modified Question:
"""

_REFORMAT_PROMPT_ZH_HK = """請你將下面的選擇題轉換為開放性問題。
請盡可能保留問題前面部分的內容不變，將最後的問句修改為開放性詢問的方式，即將原本"下列哪一項是"，"最可能的選項是"等等方式進行修改。
修改後的問題也要保持語義通暢無歧義，問題答案應該與原問題的正確答案一致。
修改後的問題使用的語言要和原問題保持一致。

請按下面的 json 格式返回
```json
{{
    "question": "修改後的問題"
}}

### 原問題:
{question}
### 答案:
{answer}

### 修改後的問題:
"""
import ujson
import json

_EXTRACT_REASONING_TRAJ_ZH = """你是一名医学专家。
给定一段关于解决医学问题的思考过程，这段话包括 <think>...</think><answer>...</answer> 标签。在 <think> 部分中，作者展示了其思考过程，并可能使用 <search> 搜索关键词 </search> 来标记需要查询的不确定知识。搜索系统会以 <document> 搜索结果 </document> 提供相关检索信息。
请根据给出的段落，总结并抽象出作者的思考步骤（思考轨迹）。

### 返回格式:
1. xxx.
2. xxx.
...
n. xxx.

要求：
1. 每个步骤应简明扼要，概括关键思考点，各个步骤以数字编号
2. 如果涉及查询和检索结果，需明确说明查询内容及其对思考的影响
3. 保持步骤的顺序和逻辑流程，确保逻辑清晰，能够反映思维路径
4. 特别关注医学问题的专业性，确保总结符合医学领域的逻辑和术语

思考段落: {reasoning_template}
思考轨迹:
"""

_EXTRACT_REASONING_TRAJ_EN = """
You are a medical expert.
Given a reasoning process for solving a medical problem in the format of <think>...</think> and <answer>...</answer>. The content within <think>...</think> demonstrates the thought process and may use <search> search query </search> to mark uncertain knowledge requiring searching. The search system provides relevant information in the format of <document> search results </document>. 
Summarize and abstract the reasoning steps (reasoning trajectory) from the provided passage.

### Return Format:
1. xxx.
2. xxx.
...
n. xxx.

Requirements:
1. Each step should be concise, summarizing key points of reasoning. Number each step sequentially.
2. If queries and retrieval results are involved, clearly explain the query content and its impact on the reasoning process.
3. Maintain the order and logical flow of the steps to ensure clarity and reflect the thought process accurately.
4. Pay special attention to the professionalism of medical issues, ensuring the summary aligns with the logic and terminology of the medical field.

Reasoning process: {reasoning_template}
Reasoning trajectory:
"""

_EXTRACT_REASONING_KG_ZH = """你是一名医学专家。
给定一段关于解决医学问题的思考过程，这段话包括 <think>...</think><answer>...</answer> 标签。在 <think> 部分中，作者展示了其思考过程，并可能使用 <search> 搜索关键词 </search> 来标记需要查询的不确定知识。搜索系统会以 <document> 搜索结果 </document> 提供相关检索信息。
你的任务是对于给定的医学思考过程，将其中的重要概念、关系和属性提取出来，并以知识图谱的格式进行表示。

### 返回格式:
```json
[[
    "entity1", "relationship", "entity2", "if_retrieval"]
], ...]
```

定义：
1. entity1、relationship、entity2：需明确提取自自然语言中的实体或关系（如“患者”、“有症状”、“发热”）。
2. if_retrieval：布尔值（True/False），表示当前三元组的实体或关系是否通过检索（即来自 <document> 搜索结果 </document> 的内容）

处理规则：
1. 总体：仅提取与医学问题解决相关的实体和关系，忽略与医学问题无关的背景信息，提取的知识图谱应简洁明了，避免冗余，同时应基于事实，避免主观推测
2. 实体提取：必须是具体医学概念（疾病、症状、药物、检查项目等），排除模糊描述（如"某些情况""相关因素"等），需合并同义表述（如"心梗"和"心肌梗死"统一为后者）
3. 关系定义：使用动词短语（引发、抑制、导致、伴随等），必须明确方向性（A->B 或 B->A需符合医学逻辑），否定关系需显式标注（如"不引起""排除"）
4. 检索标记判定：若实体或关系直接来自 <document> 搜索结果 </document> 的检索结果，则标记 if_retrieval=True，否则 False；检索关键词本身不作为实体或关系
5. 特殊处理：需保留推理过程中的假设关系（标记为[假设]），时间关系需转换为医学时间表述（急性/慢性/持续期等），概率性结论需量化为（高/中/低）风险等级

思考段落: {reasoning_template}
知识图谱:
"""

_EXTRACT_REASONING_KG_EN = """
You are a medical expert.
Given a reasoning process for solving a medical problem in the format of <think>...</think> and <answer>...</answer>. The content within <think>...</think> demonstrates the thought process and may use <search> search query </search> to mark uncertain knowledge requiring searching. The search system provides relevant information in the format of <document> search results </document>. 
Your task is to extract the important concepts, relationships, and attributes from the given medical reasoning process and represent them in the format of a knowledge graph.

### Return Format:
```json
[[
    "entity1", "relationship", "entity2", "if_retrieval"]
], ...]
```

Definitions:
1. entity1, relationship, entity2: Clearly extract entities or relationships from the natural language (e.g., "patient", "has symptom", "fever").
2. if_retrieval: A boolean value (True/False) indicating whether the entity or relationship is retrieved from the search results within <document>...</document>.

Processing rules:
1. Overall: Extract only entities and relationships related to solving medical problems, ignoring irrelevant background information. The knowledge graph should be concise and clear, avoiding redundancy, and based on facts, avoiding subjective speculation.
2. Entity extraction: Must be specific medical concepts (diseases, symptoms, drugs, examination items, etc.), excluding vague descriptions (such as "some conditions" or "related factors"). Synonymous expressions should be merged (e.g., "myocardial infarction" instead of "heart attack").
3. Relationship definition: Use verb phrases (cause, inhibit, lead to, accompany, etc.) and must have a clear directionality (A->B or B->A should conform to medical logic). Negative relationships should be explicitly marked (e.g., "does not cause", "rule out").
4. Retrieval marking determination: If an entity or relationship is directly from the <document> search results </document>, mark "if_retrieval=True"; otherwise, "if_retrieval=False". The search keywords themselves should not be considered as entities or relationships.
5. Special handling: Retain hypothetical relationships in the reasoning process (marked as [hypothesis]), convert time relationships to medical temporal expressions (acute/chronic/ongoing period, etc.), and quantify probabilistic conclusions as (high/medium/low) risk levels.

Reasoning process: {reasoning_template}
Knowledge graph:
"""

_ANSWER_PROMPT = """Please Answer the following questions:"""
_ANSWER_WITH_THINK_PROMPT_EN = """You are a medical expert.  
Given a question, you should answer it by first thinking about the reasoning process in the mind and then providing the final answer. 
Please answer the question in the format of <think>...</think><answer>...</answer>. 
That is, <think>Here is the reasoning process</think><answer>Here is the answer</answer>. 
You should perform thinking with decomposing, reflecting, brainstorming, verifying, refining, and revising. 
Besides, you can perform searching for uncertain knowledge if necessary with the format of <search> search query </search> during your thinking process.
Then, the search system will provide you with the retrieval information with the format of <document> search results </document>.
The answer needs to summarize the reasoning process and give the final answer.

Quesion: {question}
Response:
"""

_ANSWER_WITH_THINK_PROMPT_EN_RESTART = """You are a medical expert.  
Given a question, you should answer it by first thinking about the reasoning process in the mind and then providing the final answer. 
Please answer the question in the format of <think>...</think><answer>...</answer>. 
That is, <think>Here is the reasoning process</think><answer>Here is the answer</answer>. 
You should perform thinking with decomposing, reflecting, brainstorming, verifying, refining, and revising. 
Besides, you can perform searching for uncertain knowledge if necessary with the format of <search> search query </search> during your thinking process.
Then, the search system will provide you with the retrieval information with the format of <document> search results </document>.
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
你应该通过分解、反思、头脑风暴、验证、精炼和修订来进行思考。此外，如果需要查询不确定的知识，可以在思考过程中使用以下格式进行搜索：<search> 搜索关键词在这里 </search>。
然后，搜索系统会以以下格式提供检索信息：<document> 搜索结果 </document>。
答案需要对推理过程进行总结并给出最终答案。

问题: {question}
回答:
"""

# 后续的 response 需要与更新后的 prompt 组成一个闭环 <think></think><answer></answer>

_ANSWER_WITH_THINK_PROMPT_ZH_RESTART = """你是一名医学专家。
在回答问题时，你应该首先思考这个问题背后的推理过程，然后提供最终答案。
请按<think>...</think><answer>...</answer>标签格式回答问题。
即<think>这里是推理过程</think><answer>这里是答案</answer>。
你应该通过分解、反思、头脑风暴、验证、精炼和修订来进行思考。此外，如果需要查询不确定的知识，可以在思考过程中使用以下格式进行搜索：<search> 搜索关键词在这里 </search>。
然后，搜索系统会以以下格式提供检索信息：<document> 搜索结果 </document>。
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
你應該通過分解、反思、頭腦風暴、驗證、精煉和修訂來進行思考。此外，如果需要查詢不確定的知識，可以在思考過程中使用以下格式進行搜索：<search> 搜索關鍵詞在這裡 </search>。
然後，搜索系統會以以下格式提供檢索信息：<document> 搜索結果 </document>。
答案需要對推理過程進行總結並給出最終答案。

問題: {question}
回答:
"""

_ANSWER_WITH_THINK_PROMPT_ZH_HK_RESTART = """你是一名醫學專家。
在回答問題時，你應該首先思考這個問題背後的推理過程，然後提供最終答案。
請按<think>...</think><answer>...</answer>標籤格式回答問題。
即<think>這裡是推理過程</think><answer>這裡是答案</answer>。
你應該通過分解、反思、頭腦風暴、驗證、精煉和修訂來進行思考。此外，如果需要查詢不確定的知識，可以在思考過程中使用以下格式進行搜索：<search> 搜索關鍵詞在這裡 </search>。
然後，搜索系統會以以下格式提供檢索信息：<document> 搜索結果 </document>。
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
4. 反选类问题，即选出不符合条件的选项，或者“最不可能的是”，也不适合转换
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
在模型的思考过程中，对于不确定的部分，它会使用 <search> 搜索关键词 </search> 标签进行包裹，以便后续查询相关信息。当查询到相关内容后，模型将基于检索到的信息继续其推理过程。
需要注意的是，<document> 搜索结果 </document> 中的内容是外部检索的结果，并非模型自己生成的。因此，这些内容不应被用于衡量模型本身的逻辑推理能力，而仅作为评估其处理和整合外部信息的能力的依据。

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
在模型的思考过程中，对于不确定的部分，它会使用 <search> 搜索关键词 </search> 标签进行包裹，以便后续查询相关信息。当查询到相关内容后，模型将基于检索到的信息继续其推理过程。
需要注意的是，<document> 搜索结果 </document> 中的内容是外部检索的结果，并非模型自己生成的。因此，这些内容不应被用于衡量模型本身的逻辑推理能力，而仅作为评估其处理和整合外部信息的能力的依据。

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
During the model's reasoning process, uncertain parts are encapsulated using the <search> search query </search> tags to facilitate subsequent information retrieval. Once relevant content is retrieved, the model continues its reasoning based on the retrieved information.
It is important to note that the content within <document> search results </document> represents externally retrieved information and is not generated by the model itself. Therefore, this content should not be used to assess the model's logical reasoning ability. Instead, it should be used solely to evaluate the model's capability to process and integrate external information.

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
During the model's reasoning process, uncertain parts are encapsulated using the <search> search query </search> tags to facilitate subsequent information retrieval. Once relevant content is retrieved, the model continues its reasoning based on the retrieved information.
It is important to note that the content within <document> search results </document> represents externally retrieved information and is not generated by the model itself. Therefore, this content should not be used to assess the model's logical reasoning ability. Instead, it should be used solely to evaluate the model's capability to process and integrate external information.

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
请尽可能保留问题前面部分的内容不变，将最后的问句修改为开放性询问的方式，即将原本“下列哪一项是”，”最可能的选项是“等等方式进行修改。
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
請盡可能保留問題前面部分的內容不變，將最後的問句修改為開放性詢問的方式，即將原本“下列哪一項是”，”最可能的選項是“等等方式進行修改。
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
from openai import OpenAI

# ============ 配置区域 ============
client = OpenAI(api_key="")

# 示例输入文本（你可以换成任意回答）
answer_text = """
This dish offers a spicy and savory profile, with the crispiness of the chicken complemented by the bold heat of the red chilies, vividly shown in <image2>. The presentation is vibrant and inviting, with textural variety provided by the tender chicken pieces and crunchy chilies. Additionally, the distinct flavor of Sichuan peppercorns is unmistakable, contributing to its unique taste. An alternate perspective in <image3> highlights the intricate textures and highlights the alluring sheen of the flavorful oil.
"""

# Prompt 模板
prompt = f"""
You are an expert evaluator of open-ended text responses.
Your task is to assess the quality of a given answer TEXT across several dimensions.
Do not use external knowledge or check correctness against facts.
Only judge the quality of the text itself.

Evaluate the TEXT according to the following dimensions.
For each dimension, output a score between 1 and 5 (integers only, 1=very poor, 5=excellent).
Provide no explanation, just the scores.

### Dimensions:
1. Clarity: Is the text easy to read and understand, free of ambiguity or confusing phrasing?
2. Coherence & Logic: Does the answer flow logically, without contradictions or abrupt jumps?
3. Informativeness & Specificity: Does the text provide meaningful, concrete content instead of being vague or generic?
4. Conciseness: Is the text free of redundancy and unnecessary filler?
5. Style & Tone: Is the language polished, professional, and appropriate?

After scoring each dimension, compute the Overall Score as the average of the five scores, rounded to one decimal.

### Output format (JSON only):
{{
  "clarity": <int>,
  "coherence": <int>,
  "informativeness": <int>,
  "conciseness": <int>,
  "style": <int>,
  "overall": <float>
}}

### TEXT:
{answer_text}
"""

# 调用 4o
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}],
    temperature=0
)

print(response.choices[0].message.content)

import json
from openai import OpenAI

# ============ Configuration ============
API_KEY = ""
MODEL_NAME = "gpt-4o"
OUTPUT_FILE = "text_eval_result.json"
# =======================================

client = OpenAI(api_key=API_KEY)

# ---- Manually constructed texts (edit as you like) ----
predicted_text = (
    "The study concludes that regular aerobic exercise improves cardiovascular health, "
    "reduces stress, and can slightly lower blood pressure in adults."
)

ground_truth_text = (
    "Regular aerobic exercise enhances heart health, lowers stress levels, "
    "and modestly decreases adult blood pressure."
)

# ---- Evaluation Prompt (English, ABCD with weights 6/2/1/1) ----
prompt = """
You are an impartial evaluator for text generation tasks.  
You will be given two texts:

1. Predicted Text (produced by a model).
2. Ground Truth Text (the correct reference).

Your task is to evaluate the semantic correctness of the predicted text compared to the ground truth across four dimensions.
For each dimension, assign a score between 0 and 1, where 1 means perfect and 0 means completely incorrect.

Dimensions:
A. Semantic Similarity (Weight 6)
- Compare the overall meaning of the two texts.
- Judge whether they convey the same main facts, intent, and semantics.
- Output a similarity score between 0 and 1.

B. Content Coverage / Key Facts (Weight 2)
- Identify the main facts or key points in the ground truth.
- Check if the predicted text covers those key points.
- Compute an approximate F1 score (harmonic mean of precision and recall).
- Consider synonyms and paraphrased content as matches.

C. Attribute / Detail Accuracy (Weight 1)
- Evaluate whether specific details (numbers, named entities, colors, dates, places, etc.) are correct.
- Output an accuracy score between 0 and 1.

D. Faithfulness / Non-Hallucination (Weight 1)
- Check whether the predicted text introduces facts that are not supported by the ground truth.
- If no hallucinations, give a high score; penalize if extra unsupported facts are present.

Final Score
Compute the weighted average:
Final Score = (6*A + 2*B + 1*C + 1*D) / 10

Output format
Respond strictly in JSON with the following fields:
{
  "semantic_similarity": float,
  "content_coverage_f1": float,
  "detail_accuracy": float,
  "faithfulness": float,
  "final_score": float
}

Do not include explanations, reasoning, or any extra text outside the JSON.
"""

# ---- Call 4o to evaluate ----
messages = [
    {"role": "system", "content": prompt},
    {"role": "user", "content": f"Predicted Text:\n{predicted_text}\n\nGround Truth Text:\n{ground_truth_text}"}
]

resp = client.chat.completions.create(
    model=MODEL_NAME,
    messages=messages,
    temperature=0.0
)

raw = resp.choices[0].message.content.strip()

# ---- Parse & save ----
try:
    result = json.loads(raw)
except json.JSONDecodeError:
    print("⚠️ Could not parse model output as JSON. Raw output:")
    print(raw)
else:
    print(json.dumps(result, indent=2))
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"✅ Saved to {OUTPUT_FILE}")

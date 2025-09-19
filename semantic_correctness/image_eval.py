import json
from openai import OpenAI

# ============ Configuration ============
API_KEY = ""  # <-- replace with your real API key
MODEL_NAME = "gpt-4o" 
# =======================================

client = OpenAI(api_key=API_KEY)

# Manually constructed Dense Captions
predicted_caption = {
    "global_summary": "A golden retriever is standing on a grassy field outdoors.",
    "key_elements": ["golden retriever", "outdoor scene", "grassy field", "standing posture"]
}

ground_truth_caption = {
    "global_summary": "A dog is standing on grass outside.",
    "key_elements": ["dog", "grass", "outdoor", "standing"]
}

# Evaluation Prompt
prompt = """
You are an impartial evaluator for image captioning tasks. 
You will be given two Dense Caption JSON objects:

1. Predicted Caption (produced by a model).
2. Ground Truth Caption (the correct reference).

Each JSON has the structure:
{
  "global_summary": "...",
  "key_elements": ["...", "...", ...]
}

Your task is to evaluate the **semantic correctness** of the predicted caption compared to the ground truth across three dimensions. 
For each dimension, assign a score between 0 and 1, where 1 means perfect match and 0 means completely incorrect.

### Dimensions:
A. Semantic Similarity (Weight 0.7)
- Compare the `global_summary` of both captions.
- Judge whether they describe the same overall scene, subjects, and main actions.
- Output a similarity score between 0 and 1.

B. Key Element Coverage (Weight 0.2)
- Compare the `key_elements` lists.
- Compute an approximate F1 score (harmonic mean of precision and recall) for overlap in meaning, not just exact string match.
- Consider synonyms and near-synonyms as matches (e.g., "dog" ≈ "golden retriever").

C. Attribute & Detail Accuracy (Weight 0.1)
- Evaluate whether descriptive attributes (colors, numbers, modifiers) in the prediction are correct compared to the ground truth.
- Output an accuracy score between 0 and 1.

### Final Score
Compute the weighted sum:
Final Score = 0.7 * A + 0.2 * B + 0.1 * C

### Output format
Respond strictly in JSON with the following fields:
{
  "semantic_similarity": float,
  "key_element_f1": float,
  "attribute_accuracy": float,
  "final_score": float
}

Do not include any explanation, only output the JSON.
"""

# Call 4o for evaluation
response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Predicted Caption:\n{json.dumps(predicted_caption)}\n\nGround Truth Caption:\n{json.dumps(ground_truth_caption)}"}
    ],
    temperature=0.0
)

# Parse result
output_text = response.choices[0].message.content.strip()

try:
    eval_result = json.loads(output_text)
except json.JSONDecodeError:
    print("⚠️ Could not parse model output as JSON. Raw output:")
    print(output_text)
else:
    print(json.dumps(eval_result, indent=2))

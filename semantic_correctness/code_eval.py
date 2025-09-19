import json
from openai import OpenAI

# ============ Configuration ============
API_KEY = ""
MODEL_NAME = "gpt-4o"
OUTPUT_FILE = "code_eval_result.json"
# =======================================

client = OpenAI(api_key=API_KEY)

# ---- Manually constructed Markdown code snippets ----
# Example task: Two Sum (showing different design/performance between solutions)

predicted_code_md = """```python
# Predicted: brute-force O(n^2)
from typing import List, Tuple, Optional

def two_sum(nums: List[int], target: int) -> Optional[Tuple[int, int]]:
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if nums[i] + nums[j] == target:
                return (i, j)
    return None

if __name__ == "__main__":
    print(two_sum([2,7,11,15], 9))
```"""

ground_truth_code_md = """```python
# Ground Truth: hash map O(n)
from typing import List, Tuple, Optional

def two_sum(nums: List[int], target: int) -> Optional[Tuple[int, int]]:
    index = {}
    for i, x in enumerate(nums):
        need = target - x
        if need in index:
            return (index[need], i)
        index[x] = i
    return None

if __name__ == "__main__":
    print(two_sum([2,7,11,15], 9))
```"""

# ---- Evaluation Prompt (English, ACDEF with weights 5/1/1/1/1) ----
prompt = """
You are an impartial evaluator for code generation tasks.
You will be given two code snippets in Markdown format:

1. Predicted Code (produced by a model).
2. Ground Truth Code (the correct reference).

Your task is to evaluate the quality and semantic correctness of the predicted code compared to the ground truth across five dimensions.
For each dimension, assign a score between 0 and 1, where 1 means perfect and 0 means completely incorrect.

Dimensions:
A. Correctness / Functional Equivalence (Weight 5)
- Does the predicted code correctly implement the same functionality as the ground truth?
- Consider logic, syntax, and whether the output behavior would be equivalent.
- This is the most important dimension.

C. Design & Structure (Weight 1)
- Evaluate whether the code structure and chosen approach are reasonable.
- Does it use appropriate algorithms, data structures, and coding patterns?

D. Performance & Efficiency (Weight 1)
- Assess the predicted code’s computational efficiency.
- Does it avoid unnecessary loops, expensive operations, or inefficient data handling?

E. Security & Robustness (Weight 1)
- Check whether the code safely handles edge cases and avoids obvious vulnerabilities.
- Examples: input validation, error handling, avoiding unsafe practices.

F. Testability & Extensibility (Weight 1)
- Is the code modular, readable, and easy to test or extend?
- Does it follow best practices for maintainability?

Final Score
Compute the weighted average:
Final Score = (5*A + 1*C + 1*D + 1*E + 1*F) / 9

Output format
Respond strictly in JSON with the following fields:
{
  "correctness": float,
  "design": float,
  "performance": float,
  "security": float,
  "testability": float,
  "final_score": float
}

Do not include explanations, reasoning, or any extra text outside the JSON.
"""

# ---- Call 4o to evaluate ----
messages = [
    {"role": "system", "content": prompt},
    {
        "role": "user",
        "content": (
            "Predicted Code (Markdown):\n"
            f"{predicted_code_md}\n\n"
            "Ground Truth Code (Markdown):\n"
            f"{ground_truth_code_md}"
        )
    }
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

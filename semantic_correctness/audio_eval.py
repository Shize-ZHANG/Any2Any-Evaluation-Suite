import json
import re
from openai import OpenAI

# ============ Configuration ============
API_KEY = ""  # <-- replace with your real API key
MODEL_NAME = "gpt-4o"
OUTPUT_FILE = "audio_eval_result.json"
# =======================================

client = OpenAI(api_key=API_KEY)

# ---- Manually provide two global summaries (edit as needed) ----
predicted_summary = (
    "A calm news announcer delivers a short weather update indoors, with faint background music."
)
ground_truth_summary = (
    "An indoor news bulletin provides a concise weather report in a steady, calm tone; soft background music is present."
)

# ---- Evaluation Prompt (Audio Global Summary, ABD weights 7/2/1) ----
prompt = """
You are an impartial evaluator for audio dense captioning.

You will be given TWO global summaries for the SAME audio:
1) Predicted Global Summary (produced by a model)
2) Ground Truth Global Summary (the reference)

Your job is to score the predicted summary against the ground truth along THREE dimensions.
For each dimension, output a score between 0 and 1 (1 = perfect, 0 = completely incorrect).

Important rules:
- Judge ONLY based on the two summaries. Do NOT invent details.
- Treat paraphrases and near-synonyms as equivalent if they convey the same meaning.
- N/A handling for B and D:
  • If the ground truth does NOT mention an aspect and the prediction also does NOT mention it, set that dimension to 1.0 (no penalty).
  • If the ground truth DOES mention an aspect but the prediction omits it or contradicts it, lower the score accordingly.
  • If the prediction adds extra detail not present in the ground truth and it does NOT contradict the ground truth, do NOT penalize.
- Direct contradictions should reduce the relevant dimension and may also reduce A.

### Dimensions
A. Content Semantic Similarity (Weight 7)
- Do both summaries convey the same core meaning about WHO/WHAT and the overall topic/intent of the audio?
- Ignore wording differences; focus on semantic equivalence of the main content.

B. Audio Scene & Source Alignment (Weight 2)
- Do the summaries align on the audio scene and sources explicitly mentioned in the ground truth?
  Examples: speech vs. music vs. environmental sounds (rain, traffic, applause), presence/absence of background music/noise, indoor vs. outdoor context.
- Score for consistency with ground-truth mentions (recall-focused per the rules above).

D. Paralinguistics & Prosody (Weight 1)
- Do the summaries align on paralinguistic cues mentioned in the ground truth?
  Examples: speaking rate (fast/slow), intensity (soft/loud), affect (calm/excited), laughter/sighs, notable pauses.
- Score for consistency with ground-truth mentions (recall-focused per the rules above).

### Final Score
Compute the weighted average:
Final Score = (7*A + 2*B + 1*D) / 10

### Output
Respond STRICTLY in JSON with these fields only:
{
  "content_semantic_similarity": float,
  "audio_scene_source_alignment": float,
  "paralinguistics_prosody": float,
  "final_score": float
}

Do NOT include explanations, intermediate notes, or any extra text outside the JSON.
""".strip()

# ---- Call 4o to evaluate ----
messages = [
    {"role": "system", "content": prompt},
    {
        "role": "user",
        "content": (
            "Predicted Global Summary:\n"
            f"{predicted_summary}\n\n"
            "Ground Truth Global Summary:\n"
            f"{ground_truth_summary}"
        ),
    },
]

resp = client.chat.completions.create(
    model=MODEL_NAME,
    messages=messages,
    temperature=0.0,
)

raw = resp.choices[0].message.content.strip()

# ---- Parse JSON (with a small fallback if the model added wrappers) ----
def try_parse_json(s: str):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise

try:
    result = try_parse_json(raw)
except Exception:
    print("⚠️ Could not parse model output as JSON. Raw output:")
    print(raw)
else:
    print(json.dumps(result, indent=2))
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"✅ Saved to {OUTPUT_FILE}")

import json
import re
from openai import OpenAI

# ============ Configuration ============
API_KEY = ""  # <-- replace with your real API key
MODEL_NAME = "gpt-4o"
OUTPUT_FILE = "video_eval_result.json"
# =======================================

client = OpenAI(api_key=API_KEY)

# ---- Manually provide two dense captions (edit as needed) ----
predicted_caption = (
    "A woman jogs along a riverside path at sunrise. She stops to tie her shoelaces, "
    "waves at a passing cyclist, then resumes running across a small wooden bridge."
)

ground_truth_caption = (
    "At dawn, a woman runs beside a river. She pauses to retie her shoe, greets a cyclist, "
    "and continues her run, crossing a wooden footbridge."
)

# ---- Evaluation Prompt (English, ABCD weights 6/2/1/1) ----
prompt = """
You are an impartial evaluator for video dense captioning.
You will be given TWO dense captions about the SAME video:
1) Predicted Dense Caption (produced by a model)
2) Ground Truth Dense Caption (the reference)

Optionally, you may also receive URLs to the underlying videos. Use them only to resolve ambiguities; do NOT invent events that are not described in the captions.

Your job is to score the predicted caption against the ground truth along FOUR dimensions.
For each dimension, output a score between 0 and 1 (1 = perfect, 0 = completely incorrect).

Before scoring, extract from EACH caption:
- A brief global summary (1–2 sentences you infer from the caption).
- An ordered event timeline: a list of events as objects
  {actors[], action, objects[], scene/location?, time marker?}, in the order implied by the caption.

### Dimensions
A. Global Temporal Semantics (Weight 6)
- Do both captions convey the same overall storyline/theme, main characters and scenes, and the core goals/actions across time?
- Score 0–1 for semantic equivalence of the overall narrative (ignore wording differences).

B. Event Coverage & Ordering (Weight 2)
- First, match events across the two timelines using semantic similarity and role alignment (actors/action/objects).
- Let coverage_F1 be the F1 of matched events (precision/recall over ground-truth events).
- Let order_agreement be the relative ordering agreement over the matched pairs (e.g., normalized Kendall-τ or proportion of correctly ordered pairs).
- Compute B as: B = 0.7 * coverage_F1 + 0.3 * order_agreement.

C. Action–Actor–Object Accuracy (Weight 1)
- For the matched event pairs, assess how well the predicted caption preserves WHO–did–WHAT–to–WHOM (actor(s), verb, object(s)/target).
- Allow near-synonyms and class hierarchy matches (e.g., “dog” ≈ “golden retriever”) with partial credit.
- Output the average correctness 0–1 across matched events.

D. Scene & Identity Persistence (Weight 1)
- Across time, do the captions maintain consistent identities (the same person/thing not confused later) and comparable scene transitions?
- Are the number and order of scene changes roughly aligned?
- Score 0–1.

### Final Score
Compute the weighted average:
Final Score = (6*A + 2*B + 1*C + 1*D) / 10

### Output
Respond STRICTLY in JSON with these fields only:
{
  "global_temporal_semantics": float,
  "event_coverage_ordering": float,
  "action_actor_object": float,
  "scene_identity_persistence": float,
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
            "Predicted Dense Caption:\n"
            f"{predicted_caption}\n\n"
            "Ground Truth Dense Caption:\n"
            f"{ground_truth_caption}"
        ),
    },
]

resp = client.chat.completions.create(
    model=MODEL_NAME,
    messages=messages,
    temperature=0.0,
)

raw = resp.choices[0].message.content.strip()

# ---- Parse JSON (with a small fallback if model added wrappers) ----
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

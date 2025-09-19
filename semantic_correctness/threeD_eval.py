import json
import re
from openai import OpenAI

# ============ Configuration ============
API_KEY = ""  # <-- replace with your real API key
MODEL_NAME = "gpt-4o"
OUTPUT_FILE = "3d_eval_result.json"
# =======================================

client = OpenAI(api_key=API_KEY)

# ---- Manually provide two 3D captions (edit as needed) ----
predicted_caption = (
    "A modern wooden dining chair with a slightly curved backrest and four straight legs; "
    "the seat is flat and thin, overall proportions are slim and upright, finished with a matte wood texture."
)

ground_truth_caption = (
    "A wooden chair intended for dining, featuring a curved backrest, four legs, and a slim upright silhouette. "
    "The seat is flat; the surface looks matte wood rather than glossy."
)

# ---- Evaluation Prompt (3D captions, ABCD with weights 6/2/1/1) ----
prompt = """
You are an impartial evaluator for 3D content descriptions.

You will be given TWO 3D captions about the SAME 3D object or scene:
1) Predicted 3D Caption (produced by a model)
2) Ground Truth 3D Caption (the reference)

Your job is to score the predicted caption against the ground truth along FOUR dimensions.
For each dimension, output a score between 0 and 1 (1 = perfect, 0 = completely incorrect).

General rules:
- Judge ONLY based on the two captions. Do NOT invent details.
- Treat paraphrases, near-synonyms, and reasonable hypernym/hyponym matches as equivalent when semantics align
  (e.g., “sofa ≈ couch”, “dog ≈ golden retriever”, “metal ≈ brushed steel” with partial credit if needed).
- Penalize direct contradictions (e.g., “three legs” vs “four legs”, “wood” vs “glass”).
- If the ground truth does NOT mention a given aspect and the prediction also does NOT mention it, do NOT penalize for that aspect (N/A).

### Dimensions
A. Global 3D Semantic Similarity (Weight 6)
- Do both captions describe the same object/scene category and intended function/theme?
- Consider high-level form and style (e.g., realistic vs low-poly) when explicitly stated.
- Score 0–1 for overall semantic equivalence; ignore wording differences.

B. Part Structure & Topology Coverage (Weight 2)
- Does the predicted caption cover the ground-truth key parts (and their hierarchical/attachment/symmetry relations)?
  Examples: chair back/seat/legs/armrests; car body/wheels/lights; “four legs”, “left–right symmetry”.
- Compute a coverage-style judgment (recall-focused) with partial credit for near-synonymous or closely related parts.
- Score 0–1.

C. Geometric Quantities, Pose & Proportions (Weight 1)
- Do the captions align on shape proportions (tall/short, thin/thick, rounded/angular), pose/orientation (upright/lying, facing), and simple counts when given?
- Allow tolerance for coarse descriptors and typical numeric fuzziness (e.g., ±20% on proportions, ±1 on small counts).
- Score 0–1.

D. Materials, Colors & Surface Appearance (Weight 1)
- Do they agree on material families (wood/metal/plastic/glass/fabric), color themes, and simple texture/finish cues (matte/gloss/rough/transparent) when mentioned?
- If the ground truth omits appearance and the prediction also omits it, treat as N/A (do not penalize).
- Score 0–1 when applicable.

### Final Score
Compute the weighted average:
Final Score = (6*A + 2*B + 1*C + 1*D) / 10

### Output
Respond STRICTLY in JSON with these fields only:
{
  "global_3d_semantic_similarity": float,
  "part_structure_topology_coverage": float,
  "geometric_pose_proportions": float,
  "materials_colors_appearance": float,
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
            "Predicted 3D Caption:\n"
            f"{predicted_caption}\n\n"
            "Ground Truth 3D Caption:\n"
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

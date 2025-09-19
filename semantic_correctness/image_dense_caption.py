import json
from openai import OpenAI

# ============ Configuration ============
API_KEY = ""
IMAGE_URL = ""
MODEL_NAME = "gpt-4o"
OUTPUT_FILE = "dense_caption.json"
# =======================================

client = OpenAI(api_key=API_KEY)

# Perfected Prompt (English, detailed & strict)
prompt = """
You are a precise and reliable image description assistant.

Your task is to generate a **Dense Caption JSON** for the input image that captures both the overall semantics and the essential details. Follow these rules:

1. The output must be a **strictly valid JSON object**. Do not include explanations, markdown, or extra text outside the JSON.
2. The JSON must contain two top-level fields:
   - global_summary: A single sentence or a short paragraph that summarizes the main semantics of the image (scene, primary subjects, main action, or intent).
   - key_elements: An array listing 3–6 essential elements of the image (subjects, scene, main actions, or salient attributes).
3. The description must **cover the main information** without being overly abstract. Ensure that it includes:
   - Main subjects (e.g., a boy, a dog, a car).
   - Scene or environment (e.g., in a classroom, in a park, in a kitchen).
   - Main actions or states (e.g., eating, reading, running).
   - (Optional) Notable attributes (color, quantity, visible text, distinctive objects).
4. **Do not hallucinate or invent invisible details.** Do not infer identity, brand, emotions, or locations unless there is direct visual evidence.
5. If something is uncertain, use cautious natural language (e.g., “possibly a billboard”) but never fabricate details.

Output format example:
{
  "global_summary": "A young boy is eating breakfast in a kitchen.",
  "key_elements": [
    "young boy",
    "kitchen setting",
    "eating breakfast",
    "table with milk and toast"
  ]
}
"""

# Call 4o for dense captioning
response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": [
            {"type": "text", "text": "Generate the Dense Caption JSON for this image:"},
            {"type": "image_url", "image_url": {"url": IMAGE_URL}}
        ]}
    ],
    temperature=0.3
)

# Parse result
output_text = response.choices[0].message.content.strip()

try:
    caption_json = json.loads(output_text)
except json.JSONDecodeError:
    print("⚠️ Could not parse model output as JSON. Raw output:")
    print(output_text)
else:
    print(json.dumps(caption_json, ensure_ascii=False, indent=2))
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(caption_json, f, ensure_ascii=False, indent=2)
    print(f"✅ Dense caption saved to {OUTPUT_FILE}")

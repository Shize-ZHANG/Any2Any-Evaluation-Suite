import io
import re
import json
import requests
import pytesseract
from PIL import Image
from openai import OpenAI

# ============ Configuration ============
API_KEY = ""  # <-- put your real API key here
MODEL_NAME = "gpt-4o"
OUTPUT_FILE = "doc_eval_result.json"

# Example URLs (replace with yours)
PRED_URL = "https://raw.githubusercontent.com/Shize-ZHANG/Any2Any-Interleaved-Data-Pipline/main/original_data/document/doc_0791_01.png"
GT_URL   = "https://raw.githubusercontent.com/Shize-ZHANG/Any2Any-Interleaved-Data-Pipline/main/original_data/document/doc_0791_01.png"

# If needed, specify Tesseract path:
# pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"
# ======================================

client = OpenAI(api_key=API_KEY)

def download_image(url: str) -> Image.Image:
    """Download image and return PIL.Image."""
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content))
    # Some PNGs are palette/LA; convert to RGB for more stable OCR
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return img

def image_to_text(img: Image.Image) -> str:
    """OCR to text using pytesseract."""
    return pytesseract.image_to_string(img)

def text_to_markdown(text: str) -> str:
    """
    Convert OCR text to Markdown:
      - If lines look like columns separated by 2+ spaces/tabs, try table.
      - Otherwise return as fenced code block for readability/stability.
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    non_empty = [ln for ln in lines if ln.strip()]

    # Heuristic: if >= 3 lines have >= 2 columns -> treat as table
    table_like = 0
    split_rows = []
    for ln in non_empty:
        parts = re.split(r"\s{2,}|\t", ln.strip())
        parts = [p for p in parts if p != ""]
        split_rows.append(parts)
        if len(parts) >= 2:
            table_like += 1

    if table_like >= 3 and len(split_rows[0]) >= 2:
        # Build Markdown table
        header = split_rows[0]
        md = []
        md.append("| " + " | ".join(header) + " |")
        md.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in split_rows[1:]:
            # pad/truncate to header length
            row = (row + [""] * len(header))[:len(header)]
            md.append("| " + " | ".join(row) + " |")
        return "\n".join(md)
    else:
        # Fallback: fenced code block to preserve layout
        safe = "\n".join(non_empty)
        return f"```text\n{safe}\n```"

def build_document_eval_prompt() -> str:
    """Evaluation prompt (ABCD, weights 6/2/1/1)."""
    return """
You are an impartial evaluator for document generation tasks.
You will be given two documents, each provided in two forms:
1) An OCR-extracted Markdown text version (already preprocessed for readability).
2) The original PNG image URL of the document.

- Predicted Document: produced by a model (OCR Markdown + PNG).
- Ground Truth Document: the correct reference (OCR Markdown + PNG).

Your task is to evaluate the predicted document compared to the ground truth across FOUR dimensions.
For each dimension, assign a score between 0 and 1, where 1 means perfect and 0 means completely incorrect.

Dimensions:
A. Textual Accuracy / Semantic Similarity (Weight 6)
- Compare the OCR texts. Judge whether the predicted text accurately matches the ground truth in meaning and wording.
- Minor OCR variations (like "0" vs "O") may be tolerated but should slightly reduce the score.

B. Content Coverage / Completeness (Weight 2)
- Check whether the predicted document includes all the key content of the ground truth.
- Penalize missing paragraphs, tables, or important items (focus on recall of ground truth content).

C. Layout & Structural Consistency (Weight 1)
- Compare structures visible in the PNGs (titles, sections, tables, lists, paragraph order).
- Evaluate whether the predicted layout preserves the structural organization of the ground truth.

D. Formatting & Style Accuracy (Weight 1)
- Compare formatting and style visible in the PNGs (bold/italics, heading levels, alignment, font size cues).
- Reward faithful preservation of stylistic cues; penalize obvious loss or distortion.

Final Score
Compute the weighted average:
Final Score = (6*A + 2*B + 1*C + 1*D) / 10

Output format
Respond strictly in JSON with the following fields:
{
  "textual_accuracy": float,
  "content_coverage": float,
  "layout_consistency": float,
  "formatting_accuracy": float,
  "final_score": float
}

Do not include explanations, reasoning, or any extra text outside the JSON.
""".strip()

def call_4o_for_document_eval(pred_md: str, pred_url: str, gt_md: str, gt_url: str) -> str:
    """
    Send both Markdown OCRs and both PNG URLs to 4o with the evaluation prompt.
    Return raw model output (expected JSON string).
    """
    prompt = build_document_eval_prompt()
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Evaluate the predicted document against the ground truth "
                        "following the required dimensions and output schema.\n\n"
                        "Predicted Document (OCR Markdown):\n"
                        f"{pred_md}\n\n"
                        "Ground Truth Document (OCR Markdown):\n"
                        f"{gt_md}\n\n"
                        "Here are the corresponding PNG images for reference:"
                    ),
                },
                {"type": "image_url", "image_url": {"url": pred_url}},
                {"type": "image_url", "image_url": {"url": gt_url}},
            ],
        },
    ]
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()

def main(pred_url: str, gt_url: str):
    # 1) Download images
    pred_img = download_image(pred_url)
    gt_img = download_image(gt_url)

    # 2) OCR
    pred_text = image_to_text(pred_img)
    gt_text = image_to_text(gt_img)

    print("🔍 Predicted OCR (raw):\n", pred_text)
    print("\n🔍 Ground Truth OCR (raw):\n", gt_text)

    # 3) Convert to Markdown
    pred_md = text_to_markdown(pred_text)
    gt_md = text_to_markdown(gt_text)

    print("\n📑 Predicted OCR → Markdown:\n", pred_md)
    print("\n📑 Ground Truth OCR → Markdown:\n", gt_md)

    # 4) Call 4o with both Markdown texts + both PNGs
    raw = call_4o_for_document_eval(pred_md, pred_url, gt_md, gt_url)

    # 5) Parse & save
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        print("\n⚠️ Model did not return valid JSON. Raw output:\n", raw)
        return

    print("\n✅ Document evaluation result (JSON):")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main(PRED_URL, GT_URL)

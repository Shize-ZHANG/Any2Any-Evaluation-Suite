#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
caption_eval_pipeline.py
-----------------------------------
Two traversal modes:
 - response-mode (default): iterate response → caption → match GT → score
 - gt-mode: iterate ground truth → match response → if matched, caption response → score

Output:
 - response-mode → <response>.caption_scored.jsonl
 - gt-mode       → <response>.caption_scored.traverse_gt.jsonl
"""

import os
import re
import json
import base64
import tempfile
import argparse
from tqdm import tqdm
from typing import Dict, Any
from openai import OpenAI

# Optional heavy deps
import torch
# import open3d as o3d
from transformers import AutoTokenizer
# from pointllm.model import PointLLMLlamaForCausalLM
# from pointllm.utils import disable_torch_init
# from pointllm.model.utils import KeywordsStoppingCriteria
# from pointllm.conversation import conv_templates, SeparatorStyle


# ========== Utility ==========
def read_jsonl(path: str):
    """Support multi-line JSON array or concatenated JSONs"""
    text = open(path, "r", encoding="utf-8").read().strip()
    if text.startswith("["):
        return json.loads(text)
    items, buf = [], ""
    for line in text.splitlines():
        buf += line
        try:
            items.append(json.loads(buf))
            buf = ""
        except json.JSONDecodeError:
            continue
    return items


def write_jsonl(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)



def replace_placeholders(content: str, captions: Dict[str, str]) -> str:
    """Replace <image1> ... with <image1: caption> (format aligned with original script)"""
    pattern = re.compile(r"<([A-Za-z]+)(\d+)>")
    def repl(match):
        prefix, num = match.groups()
        tag = f"{prefix}{num}"
        caption = captions.get(tag, "[MISSING_CAPTION]")
        return f"<{tag}: {caption}>"
    return pattern.sub(repl, content)


def get_modality(tag: str):
    for prefix in ["image", "video", "audio", "document", "code", "threeD"]:
        if tag.startswith(prefix):
            return prefix
    return None


# ========== Captioners ==========
def caption_image(path, client):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    msg = [
        {"role": "user", "content": [
            {"type": "text", "text": "Describe this image in detail."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        ]}
    ]
    resp = client.chat.completions.create(
        model="gpt-5-mini", messages=msg, max_tokens=1024, temperature=0)
    return resp.choices[0].message.content.strip()


def caption_video_or_audio(path, client, model_name):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
        mtype = "video"
        mime  = "video/mp4"
        key   = "video_url"
        prompt= "Describe this video content in detail."
    else:
        mtype = "audio"
        mime  = "audio/mpeg" if ext == ".mp3" else "audio/wav"
        key   = "audio_url"
        prompt= "Describe this audio content in detail."

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    msg = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": key,  key: {"url": f"data:{mime};base64,{b64}"}}
    ]}]

    resp = client.chat.completions.create(
        model=model_name, messages=msg, max_tokens=1024, temperature=0)
    return resp.choices[0].message.content.strip()


def caption_document(path, client):
    # Minimal generic doc caption (PDF/CSV/TXT...): base64 as octet-stream + text prompt.
    prompt = "Describe the following document content in detail."
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    msg = [
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:application/octet-stream;base64,{b64}"}}
        ]}
    ]
    resp = client.chat.completions.create(
        model="gpt-5-mini", messages=msg, max_tokens=1024, temperature=0)
    return resp.choices[0].message.content.strip()


def caption_code(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()[:6000]


# ---------- PointLLM 3D ----------
def load_point_cloud(pc_path: str, pointnum: int = 8192):
    pcd = o3d.io.read_point_cloud(pc_path)
    if len(pcd.points) == 0:
        raise ValueError(f"Empty point cloud: {pc_path}")
    pts = torch.from_numpy((torch.tensor(pcd.points)).numpy()).float()
    centroid = pts.mean(dim=0, keepdim=True)
    pts = pts - centroid
    scale = (pts.pow(2).sum(dim=1).sqrt().max() + 1e-6)
    pts = pts / scale
    if pts.shape[0] >= pointnum:
        idx = torch.randperm(pts.shape[0])[:pointnum]
        pts = pts[idx]
    else:
        pad = pts[torch.randint(0, pts.shape[0], (pointnum - pts.shape[0],))]
        pts = torch.cat([pts, pad], dim=0)
    return pts.unsqueeze(0)


@torch.inference_mode()
def caption_threed(pc_path, model_path, pointnum=8192):
    # NOTE: PointLLM imports are commented out for brevity; enable in your env.
    # from pointllm.utils import disable_torch_init
    # from pointllm.model import PointLLMLlamaForCausalLM
    # from pointllm.model.utils import KeywordsStoppingCriteria
    # from pointllm.conversation import conv_templates, SeparatorStyle

    if pc_path.endswith(".off"):
        mesh = o3d.io.read_triangle_mesh(pc_path)
        pcd = mesh.sample_points_uniformly(pointnum)
        tmp = tempfile.NamedTemporaryFile(suffix=".ply", delete=False)
        o3d.io.write_point_cloud(tmp.name, pcd)
        pc_path = tmp.name

    disable_torch_init()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = PointLLMLlamaForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16).cuda()
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
    model.eval()

    point_clouds = load_point_cloud(pc_path, pointnum=pointnum).cuda().to(model.dtype)

    conv_mode = "vicuna_v1_1"
    conv = conv_templates[conv_mode].copy()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    qs = "<point>\nCaption this 3D model in detail."
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    inputs = tokenizer([prompt])
    input_ids = torch.as_tensor(inputs.input_ids).cuda()
    stopping = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    out_ids = model.generate(
        input_ids.repeat(1, 1),
        point_clouds=point_clouds,
        do_sample=True,
        temperature=0.2,
        top_p=0.95,
        max_length=2048,
        stopping_criteria=[stopping],
    )
    input_token_len = input_ids.shape[1]
    txt = tokenizer.batch_decode(out_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()
    return txt


# ========== gpt-5-mini scoring ==========
def _bin_score(x: float) -> float:
    """Map any float in [0,1] to the nearest bin in {0.2,0.4,0.6,0.8,1.0}."""
    bins = [0.2, 0.4, 0.6, 0.8, 1.0]
    if x is None:
        return None
    x = max(0.0, min(1.0, float(x)))
    return min(bins, key=lambda b: abs(b - x))


def compute_score(pred: str, ref: str, client) -> float | None:
    system = (
        "You are a strict automatic grader. Evaluate only semantic correctness. "
        "Ignore coherence, style, tone, length, politeness, formatting. "
        "Treat paraphrases, wording changes, ordering changes, and unit conversions as equivalent "
        "if they preserve meaning. Numbers may be mildly rounded, but the value/unit/range, "
        "comparative relations, causal/temporal conditions must remain semantically equivalent. "
        "If there are core factual errors, contradictions, hallucinations, or key omissions, "
        "lower the score. Return ONLY a JSON object with a single key 'score', whose value "
        "MUST be one of: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0."
    )

    user = (
        "Score the semantic correctness of the Candidate (Pred) relative to the Reference (Ref).\n"
        "Ignore style/fluency/tone/format. Focus ONLY on factual/semantic agreement and key-point coverage.\n\n"
        "Scoring rubric (choose exactly ONE value):\n"
        "- 1.0: Completely equivalent. All key facts correct/covered. No contradictions. Units/ranges/relations match (paraphrase/ordering/rounding OK).\n"
        "- 0.8: Almost equivalent. ≥90% key facts correct/covered. No major contradiction. Only minor omissions/ambiguity that do not affect the main conclusion.\n"
        "- 0.6: Partially correct. Roughly half key facts correct. Noticeable omissions or minor contradictions/misinterpretations, but the main conclusion is not fully overturned.\n"
        "- 0.4: Low correctness. <50% key facts match. Important errors/contradictions/confusions (numbers/entities), or the core conclusion drifts, but still loosely on topic.\n"
        "- 0.2: Nearly incorrect/irrelevant. Mostly wrong/missing/contradictory, hallucinated, or non-answers.\n\n"
        "- 0.0: Completely incorrect or empty. Candidate is blank, gibberish, or entirely unrelated.\n\n"
        "Return STRICT JSON only: {\"score\": 0.0|0.2|0.4|0.6|0.8|1.0}\n\n"
        f"<<<BEGIN_PRED>>>\n{pred}\n<<<END_PRED>>>\n\n"
        f"<<<BEGIN_REF>>>\n{ref}\n<<<END_REF>>>"
    )

    resp = client.chat.completions.create(
        model="gpt-5-mini",
        response_format={"type": "json_object"},
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
        score = data.get("score", None)
        if isinstance(score, (int, float)):
            return _bin_score(score)
        if isinstance(score, str):
            try:
                return _bin_score(float(score))
            except Exception:
                return None
        return None
    except Exception:
        import re
        m = re.search(r"(?<!\d)(?:1(?:\.0+)?|0?\.\d+)(?!\d)", content)
        return _bin_score(float(m.group(0))) if m else None


# ========== Pair processing (shared) ==========
def caption_response_item(response_item, openai_client, vllm_client, pointllm_path, assets_root, vllm_model_name):
    """Generate captions for response_item['output'] and return captioned text."""
    output = response_item["output"]
    modal_map = output.get("modal", {})
    captions = {}

    for tag, rel_path in modal_map.items():
        full_path = os.path.join(assets_root, rel_path)
        modality = get_modality(tag)
        try:
            if modality == "image":
                cap = caption_image(full_path, openai_client)
            elif modality in ["video", "audio"]:
                cap = caption_video_or_audio(full_path, vllm_client, vllm_model_name)
            elif modality == "document":
                cap = caption_document(full_path, openai_client)
            elif modality == "code":
                cap = caption_code(full_path)
            elif modality == "threeD":
                cap = caption_threed(full_path, pointllm_path)
            else:
                cap = "[UNKNOWN_MODALITY]"
        except Exception as e:
            cap = f"[ERROR_CAPTION:{e}]"
        captions[tag] = cap

    captioned_text = replace_placeholders(output["content"], captions)
    return captioned_text


def process_pair(response_item, gt_item, openai_client, vllm_client, pointllm_path, assets_root, vllm_model_name):
    """Caption response, score against GT, and return the augmented response item."""
    captioned_text = caption_response_item(response_item, openai_client, vllm_client, pointllm_path, assets_root, vllm_model_name)
    gt_text = gt_item["output"]["content"] if gt_item else ""
    score = compute_score(captioned_text, gt_text, openai_client) if gt_item else None

    # mutate a shallow copy so we don't alter original references if re-used elsewhere
    item = json.loads(json.dumps(response_item, ensure_ascii=False))
    item["output"]["captioned_response"] = captioned_text
    item["output"]["score"] = score
    return item


# ========== CLI ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--response-path", required=True)
    parser.add_argument("--ground-truth-path", required=True)
    parser.add_argument("--assets-root", default=".")
    parser.add_argument("--pointllm-model-path", default="/mnt/models/PointLLM_7B_v1.2", required=False)
    parser.add_argument("--vllm-endpoint", default="http://127.0.0.1:8003/v1")
    parser.add_argument("--vllm-model-name", required=True,
                    help="Model name/id exposed by vLLM (e.g., served-model-name or full path)")
    parser.add_argument("--traverse-mode", choices=["response", "gt"], default="gt",
                        help="Traversal order: iterate response (default) or iterate ground-truth then match response.")
    args = parser.parse_args()

    print(f"[INFO] Loading files ...")
    response_data = read_jsonl(args.response_path)
    ground_truth_data = read_jsonl(args.ground_truth_path)

    # build maps
    resp_map = {item["id"]: item for item in response_data if "id" in item}
    gt_map   = {item["id"]: item for item in ground_truth_data if "id" in item}

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    vllm_client   = OpenAI(base_url=args.vllm_endpoint, api_key="EMPTY")

    results = []

    if args.traverse_mode == "response":
        # existing behavior: iterate response first
        for item in tqdm(response_data, desc="Processing (response-mode)"):
            gt_item = gt_map.get(item["id"])
            if not gt_item:
                # skip or record with score=None; here we keep it and score=None
                captioned = caption_response_item(item, openai_client, vllm_client, args.pointllm_model_path, args.assets_root, args.vllm_model_name)
                out_item = json.loads(json.dumps(item, ensure_ascii=False))
                out_item["output"]["captioned_response"] = captioned
                out_item["output"]["score"] = None
                results.append(out_item)
                continue

            out_item = process_pair(item, gt_item, openai_client, vllm_client, args.pointllm_model_path, args.assets_root, args.vllm_model_name)
            results.append(out_item)

        out_path = args.response_path.replace(".jsonl", ".caption_scored.jsonl")

    else:
        # new behavior: iterate GT, match response
        for gt_item in tqdm(ground_truth_data, desc="Processing (gt-mode)"):
            resp_item = resp_map.get(gt_item["id"])
            if not resp_item:
                # strictly follow your spec: only process when matched; otherwise skip
                continue
            out_item = process_pair(resp_item, gt_item, openai_client, vllm_client, args.pointllm_model_path, args.assets_root, args.vllm_model_name)
            results.append(out_item)

        out_path = args.response_path.replace(".jsonl", ".caption_scored.traverse_gt.jsonl")

    write_jsonl(out_path, results)
    print(f"[SUCCESS] Saved results → {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
caption_eval_pipeline.py
-----------------------------------
Process multimodal response JSONL:
 - Convert all <tags> in output.content to captions
 - Match ground truth by id
 - Score captioned text vs ground truth using GPT-4o
 - Output new JSONL with captioned_response + score
"""

import os
import re
import json
import time
import base64
import tempfile
import argparse
from tqdm import tqdm
from typing import Dict, Any
import openai
from openai import OpenAI

# Optional heavy deps: open3d, pandas, matplotlib, etc.
import torch
import open3d as o3d
from transformers import AutoTokenizer
# from pointllm.model import PointLLMLlamaForCausalLM
# from pointllm.utils import disable_torch_init
# from pointllm.model.utils import KeywordsStoppingCriteria
# from pointllm.conversation import conv_templates, SeparatorStyle


# ========== Utility ==========
def read_jsonl(path: str):
    """Support multi-line JSON array or concatenated JSONs"""
    text = open(path, "r", encoding="utf-8").read()
    text = text.strip()
    if text.startswith("["):
        return json.loads(text)
    # fallback: multiple JSON blocks
    items = []
    buf = ""
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
    """Replace <image1> ... with <image1: caption>"""
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
        model="gpt-4o", messages=msg, max_tokens=1024, temperature=0)
    return resp.choices[0].message.content.strip()


def caption_video_or_audio(path, client, model_name):
    ext = os.path.splitext(path)[1].lower()
    mtype = "video" if ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"] else "audio"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    msg = [
        {"role": "user", "content": [
            {"type": "text", "text": f"Describe this {mtype} content in detail."},
            {f"type": f"{mtype}_url",
             f"{mtype}_url": {"url": f"data:{mtype}/mp4;base64,{b64}"}}
        ]}
    ]
    resp = client.chat.completions.create(
        model=model_name, messages=msg, max_tokens=1024, temperature=0)
    return resp.choices[0].message.content.strip()


def caption_document(path, client):
    prompt = "Describe the following document content in detail."
    with open(path, "rb") as f:
        data = f.read()
    # base64 for PDF/CSV/etc.
    b64 = base64.b64encode(data).decode("utf-8")
    msg = [
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:application/octet-stream;base64,{b64}"}}
        ]}
    ]
    resp = client.chat.completions.create(
        model="gpt-4o", messages=msg, max_tokens=1024, temperature=0)
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
    qs = "<point>" + "\nCaption this 3D model in detail."
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


# ========== GPT-4o scoring ==========
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
        "MUST be one of: 0.2, 0.4, 0.6, 0.8, 1.0."
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
        "Return STRICT JSON only: {\"score\": 0.2|0.4|0.6|0.8|1.0}\n\n"
        f"<<<BEGIN_PRED>>>\n{pred}\n<<<END_PRED>>>\n\n"
        f"<<<BEGIN_REF>>>\n{ref}\n<<<END_REF>>>"
    )

    resp = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},  # force pure JSON
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
        # sometimes the model returns as string; try to parse
        if isinstance(score, str):
            try:
                return _bin_score(float(score))
            except Exception:
                return None
        return None
    except Exception:
        # ultra-fallback: pull first number in [0,1] and bin it
        import re
        m = re.search(r"(?<!\d)(?:1(?:\.0+)?|0?\.\d+)(?!\d)", content)
        return _bin_score(float(m.group(0))) if m else None



# ========== Main Processing ==========
def process_item(item, gt_map, openai_client, vllm_client, pointllm_path, assets_root):
    output = item["output"]
    modal_map = output.get("modal", {})
    captions = {}

    for tag, rel_path in modal_map.items():
        full_path = os.path.join(assets_root, rel_path)
        modality = get_modality(tag)
        try:
            if modality == "image":
                cap = caption_image(full_path, openai_client)
            elif modality in ["video", "audio"]:
                cap = caption_video_or_audio(full_path, vllm_client, "qwen2.5-omni-7b")
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
    gt_text = gt_map[item["id"]]["output"]["content"]
    score = compute_score(captioned_text, gt_text, openai_client)

    output["captioned_response"] = captioned_text
    output["score"] = score
    return item


# ========== CLI ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--response-path", required=True)
    parser.add_argument("--ground-truth-path", required=True)
    parser.add_argument("--assets-root", default=".")
    parser.add_argument("--pointllm-model-path", default="/mnt/models/PointLLM_7B_v1.2", required=False)
    parser.add_argument("--vllm-endpoint", default="http://127.0.0.1:8003/v1")
    args = parser.parse_args()

    print(f"[INFO] Loading files ...")
    response_data = read_jsonl(args.response_path)
    ground_truth_data = read_jsonl(args.ground_truth_path)
    gt_map = {item["id"]: item for item in ground_truth_data}

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    vllm_client = OpenAI(base_url=args.vllm_endpoint, api_key="EMPTY")

    results = []
    for item in tqdm(response_data, desc="Processing"):
        processed = process_item(
            item, gt_map, openai_client, vllm_client,
            args.pointllm_model_path, args.assets_root)
        results.append(processed)

    out_path = args.response_path.replace(".jsonl", ".caption_scored.jsonl")
    write_jsonl(out_path, results)
    print(f"[SUCCESS] Saved results → {out_path}")


if __name__ == "__main__":
    main()

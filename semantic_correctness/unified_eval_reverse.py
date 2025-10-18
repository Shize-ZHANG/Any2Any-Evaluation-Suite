#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
caption_eval_pipeline.py (refactored)
-----------------------------------
Traversal modes:
 - response : iterate response → caption → match GT → score
 - gt (default): iterate ground truth → match response → if matched, caption response → score

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
import torch  # optional heavy dep
from transformers import AutoTokenizer  # optional for 3D captioner (kept for compatibility)


# ========== Utility ==========
def read_jsonl(path: str):
    """Support multi-line JSON array or concatenated JSONs"""
    text = open(path, "r", encoding="utf-8").read().strip()
    if not text:
        return []
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
    if buf.strip():
        # 最后残留一段解析失败就忽略，以免抛异常卡住全流程
        pass
    return items


def write_jsonl(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def is_url(s: str) -> bool:
    return isinstance(s, str) and s.startswith(("http://", "https://"))


# ---------- 占位符替换（支持不带数字的标签） ----------
_PLACEHOLDER_PATTERN = re.compile(r"<\s*([A-Za-z]+)(\d*)\s*>")

def replace_placeholders(content: str, captions: Dict[str, str]) -> str:
    """
    把 <image1> / <audio> / <video12> 等替换成 <image1: ...> / <audio: ...>。
    - captions 的 key 必须与占位符内的 tag 完全一致（含或不含数字）。
    - 未命中则原样保留（方便后续排查）。
    """
    def repl(m):
        name, num = m.groups()
        tag = f"{name}{num}"  # 'audio' + '' → 'audio'；'image' + '1' → 'image1'
        caption = captions.get(tag)
        if caption is None:
            # 兼容大小写差异（极少数数据）
            caption = captions.get(tag.lower(), captions.get(tag.capitalize()))
        if caption is None:
            return m.group(0)
        return f"<{tag}: {caption}>"
    return _PLACEHOLDER_PATTERN.sub(repl, content)


# ---------- 模态识别（优先包含，再前缀；避免 document_audio 误判） ----------
def get_modality(tag: str):
    t = (tag or "").lower()
    if "video" in t:   return "video"
    if "audio" in t:   return "audio"
    if "image" in t:   return "image"
    if "document" in t:return "document"
    if "code" in t:    return "code"
    if "threed" in t or "3d" in t: return "threeD"
    for p in ["video","audio","image","document","code","threed","3d"]:
        if t.startswith(p):
            return "threeD" if p in ("threed","3d") else p
    return None


# ========== Captioners ==========
# --- OpenAI clients (从环境读取，不硬编码) ---
def make_openai_clients(vllm_base: str):
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("[WARN] OPENAI_API_KEY is not set; OpenAI-based caption/scoring may fail.")
    openai_client = OpenAI(api_key=openai_key)

    vllm_key = os.getenv("VLLM_API_KEY", "EMPTY")
    vllm_client = OpenAI(api_key=vllm_key, base_url=vllm_base)
    return openai_client, vllm_client


def caption_image(path_or_url: str, client_openai: OpenAI) -> str:
    """
    图片 → dense caption（用 gpt-5-mini；本地文件转 base64，URL 直接传）
    """
    try:
        if is_url(path_or_url):
            content_block = {"type": "image_url", "image_url": {"url": path_or_url}}
        else:
            with open(path_or_url, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            content_block = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}

        msg = [{"role": "user", "content": [
            {"type": "text", "text": "Describe this image in detail."},
            content_block
        ]}]
        resp = client_openai.chat.completions.create(
            model="gpt-5-mini",
            messages=msg,            
            max_completion_tokens=1024
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"[ERROR_CAPTION:{e}]"


def caption_document(path_or_url: str, client_openai: OpenAI) -> str:
    """
    文档（PDF/CSV/TXT...）→ 粗粒度描述。简化处理：统一按 octet-stream 走。
    """
    try:
        if is_url(path_or_url):
            # URL 直接传给模型有的构建不支持非图像，这里统一转 base64
            import requests
            r = requests.get(path_or_url, timeout=300)
            r.raise_for_status()
            data = r.content
        else:
            with open(path_or_url, "rb") as f:
                data = f.read()
        b64 = base64.b64encode(data).decode("utf-8")

        msg = [{"role": "user", "content": [
            {"type": "text", "text": "Describe the following document content in detail."},
            {"type": "image_url", "image_url": {"url": f"data:application/octet-stream;base64,{b64}"}}
        ]}]
        resp = client_openai.chat.completions.create(
            model="gpt-5-mini",
            messages=msg,
            max_tokens=1024
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"[ERROR_CAPTION:{e}]"


def caption_code(path_or_url: str) -> str:
    try:
        if is_url(path_or_url):
            import requests
            r = requests.get(path_or_url, timeout=60)
            r.raise_for_status()
            text = r.text
        else:
            text = open(path_or_url, "r", encoding="utf-8", errors="ignore").read()
        return text[:6000]
    except Exception as e:
        return f"[ERROR_CAPTION:{e}]"


def caption_av_vllm(path_or_url: str, vllm_client: OpenAI, model_name: str, kind_hint: str) -> str:
    """
    音/视频 → vLLM（OpenAI 兼容）：
    - URL 直接传；
    - 本地文件 → data:*;base64；
    - 视频使用 {"type":"video_url"}；音频使用 {"type":"audio_url"}；
    """
    import subprocess

    try:
        is_video = (kind_hint == "video")
        is_audio = (kind_hint == "audio")
        if not (is_video or is_audio):
            raise ValueError(f"caption_av_vllm requires kind_hint in ('video','audio'), got: {kind_hint}")

        if is_url(path_or_url):
            payload_url = path_or_url
            print(f"[vLLM] Using remote URL directly: {payload_url}")
        else:
            ext = os.path.splitext(path_or_url)[1].lower()
            local_path = path_or_url

            # .webm 转码
            if ext == ".webm" and is_video:
                print("[Convert] Detected .webm video, converting to .mp4 ...")
                converted_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                subprocess.run([
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", local_path,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac",
                    converted_path
                ], check=True)
                local_path = converted_path
            elif ext == ".webm" and is_audio:
                print("[Convert] Detected .webm audio, converting to .m4a ...")
                converted_path = tempfile.NamedTemporaryFile(suffix=".m4a", delete=False).name
                subprocess.run([
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", local_path,
                    "-acodec", "aac",
                    "-ar", "44100",
                    converted_path
                ], check=True)
                local_path = converted_path

            # 视频清洗（修时间戳/帧异常）
            if is_video:
                try:
                    clean_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                    subprocess.run([
                        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-fflags", "+genpts", "-avoid_negative_ts", "make_zero",
                        "-i", local_path,
                        "-map", "0:v:0", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-an", clean_path
                    ], check=True)
                    local_path = clean_path
                    print(f"[Fix] Video cleaned and re-encoded: {local_path}")
                except Exception as e:
                    print(f"[Warn] ffmpeg clean step failed: {e}")

            with open(local_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")

            if is_video:
                payload_url = f"data:video/mp4;base64,{b64}"
                print(f"[Encode] Video as base64 ({len(b64)/1e6:.2f} MB text)")
            else:
                payload_url = f"data:audio/mp3;base64,{b64}"
                print(f"[Encode] Audio as base64 ({len(b64)/1e6:.2f} MB text)")

        content = [
            {"type": "text",
             "text": ("Describe this video in detail — scenes, actions, objects, emotions, environment, transitions. English only.")
                     if is_video else
                     ("Transcribe this audio to verbatim English text only. No summary or commentary.")}
        ]
        if is_video:
            content.append({"type": "video_url", "video_url": {"url": payload_url}})
        else:
            content.append({"type": "audio_url", "audio_url": {"url": payload_url}})

        print("[Infer] Sending to vLLM ...")
        resp = vllm_client.chat.completions.create(
            model=model_name,
            max_completion_tokens=1024,
            temperature=0.2,
            messages=[{"role": "user", "content": content}]
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"[ERROR_CAPTION:{e}]"


# ========== gpt-5-mini scoring ==========
def _bin_score(x: float):
    """Map float in [0,1] to nearest of {0.0,0.2,0.4,0.6,0.8,1.0}."""
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    if x is None:
        return None
    try:
        x = float(x)
    except Exception:
        return None
    x = max(0.0, min(1.0, x))
    return min(bins, key=lambda b: abs(b - x))


def compute_score(pred: str, ref: str, client_openai: OpenAI):
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
        "- 0.0: Completely incorrect/irrelevant. Off-topic or contradicts core facts; essentially no overlap with the reference.\n\n"
        "Return STRICT JSON only: {\"score\": 0.0|0.2|0.4|0.6|0.8|1.0}\n\n"
        f"<<<BEGIN_PRED>>>\n{pred}\n<<<END_PRED>>>\n\n"
        f"<<<BEGIN_REF>>>\n{ref}\n<<<END_REF>>>"
    )

    try:
        resp = client_openai.chat.completions.create(
            model="gpt-5-mini",
            response_format={"type": "json_object"},
            temperature=0,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        content = resp.choices[0].message.content
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
    except Exception as e:
        # 兜底正则提取
        try:
            m = re.search(r"(?<!\d)(?:0(?:\.0+)?|1(?:\.0+)?|0?\.\d+)(?!\d)", content)
            return _bin_score(float(m.group(0))) if m else None
        except Exception:
            return None


# ========== Pair processing (shared) ==========
def caption_response_item(response_item, openai_client, vllm_client, pointllm_path, assets_root, vllm_model_name):
    """
    为 response_item['output'] 里的 modal 做 caption，并返回替换后的文本。
    - 支持 URL 或 assets_root + 相对路径
    - 日志打印每个 tag 的模态与来源
    """
    output = response_item.get("output", {})
    modal_map: Dict[str, str] = output.get("modal", {}) or {}
    captions: Dict[str, str] = {}

    for tag, rel in modal_map.items():
        src = rel if is_url(rel) else os.path.join(assets_root, rel)
        modality = get_modality(tag)
        print(f"[CAPTION] id={response_item.get('id')} tag={tag} modality={modality} src={src}")

        try:
            if modality == "image":
                cap = caption_image(src, openai_client)
            elif modality == "video":
                cap = caption_av_vllm(src, vllm_client, vllm_model_name, kind_hint="video")
            elif modality == "audio":
                cap = caption_av_vllm(src, vllm_client, vllm_model_name, kind_hint="audio")
            elif modality == "document":
                cap = caption_document(src, openai_client)
            elif modality == "code":
                cap = caption_code(src)
            elif modality == "threeD":
                # 3D 留作扩展（你的数据若没有，可以忽略）
                cap = "[UNSUPPORTED_3D_IN_THIS_BUILD]"
            else:
                cap = "[UNKNOWN_MODALITY]"
        except Exception as e:
            cap = f"[ERROR_CAPTION:{e}]"
        captions[tag] = cap

    content = output.get("content", "")
    captioned_text = replace_placeholders(content, captions)
    return captioned_text


def process_pair(response_item, gt_item, openai_client, vllm_client, pointllm_path, assets_root, vllm_model_name):
    captioned_text = caption_response_item(response_item, openai_client, vllm_client, pointllm_path, assets_root, vllm_model_name)
    gt_text = (gt_item or {}).get("output", {}).get("content", "") if gt_item else ""
    score = compute_score(captioned_text, gt_text, openai_client) if gt_item else None

    if score is None:
        score = 0.0

    item = json.loads(json.dumps(response_item, ensure_ascii=False))
    item.setdefault("output", {})
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

    parser.add_argument("--vllm-endpoint", default="http://127.0.0.1:8009/v1",
                        help="OpenAI-compatible endpoint served by vLLM, e.g., http://host:port/v1")
    parser.add_argument("--vllm-model-name",  default="Qwen/Qwen2.5-Omni-3B", required=False,
                        help="Model name/id exposed by vLLM (served-model-name or path)")
    parser.add_argument("--traverse-mode", choices=["response", "gt"], default="gt",
                        help="Traversal order: iterate ground-truth (default) or iterate response.")

    args = parser.parse_args()

    print(f"[INFO] Loading files ...")
    response_data = read_jsonl(args.response_path)
    ground_truth_data = read_jsonl(args.ground_truth_path)

    # build maps
    resp_map = {item.get("id"): item for item in response_data if "id" in item}
    gt_map   = {item.get("id"): item for item in ground_truth_data if "id" in item}

    # clients
    openai_client, vllm_client = make_openai_clients(args.vllm_endpoint)

    results = []

    if args.traverse_mode == "response":
        # 以响应为主：即使没有 GT，也会做 caption 并写 score=None
        for item in tqdm(response_data, desc="Processing (response-mode)"):
            rid = item.get("id")
            gt_item = gt_map.get(rid)
            if not gt_item:
                captioned = caption_response_item(item, openai_client, vllm_client,
                                                  args.pointllm_model_path, args.assets_root, args.vllm_model_name)
                out_item = json.loads(json.dumps(item, ensure_ascii=False))
                out_item.setdefault("output", {})
                out_item["output"]["captioned_response"] = captioned
                out_item["output"]["score"] = None
                results.append(out_item)
                continue

            out_item = process_pair(item, gt_item, openai_client, vllm_client,
                                    args.pointllm_model_path, args.assets_root, args.vllm_model_name)
            results.append(out_item)

        out_path = args.response_path.replace(".jsonl", ".caption_scored.jsonl")

    else:
        # 以 GT 为主：只有 id 匹配的样本才会进入处理（你的默认）
        for gt_item in tqdm(ground_truth_data, desc="Processing (gt-mode)"):
            gid = gt_item.get("id")
            resp_item = resp_map.get(gid)
            if not resp_item:
                print(f"[WARN] Skip GT id={gid}: no matching response; no caption/replacement done.")
                continue
            out_item = process_pair(resp_item, gt_item, openai_client, vllm_client,
                                    args.pointllm_model_path, args.assets_root, args.vllm_model_name)
            results.append(out_item)

        out_path = args.response_path.replace(".jsonl", ".caption_scored.traverse_gt.jsonl")

    write_jsonl(out_path, results)
    print(f"[SUCCESS] Saved results → {out_path}")


if __name__ == "__main__":
    main()

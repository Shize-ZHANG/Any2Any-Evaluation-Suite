#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
caption_eval_pipeline.py (refactored + 3D captioning)
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
import torch
from transformers import AutoTokenizer


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


# ---------- 占位符替换（保持你的原逻辑，不改） ----------
_PLACEHOLDER_PATTERN = re.compile(r"<\s*([A-Za-z]+)(\d*)\s*>")


def replace_placeholders(content: str, captions: Dict[str, str]) -> str:
    """
    把 <image1> / <audio> / <video12> 等替换成 <image1: ...> / <audio: ...>。
    - captions 的 key 必须与占位符内的 tag 完全一致（含或不含数字）。
    - 未命中则原样保留（方便后续排查）。
    """
    def repl(m):
        name, num = m.groups()
        tag = f"{name}{num}"
        caption = captions.get(tag)

        if caption is None:
            # 兼容大小写差异（极少数数据）
            caption = captions.get(tag.lower(), captions.get(tag.capitalize()))

        if caption is None:
            return m.group(0)

        return f"<{tag}: {caption}>"

    return _PLACEHOLDER_PATTERN.sub(repl, content)


# ---------- 模态识别 ----------
def get_modality(tag: str):
    t = (tag or "").lower()

    if "video" in t:
        return "video"
    if "audio" in t:
        return "audio"
    if "image" in t:
        return "image"
    if "document" in t:
        return "document"
    if "code" in t:
        return "code"
    if "threed" in t or "3d" in t:
        return "threeD"

    for p in ["video", "audio", "image", "document", "code", "threed", "3d"]:
        if t.startswith(p):
            return "threeD" if p in ("threed", "3d") else p

    return None


# ========== Captioners ==========
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
            content_block = {
                "type": "image_url",
                "image_url": {"url": path_or_url}
            }
        else:
            with open(path_or_url, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")

            content_block = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            }

        msg = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in detail."},
                    content_block
                ]
            }
        ]

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
            import requests
            r = requests.get(path_or_url, timeout=300)
            r.raise_for_status()
            data = r.content
        else:
            with open(path_or_url, "rb") as f:
                data = f.read()

        b64 = base64.b64encode(data).decode("utf-8")

        msg = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the following document content in detail."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:application/octet-stream;base64,{b64}"
                        }
                    }
                ]
            }
        ]

        resp = client_openai.chat.completions.create(
            model="gpt-5-mini",
            messages=msg,
            max_completion_tokens=1024
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
    - 视频使用 {"type":"video_url"}；
    - 音频使用 {"type":"audio_url"}。
    """
    import subprocess

    try:
        is_video = (kind_hint == "video")
        is_audio = (kind_hint == "audio")

        if not (is_video or is_audio):
            raise ValueError(
                f"caption_av_vllm requires kind_hint in ('video','audio'), got: {kind_hint}"
            )

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

                subprocess.run(
                    [
                        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-i", local_path,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac",
                        converted_path
                    ],
                    check=True
                )

                local_path = converted_path

            elif ext == ".webm" and is_audio:
                print("[Convert] Detected .webm audio, converting to .m4a ...")
                converted_path = tempfile.NamedTemporaryFile(suffix=".m4a", delete=False).name

                subprocess.run(
                    [
                        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-i", local_path,
                        "-acodec", "aac",
                        "-ar", "44100",
                        converted_path
                    ],
                    check=True
                )

                local_path = converted_path

            # 视频清洗（修时间戳/帧异常）
            if is_video:
                try:
                    clean_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

                    subprocess.run(
                        [
                            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                            "-fflags", "+genpts",
                            "-avoid_negative_ts", "make_zero",
                            "-i", local_path,
                            "-map", "0:v:0",
                            "-c:v", "libx264",
                            "-pix_fmt", "yuv420p",
                            "-an",
                            clean_path
                        ],
                        check=True
                    )

                    local_path = clean_path
                    print(f"[Fix] Video cleaned and re-encoded: {local_path}")

                except Exception as e:
                    print(f"[Warn] ffmpeg clean step failed: {e}")

            with open(local_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")

            if is_video:
                payload_url = f"data:video/mp4;base64,{b64}"
                print(f"[Encode] Video as base64 ({len(b64) / 1e6:.2f} MB text)")
            else:
                payload_url = f"data:audio/mp3;base64,{b64}"
                print(f"[Encode] Audio as base64 ({len(b64) / 1e6:.2f} MB text)")

        content = [
            {
                "type": "text",
                "text": (
                    "Describe this video in detail — scenes, actions, objects, emotions, "
                    "environment, transitions. English only."
                )
                if is_video else
                "Transcribe this audio to verbatim English text only. No summary or commentary."
            }
        ]

        if is_video:
            content.append(
                {
                    "type": "video_url",
                    "video_url": {"url": payload_url}
                }
            )
        else:
            content.append(
                {
                    "type": "audio_url",
                    "audio_url": {"url": payload_url}
                }
            )

        print("[Infer] Sending to vLLM ...")

        resp = vllm_client.chat.completions.create(
            model=model_name,
            max_completion_tokens=1024,
            temperature=0.2,
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ]
        )

        return (resp.choices[0].message.content or "").strip()

    except Exception as e:
        return f"[ERROR_CAPTION:{e}]"


# ========== PointLLM 3D Captioner ==========
_POINTLLM_CACHE = {}


def _download_3d_if_url(path_or_url: str) -> str:
    """
    PointLLM / Open3D 更适合处理本地 3D 文件。
    如果 3D 资源是 URL，则先下载到临时文件。
    """
    if not is_url(path_or_url):
        return path_or_url

    import requests
    from urllib.parse import urlparse

    parsed = urlparse(path_or_url)
    ext = os.path.splitext(parsed.path)[1].lower()

    if not ext:
        ext = ".ply"

    tmp_path = tempfile.NamedTemporaryFile(suffix=ext, delete=False).name

    print(f"[3D] Downloading remote 3D asset: {path_or_url}")

    with requests.get(path_or_url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    print(f"[3D] Saved remote 3D asset to: {tmp_path}")

    return tmp_path


def _load_pointllm_once(model_path: str):
    """
    只加载一次 PointLLM，避免每个 3D 样本都重复加载模型。
    """
    global _POINTLLM_CACHE

    key = os.path.abspath(model_path)

    if key in _POINTLLM_CACHE:
        return _POINTLLM_CACHE[key]

    try:
        from pointllm.model import PointLLMLlamaForCausalLM
        from pointllm.utils import disable_torch_init
        from pointllm.model.utils import KeywordsStoppingCriteria
        from pointllm.conversation import conv_templates, SeparatorStyle
    except Exception as e:
        raise ImportError(
            "PointLLM dependencies are not available. "
            "Please make sure pointllm is installed and importable in this environment. "
            f"Original error: {e}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("PointLLM 3D captioning requires CUDA, but torch.cuda.is_available() is False.")

    print(f"[3D] Loading PointLLM from: {model_path}")

    disable_torch_init()

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = PointLLMLlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    ).cuda()

    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
    model.eval()

    deps = {
        "tokenizer": tokenizer,
        "model": model,
        "KeywordsStoppingCriteria": KeywordsStoppingCriteria,
        "conv_templates": conv_templates,
        "SeparatorStyle": SeparatorStyle,
    }

    _POINTLLM_CACHE[key] = deps

    print("[3D] PointLLM loaded successfully.")

    return deps


def load_point_cloud(pc_path: str, pointnum: int = 8192):
    """
    读取 3D 文件并转成 PointLLM 需要的 normalized point cloud tensor。

    支持：
    - 点云文件：.ply / .pcd / .xyz / .xyzn / .xyzrgb 等 Open3D 可读格式
    - Mesh 文件：.off / .obj / .stl / .glb / .gltf
      Mesh 会先 uniform sample 成 point cloud。
    """
    try:
        import open3d as o3d
        import numpy as np
    except Exception as e:
        raise ImportError(
            "open3d and numpy are required for 3D captioning. "
            f"Original error: {e}"
        )

    pc_path = _download_3d_if_url(pc_path)
    ext = os.path.splitext(pc_path)[1].lower()

    mesh_exts = {".off", ".obj", ".stl", ".glb", ".gltf"}
    pcd = None

    if ext in mesh_exts:
        print(f"[3D] Reading mesh file: {pc_path}")
        mesh = o3d.io.read_triangle_mesh(pc_path)

        if mesh is None or len(mesh.vertices) == 0:
            raise ValueError(f"Failed to read mesh or empty mesh: {pc_path}")

        pcd = mesh.sample_points_uniformly(number_of_points=pointnum)

    else:
        print(f"[3D] Reading point cloud file: {pc_path}")
        pcd = o3d.io.read_point_cloud(pc_path)

        # 有些 .ply 实际是 mesh；如果按点云读不到点，则再尝试按 mesh 读
        if pcd is None or len(pcd.points) == 0:
            print("[3D] Point cloud is empty. Trying to read it as a mesh ...")
            mesh = o3d.io.read_triangle_mesh(pc_path)

            if mesh is not None and len(mesh.vertices) > 0:
                pcd = mesh.sample_points_uniformly(number_of_points=pointnum)

    if pcd is None or len(pcd.points) == 0:
        raise ValueError(f"Empty or unsupported 3D file: {pc_path}")

    pts = torch.from_numpy(np.asarray(pcd.points)).float()

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Invalid point cloud shape: {pts.shape}, path={pc_path}")

    # 中心化
    centroid = pts.mean(dim=0, keepdim=True)
    pts = pts - centroid

    # 归一化到单位尺度
    scale = pts.pow(2).sum(dim=1).sqrt().max() + 1e-6
    pts = pts / scale

    # 采样或 padding 到固定点数
    if pts.shape[0] >= pointnum:
        idx = torch.randperm(pts.shape[0])[:pointnum]
        pts = pts[idx]
    else:
        pad_idx = torch.randint(0, pts.shape[0], (pointnum - pts.shape[0],))
        pad = pts[pad_idx]
        pts = torch.cat([pts, pad], dim=0)

    return pts.unsqueeze(0)


@torch.inference_mode()
def caption_threed(pc_path: str, model_path: str, pointnum: int = 8192) -> str:
    """
    3D model / point cloud → dense caption.

    输入可以是本地路径，也可以是 URL。
    输出是英文 caption，后续会被 replace_placeholders() 填回 <threeD1: ...>。
    """
    try:
        deps = _load_pointllm_once(model_path)

        tokenizer = deps["tokenizer"]
        model = deps["model"]
        KeywordsStoppingCriteria = deps["KeywordsStoppingCriteria"]
        conv_templates = deps["conv_templates"]
        SeparatorStyle = deps["SeparatorStyle"]

        point_clouds = load_point_cloud(pc_path, pointnum=pointnum)
        point_clouds = point_clouds.cuda().to(model.dtype)

        conv_mode = "vicuna_v1_1"
        conv = conv_templates[conv_mode].copy()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        qs = "<point>\nCaption this 3D model in detail. English only."

        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()

        inputs = tokenizer([prompt])
        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        stopping = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

        print(f"[3D] Running PointLLM captioning for: {pc_path}")

        out_ids = model.generate(
            input_ids=input_ids,
            point_clouds=point_clouds,
            do_sample=True,
            temperature=0.2,
            top_p=0.95,
            max_length=2048,
            stopping_criteria=[stopping],
        )

        input_token_len = input_ids.shape[1]

        txt = tokenizer.batch_decode(
            out_ids[:, input_token_len:],
            skip_special_tokens=True
        )[0].strip()

        if stop_str and txt.endswith(stop_str):
            txt = txt[:-len(stop_str)].strip()

        return txt

    except Exception as e:
        return f"[ERROR_CAPTION_3D:{e}]"


# ========== gpt-5-mini scoring ==========
def compute_score(pred: str, ref: str, client_openai):
    """
    语义一致性（1–5档）评分：更侧重“是否满足任务意图/语义要求”，
    不强制与参考中的具体实体一一对应，除非题面/参考明确要求唯一性或精确名单。
    返回 1..5 的整数。
    """

    system = (
        "You are a strict but intention-aware grader for semantic consistency. "
        "Evaluate whether the Candidate (Pred) satisfies the task intent and semantic constraints "
        "expressed by the Reference (Ref). "
        "Ignore style, fluency, tone, politeness, formatting, and layout. "
        "Treat as equivalent: paraphrases, reordering, mild numeric rounding, unit conversions, "
        "synonyms, hypernym/hyponym substitutions (category↔exemplar), and alternative but correct exemplars "
        "from the same valid set (e.g., representative American foods: hamburger, fries, hot dog). "
        "Require strict entity alignment ONLY when the intent/Ref explicitly demands unique identity, "
        "exact names, exact counts/order, roles, or closed lists (e.g., 'exactly two authors: X and Y'). "
        "Consider lists in Ref as NON-EXHAUSTIVE unless they explicitly say 'must include'/'only'/'exactly N'. "
        "Penalize contradictions, violation of explicit constraints (wrong entity/category/country/time), "
        "and key omissions that make the main conclusion invalid. "
        "Return ONLY a JSON object with key 'score' and value in {5,4,3,2,1}."
    )

    user = (
        "Score semantic consistency of Pred relative to Ref by choosing EXACTLY ONE value.\n\n"
        "Equivalence policy:\n"
        "- Alternative-but-correct exemplars from the same valid set are acceptable "
        "(e.g., Q: 'Name a representative American food.' Ref: 'hamburger'; Pred: 'fries' → treat as equivalent).\n"
        "- Category↔exemplar and synonym substitutions are acceptable if intent is preserved.\n"
        "- Enforce strict entity identity ONLY if the prompt/Ref specifies unique names, exact counts/order, roles, or closed lists.\n"
        "- Assume enumerations in Ref are NON-EXHAUSTIVE unless it says 'must include/only/exactly'.\n\n"
        "Rubric (5 = best, 1 = worst):\n"
        "- 5: Fully satisfies intent and preserves all explicit constraints; no contradictions. "
        "Key facts/relations are correct, allowing alternative correct exemplars or category↔exemplar shifts.\n"
        "- 4: Nearly satisfies intent with only minor non-critical gaps or imprecision; main conclusion unchanged; "
        "no major constraint violations.\n"
        "- 3: Partially satisfies intent. About half of key requirements covered; notable omissions or local "
        "misunderstandings; main conclusion only partially supported.\n"
        "- 2: Low consistency. Few correct points; most requirements unmet or important contradictions/constraint violations; "
        "main conclusion mostly invalid.\n"
        "- 1: Wholly inconsistent/irrelevant. Opposes core facts/intent or almost no semantic overlap; "
        "empty/meaningless answers also receive 1.\n\n"
        "Return STRICT JSON only: {\"score\": 5|4|3|2|1}\n\n"
        f"<<<BEGIN_PRED>>>\n{pred}\n<<<END_PRED>>>\n\n"
        f"<<<BEGIN_REF>>>\n{ref}\n<<<END_REF>>>"
    )

    raw_output = ""

    try:
        resp = client_openai.chat.completions.create(
            model="gpt-5-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )

        raw_output = (resp.choices[0].message.content or "").strip()

        print("\n========== [RAW GPT OUTPUT] ==========")
        print(raw_output)
        print("======================================\n")

        data = json.loads(raw_output)
        score = data.get("score", None)

        if isinstance(score, (int, float)) and float(score).is_integer():
            si = int(score)
            if 1 <= si <= 5:
                return si

        if isinstance(score, str):
            m = re.search(r"\b([1-5])\b", score.strip())
            if m:
                return int(m.group(1))

        m2 = re.search(r"\b([1-5])\b", raw_output)
        return int(m2.group(1)) if m2 else None

    except Exception as e:
        print(f"[ERROR][compute_score] Exception: {e}")

        try:
            print(f"[ERROR][compute_score] raw content: {raw_output}")
        except Exception:
            pass

        try:
            m = re.search(r"\b([1-5])\b", str(e))
            return int(m.group(1)) if m else None
        except Exception:
            return None


# ========== Pair processing ==========
def caption_response_item(
    response_item,
    openai_client,
    vllm_client,
    pointllm_path,
    assets_root,
    vllm_model_name
):
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

        print(
            f"[CAPTION] id={response_item.get('id')} "
            f"tag={tag} modality={modality} src={src}"
        )

        try:
            if modality == "image":
                cap = caption_image(src, openai_client)

            elif modality == "video":
                cap = caption_av_vllm(
                    src,
                    vllm_client,
                    vllm_model_name,
                    kind_hint="video"
                )

            elif modality == "audio":
                cap = caption_av_vllm(
                    src,
                    vllm_client,
                    vllm_model_name,
                    kind_hint="audio"
                )

            elif modality == "document":
                cap = caption_document(src, openai_client)

            elif modality == "code":
                cap = caption_code(src)

            elif modality == "threeD":
                cap = caption_threed(src, pointllm_path)

            else:
                cap = "[UNKNOWN_MODALITY]"

        except Exception as e:
            cap = f"[ERROR_CAPTION:{e}]"

        captions[tag] = cap

    content = output.get("content", "")
    captioned_text = replace_placeholders(content, captions)

    return captioned_text


def process_pair(
    response_item,
    gt_item,
    openai_client,
    vllm_client,
    pointllm_path,
    assets_root,
    vllm_model_name
):
    captioned_text = caption_response_item(
        response_item,
        openai_client,
        vllm_client,
        pointllm_path,
        assets_root,
        vllm_model_name
    )

    gt_text = (gt_item or {}).get("output", {}).get("content", "") if gt_item else ""
    score = compute_score(captioned_text, gt_text, openai_client) if gt_item else None

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
    parser.add_argument(
        "--pointllm-model-path",
        default="/mnt/models/PointLLM_7B_v1.2",
        required=False
    )

    parser.add_argument(
        "--vllm-endpoint",
        default="http://127.0.0.1:8009/v1",
        help="OpenAI-compatible endpoint served by vLLM, e.g., http://host:port/v1"
    )

    parser.add_argument(
        "--vllm-model-name",
        default="Qwen/Qwen2.5-Omni-3B",
        required=False,
        help="Model name/id exposed by vLLM, e.g., served-model-name or model path"
    )

    parser.add_argument(
        "--traverse-mode",
        choices=["response", "gt"],
        default="gt",
        help="Traversal order: iterate ground-truth (default) or iterate response."
    )

    args = parser.parse_args()

    print("[INFO] Loading files ...")

    response_data = read_jsonl(args.response_path)
    ground_truth_data = read_jsonl(args.ground_truth_path)

    resp_map = {
        item.get("id"): item
        for item in response_data
        if "id" in item
    }

    gt_map = {
        item.get("id"): item
        for item in ground_truth_data
        if "id" in item
    }

    openai_client, vllm_client = make_openai_clients(args.vllm_endpoint)

    results = []

    if args.traverse_mode == "response":
        # 以响应为主：即使没有 GT，也会做 caption 并写 score=None
        for item in tqdm(response_data, desc="Processing (response-mode)"):
            rid = item.get("id")
            gt_item = gt_map.get(rid)

            if not gt_item:
                captioned = caption_response_item(
                    item,
                    openai_client,
                    vllm_client,
                    args.pointllm_model_path,
                    args.assets_root,
                    args.vllm_model_name
                )

                out_item = json.loads(json.dumps(item, ensure_ascii=False))
                out_item.setdefault("output", {})
                out_item["output"]["captioned_response"] = captioned
                out_item["output"]["score"] = None

                results.append(out_item)
                continue

            out_item = process_pair(
                item,
                gt_item,
                openai_client,
                vllm_client,
                args.pointllm_model_path,
                args.assets_root,
                args.vllm_model_name
            )

            results.append(out_item)

        out_path = args.response_path.replace(".jsonl", ".caption_scored.jsonl")

    else:
        # 以 GT 为主：只有 id 匹配的样本才会进入处理
        for gt_item in tqdm(ground_truth_data, desc="Processing (gt-mode)"):
            gid = gt_item.get("id")
            resp_item = resp_map.get(gid)

            if not resp_item:
                print(f"[WARN] Skip GT id={gid}: no matching response; no caption/replacement done.")
                continue

            out_item = process_pair(
                resp_item,
                gt_item,
                openai_client,
                vllm_client,
                args.pointllm_model_path,
                args.assets_root,
                args.vllm_model_name
            )

            results.append(out_item)

        out_path = args.response_path.replace(".jsonl", ".caption_scored.traverse_gt.jsonl")

    write_jsonl(out_path, results)

    print(f"[SUCCESS] Saved results → {out_path}")


if __name__ == "__main__":
    main()

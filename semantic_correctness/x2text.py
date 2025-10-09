# x2text.py
import os
import pandas as pd
import io
os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchvision"
import re
import base64
from io import BytesIO
import easyocr
import requests
from qwen_omni_utils import process_mm_info
from pathlib import Path
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
import warnings
import logging
from transformers.utils import logging as hf_logging
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 关闭所有 warning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 关闭 transformers 的 log
hf_logging.set_verbosity_error()
load_dotenv()

# --- 配置 GitHub ---
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_OWNER = "yesbobo1"
GITHUB_REPO = "any2any-entertainment"
GITHUB_BRANCH = "main"

# 关闭 root logger 的 WARNING 信息
logging.getLogger().setLevel(logging.ERROR)

# ======== 模型加载（video/audio 模态共用） ========
processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-3B",
    torch_dtype="auto",
    device_map="auto",
    use_safetensors=True
)
USE_AUDIO_IN_VIDEO = False

# ======== 各模态接口函数 ========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

reader = easyocr.Reader(['en'], gpu=False)  # GPU 可选

def upload_to_github(file_bytes, file_path_in_repo, commit_message="Upload projection image"):
    """
    上传文件到 GitHub 并返回 raw URL
    """
    url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents/{file_path_in_repo}"
    content_b64 = base64.b64encode(file_bytes).decode("utf-8")
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

    # 检查文件是否存在，决定 PUT 请求是否带 sha
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        sha = r.json()["sha"]
        data = {"message": commit_message, "content": content_b64, "sha": sha, "branch": GITHUB_BRANCH}
    else:
        data = {"message": commit_message, "content": content_b64, "branch": GITHUB_BRANCH}

    resp = requests.put(url, headers=headers, json=data)
    if resp.status_code in [200, 201]:
        raw_url = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{file_path_in_repo}"
        return raw_url
    else:
        raise Exception(f"GitHub upload failed: {resp.status_code}, {resp.text}")

def ocr_extract_text(url):
    try:
        text_list = reader.readtext(url, detail=0)  # 直接传 URL
        return ' '.join(text_list)
    except Exception as e:
        print(f"OCR failed for {url}: {e}")
        return ""

def image_to_text(image_url, model_name="gpt-4o"):
    """
    使用 GPT-4o 对图片进行 dense caption 生成
    image_url: 图片的公开 URL (如 GitHub raw 链接) 或本地文件上传后的 URL
    model_name: GPT 模型名称，默认 gpt-4o
    """
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a visual captioning assistant. Provide dense, detailed descriptions."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please generate a dense caption for this image."},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            max_tokens=500
        )
        caption = response.choices[0].message.content.strip()
        return caption
    except Exception as e:
        print("Error generating image caption:", e)
        return ""

def csv_to_png_base64(csv_url):
    """
    将 CSV 渲染成 PNG，并返回 base64 编码
    """
    try:
        df = pd.read_csv(csv_url)
        fig, ax = plt.subplots(figsize=(min(12, len(df.columns)*1.2), 6))
        ax.axis('off')
        table = ax.table(cellText=df.values,
                         colLabels=df.columns,
                         loc='center',
                         cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.2)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=200)
        plt.close(fig)
        buf.seek(0)

        # 转 base64
        img = Image.open(buf)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_b64
    except Exception as e:
        print("Error converting CSV to PNG:", e)
        return None

def document_to_text(input_file_url, model_name="gpt-4o"):
    """
    根据文件类型决定处理方式:
    - 如果是 csv 文件，转成 png 后进行 OCR，并带上 base64 图片
    - 如果是 pdf/docx/txt 文件，直接交给 GPT-4o
    - 其他情况（如图片），先用 OCR 提取文本
    """
    file_ext = os.path.splitext(input_file_url.split("?")[0])[1].lower()
    no_ocr_exts = {".pdf", ".docx", ".txt"}

    extracted_text = ""
    img_base64 = None

    if file_ext == ".csv":
        # CSV → PNG → OCR
        img_base64 = csv_to_png_base64(input_file_url)
        """if img_base64:
            extracted_text = ocr_extract_text(img_base64)"""  # 这里假设 ocr_extract_text 支持 base64 图片输入
    elif file_ext not in no_ocr_exts:
        # 其他文件（如图片）
        extracted_text = ocr_extract_text(input_file_url)

    # 构造 prompt
    prompt = f"""
    You are a visual and document captioning assistant. 

    Below is a document (file or image) and its extracted text (if available). 
    Generate a **dense description** that covers both the visual layout 
    and the textual content. Describe objects, layout, text content, 
    relationships, and context in detail.

    [File URL]
    {input_file_url}

    [Extracted Text]
    {extracted_text if extracted_text else "N/A"}

    [Response Requirement]
    Return only a dense caption, do not include any system/user markers or extra explanation.
    """

    # 构造输入消息
    messages = [
        {"role": "system", "content": "You are a strict visual captioning assistant."},
        {"role": "user", "content": prompt}
    ]

    if img_base64:  # 如果是 CSV，额外上传图片
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is the CSV rendered as an image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                ]
            }
        )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages
        )
        caption = response.choices[0].message.content.strip()
        return caption
    except Exception as e:
        print("Error generating dense caption:", e)
        return ""

def code_to_text(code_snippet: str, model: str = "gpt-4o") -> str:
    """
    Send the code snippet to GPT-4o and return an English summary.
    If `model` is not provided, defaults to "gpt-4o".
    """
    print("code")
    system_prompt = (
        "You are a concise programming assistant. "
        "Summarize the provided code in one short paragraph. "
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": code_snippet}
            ],
            temperature=0.2,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # 可选：根据需求记录日志或重新抛出
        return f"[ERROR] Failed to summarize code: {e}"

def video_to_text(input_video_url,
                  model_name="Qwen/Qwen2.5-Omni-3B",
                  summary: bool = True):   # 新增开关，默认要 summary
    """
    使用 Qwen2.5-Omni-3B 对视频进行 dense caption；
    若 summary=True，再让模型把 dense caption 压缩成一句英文摘要并返回。
    """

    try:
        # ===== 1. 与原函数完全一致：生成 dense caption =====
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text",
                             "text": "You are a dense captioning assistant. Given a video, describe it in detail, including visual content, actions, objects and context."}],
            },
            {
                "role": "user",
                "content": [{"type": "video", "video": input_video_url}],
            },
        ]

        use_audio_in_video = False
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
        inputs = processor(text=text,
                           audio=audios,
                           images=images,
                           videos=videos,
                           return_tensors="pt",
                           padding=True,
                           use_audio_in_video=False)
        inputs = inputs.to(model.device).to(model.dtype)

        text_ids, _ = model.generate(**inputs, use_audio_in_video=use_audio_in_video)
        captions = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        dense_caption = captions[0] if captions else ""
        if not dense_caption:
            return ""

        # ===== 2. 可选：把 dense caption 再 summary 成一句英文 =====
        if not summary:
            return dense_caption

        summary_prompt = (
            "Summarize the following video description in one English sentence.\n\n"
            f"{dense_caption}"
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a concise video-caption summarizer."},
                {"role": "user", "content": summary_prompt}
            ]
        )

        summary_text = response.choices[0].message.content.strip()
        return summary_text

    except Exception as e:
        print("Error in video_to_text:", e)
        return ""

def audio_to_text(input_audio_url, model_name="Qwen/Qwen2.5-Omni-3B"):
    """
    使用 Qwen2.5-Omni-7B 对音频生成 dense caption（只输出文本）
    """
    
    try:
        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an audio dense captioning assistant. Given an audio clip, provide a highly detailed description, including speech content, speakers, tone, background sounds, and overall context."
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": input_audio_url},
                ],
            },
        ]

        # 构造输入
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False
        )
        inputs = inputs.to(model.device).to(model.dtype)

        # 推理生成 dense caption（只取 text_ids，忽略 audio）
        text_ids, _ = model.generate(**inputs, use_audio_in_video=False)
        captions = processor.batch_decode(
            text_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        return captions[0] if captions else ""
    except Exception as e:
        print("Error in audio_to_text:", e)
        return ""

def download_file(url: str, save_dir: str = "/tmp") -> str:
    """
    支持 GitHub/raw URL 下载文件，返回本地文件路径
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    save_path = os.path.join(save_dir, filename)

    if os.path.exists(save_path):
        # 已存在文件直接返回
        return save_path

    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return save_path
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}")

def threed_to_text(
    input_3d_url,
    model_name="gpt-4o",
    render_projection_image: bool = True,
    subdomain: str = None,
    content: str = None
):
    """
    生成三视图投影并使用 base64 编码上传给 GPT-4o 生成 dense caption
    """

    # --- 下载文件到本地（可选） ---
    if input_3d_url.startswith("http://") or input_3d_url.startswith("https://"):
        try:
            input_3d_url = download_file(input_3d_url, save_dir="/tmp/pointclouds")
        except Exception as e:
            print(f"[ERROR] 下载 3D 文件失败: {e}")
            return ""

    file_ext = Path(input_3d_url).suffix.lower()

    # --- 渲染三视图投影图像 ---
    if render_projection_image:
        try:
            if file_ext == ".ply":
                pcd = o3d.io.read_point_cloud(input_3d_url)
                points = np.asarray(pcd.points)
            elif file_ext == ".off":
                mesh = o3d.io.read_triangle_mesh(input_3d_url)
                mesh.compute_vertex_normals()
                points = np.asarray(mesh.vertices)
            else:
                print(f"[WARN] 不支持的文件类型: {file_ext}")
                return ""

            if points.shape[0] == 0:
                print("[WARN] 点云/网格为空")
                return ""

            # 三视图投影
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].scatter(points[:, 0], points[:, 1], s=1, c=points[:, 2], cmap="viridis")
            axs[0].set_title("XY Projection"); axs[0].axis("equal"); axs[0].axis("off")
            axs[1].scatter(points[:, 0], points[:, 2], s=1, c=points[:, 1], cmap="viridis")
            axs[1].set_title("XZ Projection"); axs[1].axis("equal"); axs[1].axis("off")
            axs[2].scatter(points[:, 1], points[:, 2], s=1, c=points[:, 0], cmap="viridis")
            axs[2].set_title("YZ Projection"); axs[2].axis("equal"); axs[2].axis("off")
            plt.tight_layout()

            # 保存到内存 buffer
            buf = BytesIO()
            plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close()
            buf.seek(0)
            print("[INFO] 三视图投影已生成到内存")

        except Exception as e:
            print(f"[ERROR] 渲染投影失败: {e}")
            return ""
    else:
        print("[WARN] 未渲染投影图像，无法生成 caption")
        return ""

    # --- 将图片转为 base64 ---
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    print("[INFO] 图片已编码为 base64")

    # --- 调用 GPT-4o 生成 dense caption ---
    try:
        user_prompt = f"""
You are a visual captioning assistant.
Given the image below (a three-view projection of a 3D object), generate a **highly detailed and dense caption**.

- The object type is: {subdomain}.
- Use the following extra context if useful: {content}.
- Do not describe raw points or dots. Instead, infer the real-world object this point cloud represents.
- Write a coherent paragraph that clearly conveys the object's identity, shape, components, spatial structure, and possible function.
- Return only the caption text; no extra explanation or markup.
        """

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a strict visual captioning assistant."},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                ]}
            ],
            max_tokens=500
        )

        caption = response.choices[0].message.content.strip()
        return caption

    except Exception as e:
        print(f"[ERROR] GPT-4o 生成 dense caption 失败: {e}")
        return ""

def replace_placeholders(text, code_map=None, fn_map=None, model_map=None, modal_map=None):
    """
    替换文本中的占位符 <codeX>, <imageX>, <documentX>
    替换后格式为 <tag: dense caption>
    """
    if code_map is None:
        code_map = {}
    if modal_map is None:
        modal_map = {}

    def replacer(match):
        tag = match.group(0).strip("<> \n")
        print("Processing tag:", tag)

        # code 模态
        if tag in code_map:
            return f"<{tag}:{code_map[tag]}>"

        # 所有 modal_map 模态
        if tag in modal_map:
            path_or_url = modal_map[tag]
            # 如果有 fn_map 对应的处理函数，则调用
            prefix = re.match(r"[a-zA-Z]+", tag)
            if prefix and fn_map and prefix.group(0) in fn_map:
                fn = fn_map[prefix.group(0)]
                model_name = model_map.get(prefix.group(0)) if model_map else None
                caption = fn(path_or_url, model_name)
                return f"<{tag}:{caption}>"
            # 否则直接返回 modal_map 对应内容
            return f"<{tag}:{path_or_url}>"

        # 默认不替换
        return match.group(0)

    return re.sub(r"<(code|image|document|audio|video|threeD)\d+>", replacer, text)

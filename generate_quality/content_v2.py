#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_text_quality.py
------------------------------------------------
为每条数据的 output.content 评估文本质量（1–5分），写入 scores.text。

特点：
- 模型固定使用 gpt-5-mini
- output.content 为空直接判 1 分（不调用模型）
- 输入是多个 JSON 对象拼接（非数组）
- 输出为多个 JSON 对象，indent=2 展开（对象之间空一行）
- 结果写入：obj["scores"]["text"] = score
------------------------------------------------
用法：
    python evaluate_text_quality.py \
        --input /path/to/data.json \
        --output /path/to/data.scored.json
"""

import json
import argparse
import openai
from tqdm import tqdm

# ========== 评分用 Prompt（你的定稿版） ==========
SYSTEM_PROMPT = """You are a strict evaluator of text quality. 
Your task is to assign a single integer score from 1 to 5 for a model-generated answer, 
based solely on its textual quality — not its factual correctness or relevance.

Judge the text holistically along four aspects:
1. Information completeness and richness
2. Logical clarity and structure
3. Linguistic fluency and readability
4. Language consistency (no unjustified mixing of languages)

[Scoring Rubric]

Score 5:
1. Content is rich, self-consistent, and detailed; no major omissions or vague generalities; stands alone as a coherent text.
2. Structure is clear and well-organized; transitions are smooth (e.g., “firstly…then…therefore…”); reasoning shows causal or hierarchical logic.
3. Language is natural and grammatically flawless; diverse sentence structures; no spelling or syntactic errors.
4. Entirely in one language; any foreign words appear only as necessary terminology.

Score 4:
1. Content is generally complete with sufficient detail but slightly shallow or missing minor points.
2. Structure is good, with only mild jumps or awkward transitions that don’t affect comprehension.
3. Language is fluent with few minor grammatical or collocation issues.
4. Mostly consistent language, with rare short foreign terms that do not disrupt flow.

Score 3:
1. Content covers the main idea but lacks depth or specific details.
2. Organization somewhat weak; order or topic shifts slightly; meaning still clear overall.
3. Several grammatical or spelling mistakes; simple or repetitive sentence patterns.
4. Minor language switching between sentences or paragraphs, noticeable but not confusing.

Score 2:
1. Content is shallow, missing key details or explanations; very low information density.
2. Poor structure; sentences disjointed; reader must infer connections.
3. Frequent grammar errors; awkward or broken phrasing; readability low.
4. Frequent in-sentence language mixing that affects readability.

Score 1:
1. Content is empty or meaningless; repetitive or irrelevant phrases; conveys no clear information.
2. No logical order; severe contradictions; text barely comprehensible.
3. Major grammatical breakdowns; unnatural or non-human syntax.
4. Chaotic multilingual mixing (e.g., Chinese + English + Spanish, random spelling noise) making it unreadable.

[Few-shot Examples]

Score 5:
The bloodhound stands as a legendary figure among scent-tracking dogs, combining anatomical precision with cinematic beauty. <image2>
Its olfactory bulb is proportionally forty times larger than that of humans, enabling it to distinguish a single human scent among thousands. <image3>
Historically, bloodhounds have been employed in everything from medieval manhunts to modern forensic investigations, revered for their near-mythic perseverance. <image4>
Their legacy embodies both science and devotion: the perfect harmony between instinct and discipline.

Score 4:
The bloodhound is renowned for having the best sense of smell among all dog breeds. <image2>
These dogs are equipped with an exceptionally large olfactory bulb, allowing them to detect scents over great distances. <image3>
They have long been used for search and rescue due to their remarkable scent-detection ability. <image4>

Score 3:
The bloodhound is a kind of dog famous for its strong smelling power. <image2>
It can find things or people by smell because its nose is very good. <image3>
People use bloodhounds to help look for missing people or to follow trails in forests. <image4>

Score 2:
Bloodhound smell very strong, it can 找到 people in forest quickly sometimes. <image2>
dog nose grass picture, 光线 warm.
Nose big so many police 使用 it for track, but sentences not good order. <image3>

Score 1:
狗 nose wow bloodhound smellings muy bueno?? <image2>
imageno 混乱 blur 光线 hard find 人 person trackerz olor sniff haha ###, 英语中文español一起转来转去—读不懂.

[Output Format]
Return only valid JSON:
{
  "score": 1-5,
  "reason": "Briefly explain why this score was assigned."
}
"""

# ========== 解析多对象 JSON 文件 ==========
def load_multi_json(path: str):
    """解析多个 JSON 对象拼接的文件（非数组）。"""
    objs, buf, brace_count = [], "", 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # 允许空行
            if not line.strip():
                continue
            brace_count += line.count("{") - line.count("}")
            buf += line
            if brace_count == 0 and buf.strip():
                try:
                    objs.append(json.loads(buf))
                except Exception as e:
                    print(f"[WARN] 无法解析对象：{e}\n片段：{buf[:200]}...")
                buf = ""
    # 处理文件尾未闭合的情况
    if buf.strip():
        try:
            objs.append(json.loads(buf))
        except Exception as e:
            print(f"[WARN] 尾部对象解析失败：{e}\n片段：{buf[:200]}...")
    return objs

# ========== 写出（indent=2，多对象，空行分隔） ==========
def write_multi_json(path: str, objs):
    with open(path, "w", encoding="utf-8") as f:
        for i, obj in enumerate(objs):
            json.dump(obj, f, ensure_ascii=False, indent=2)
            if i != len(objs) - 1:
                f.write("\n\n")

# ========== 调用 GPT-5-mini 评分 ==========
def evaluate_text_quality(content: str, client) -> int:
    """调用 gpt-5-mini 对文本质量打分（1–5）。空内容直接返回 1。"""
    if not content or not content.strip():
        return 1  # 空内容直接 1 分

    try:
        completion = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Evaluate the following model answer using the above rubric and few-shot references.\n\n"
                        "[Model Answer]\n" + content
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )
        resp = completion.choices[0].message.content
        data = json.loads(resp)
        score = int(data.get("score", 3))
        return max(1, min(score, 5))
    except Exception as e:
        print(f"[ERROR] 模型评估失败：{e}")
        # 失败时给一个中间分，避免中断整批处理
        return 3

# ========== 主流程 ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入文件路径（多个 JSON 对象拼接）")
    parser.add_argument("--output", required=True, help="输出文件路径（多个 JSON 对象、indent=2 展开）")
    args = parser.parse_args()

    # 使用环境变量中的 OPENAI_API_KEY
    client = openai.OpenAI()

    objs = load_multi_json(args.input)
    print(f"共解析 {len(objs)} 条对象，开始评估...")

    for obj in tqdm(objs):
        # 读取 content
        content = ""
        out = obj.get("output")
        if isinstance(out, dict):
            content = out.get("content", "")

        # 评分
        score = evaluate_text_quality(content, client)

        # 写入 scores.text
        scores = obj.get("scores")
        if not isinstance(scores, dict):
            scores = {}
        scores["text"] = score
        obj["scores"] = scores  # 回写

    write_multi_json(args.output, objs)
    print(f"✅ 完成：已写出到 {args.output}")

if __name__ == "__main__":
    main()

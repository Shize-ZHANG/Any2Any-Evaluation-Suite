# 🧠 Caption Evaluation Pipeline

## Overview
This script `caption_eval_pipeline.py` processes multimodal model response JSONL files:
- Converts all `<imageX>`, `<audioY>`, `<videoZ>` ... tags into detailed captions.
- Matches each sample with its ground-truth JSONL (which already contains textual captions).
- Evaluates **semantic correctness** of the generated text using **GPT-4o**, producing a 5-level score in `{0.2, 0.4, 0.6, 0.8, 1.0}`.
- Outputs a new JSONL file with `captioned_response` and `score` added.

---

## ⚙️ Environment Setup

### 1️⃣ Create Conda environment
```bash
conda env create -f environment.yml
conda activate caption-eval
```

### 2️⃣ Export your OpenAI API key
```
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxx"
```

### 3️⃣ (If video or audio) Launch your local vLLM server for Qwen2.5-Omni
```
vllm serve model/path
--dtype auto   
--host localhost   
--trust-remote-code   
--port 8003
--gpu-memory-utilization 0.7
```

### 4️⃣ move resources files to the same path with input response file
Expected structure under your project root:
```
project_root/
├── caption_eval_pipeline.py
├── environment.yml
├── models/
│   └── PointLLM_7B_v1.2/             # your PointLLM model weights
├── dataset/
│   ├── response.jsonl                # model-generated outputs
│   ├── ground_truth.jsonl            # ground-truth with captions
│   ├── image/
│   │   ├── img_0001_01.jpg
│   │   └── img_0002_01.jpg
│   ├── audio/
│   │   ├── aud_0001_01.mp3
│   │   └── aud_0002_01.mp3
│   ├── video/
│   │   └── vid_0001_01.mp4
│   ├── document/
│   │   └── doc_0001_01.pdf
│   └── threeD/
│       └── model_0001_01.ply
```

### Run the script:
```
python caption_eval_pipeline.py \
  --response-path path.jsonl \
  --ground-truth-path path.jsonl \
  (optional)
  --pointllm-model-path models/PointLLM_7B_v1.2 \
  --vllm-endpoint http://127.0.0.1:8003/v1
```

## Output Format Example:
```
{
  "domain": "natural_science",
  "subdomain": "math",
  "id": "1",
  "output": {
    "modal": {
      "audio2": "audio/aud_0001_02.mp3"
    },
    "content": "The missing shape number is <audio2>.",
    "captioned_response": "The missing shape number is <audio2: The audio explains that the answer is number 3.>",
    "score": 0.8
  }
}
```
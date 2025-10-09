# Multi-Modal Quality Evaluation Tool

A unified evaluation script for assessing quality across 7 different modalities: image, audio, video, text, code, document, and 3D models.

## Overview

This tool processes JSONL files containing multi-modal data and outputs quality scores (0-100) for each modality resource.

### Supported Modalities

| Modality | Method | Score Range | Dependencies |
|----------|--------|-------------|--------------|
| **Image** | BRISQUE/NIQE | 0-100 | OpenCV, NumPy, SciPy |
| **Audio** | Statistical analysis (SNR, LUFS, etc.) | 0-100 | librosa, pyloudnorm |
| **Video** | DOVER model (TQE+AQE) | 0-100 | PyTorch, DOVER model |
| **3D** | Topology/Geometry/Sampling | 0-100 | Open3D, trimesh |
| **Document** | OCR + GPT-4o evaluation | 0-100 | Tesseract, OpenAI API |
| **Text** | GPT-4o multi-dimension | 0-100 | OpenAI API |
| **Code** | GPT-4o code review | 0-100 | OpenAI API |

## Installation

### 1. Basic Dependencies

```bash
pip install numpy scipy opencv-python librosa pyloudnorm open3d trimesh Pillow pytesseract openai requests
```

### 2. System Dependencies

**Tesseract OCR** (for document evaluation):
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
```

### 3. Optional: DOVER Model (for video evaluation)

If you need video quality assessment:

```bash
# Clone DOVER repository
cd /home/wang/Downloads/db_agent_bench/eval/
git clone https://github.com/VQAssessment/DOVER.git

# Install DOVER dependencies
cd DOVER
pip install -r requirements.txt

# Download pretrained weights
# Follow DOVER documentation: https://github.com/VQAssessment/DOVER
```

### 4. OpenAI API Key (for text/code/document evaluation)

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or the script will use the hardcoded fallback key (not recommended for production).

## Usage

### Basic Command

```bash
python unified_eval.py --input input.jsonl --output output.jsonl
```

### Arguments

- `--input`: Path to input JSONL file (required)
- `--output`: Path to output JSONL file (optional, defaults to `input_name_output.jsonl`)

### Example

```bash
# Use default output path
python unified_eval.py --input data.jsonl

# Specify custom output path
python unified_eval.py --input data/samples.jsonl --output results/scores.jsonl
```

## Input Format

The input JSONL file should contain JSON objects in multi-line expanded format:

```json
{
  "domain": "general_domain",
  "subdomain": "architecture",
  "id": "1",
  "input": {
    "modal": {
      "image1": "image/img_0001_01.jpg",
      "audio1": "audio/aud_0001_01.mp3"
    },
    "content": "Description of the input..."
  },
  "output": {
    "modal": {
      "image2": "image/img_0001_02.png",
      "audio2": "audio/aud_0001_02.mp3",
      "text1": "This is a text answer...",
      "code1": "def example(): pass"
    },
    "content": "Description of the output..."
  },
  "difficulty_level": 3
}
```

### Key Points:

1. **File paths** in `output.modal` are relative to the JSONL file's directory
2. **Text and code** modalities contain the actual content (not file paths)
3. The script only evaluates resources in `output.modal` (not `input.modal`)

### Modal Type Detection

The script automatically detects modal type from key prefixes:

- `image*` → Image evaluation
- `audio*` → Audio evaluation
- `video*` → Video evaluation
- `threeD*` → 3D model evaluation
- `document*` → Document evaluation
- `text*` → Text evaluation
- `code*` → Code evaluation

## Output Format

The output JSONL file contains the original data plus a new `scores` field:

```json
{
  "domain": "general_domain",
  "subdomain": "architecture",
  "id": "1",
  "output": {
    "modal": {
      "image2": "image/img_0001_02.png",
      "audio2": "audio/aud_0001_02.mp3"
    },
    "content": "..."
  },
  "scores": {
    "image2": 85.5,
    "audio2": 78.3
  },
  "difficulty_level": 3
}
```

### Score Interpretation

- **0-100**: Higher scores indicate better quality
- **null**: Evaluation failed (check logs for details)

## Error Handling

The script handles errors gracefully:

- **Missing files**: Logs error and sets score to `null`
- **Unsupported formats**: Skips resource and logs warning
- **API failures**: Sets score to `null` and continues
- **Missing dependencies**: Logs warning and skips that modality

Check the console output for detailed logs during execution.

## Examples

### Example 1: Evaluate Image and Audio

```bash
# Input: data.jsonl
{
  "id": "1",
  "output": {
    "modal": {
      "image1": "samples/photo.jpg",
      "audio1": "samples/music.mp3"
    }
  }
}

# Run evaluation
python unified_eval.py --input data.jsonl

# Output: data_output.jsonl
{
  "id": "1",
  "output": { ... },
  "scores": {
    "image1": 82.5,
    "audio1": 76.8
  }
}
```

### Example 2: Evaluate Text and Code

```bash
# Input: text_code.jsonl
{
  "id": "2",
  "output": {
    "modal": {
      "text1": "This is a well-written explanation of quantum physics...",
      "code1": "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    ..."
    }
  }
}

# Run evaluation (requires OpenAI API key)
export OPENAI_API_KEY="your-key"
python unified_eval.py --input text_code.jsonl

# Output: text_code_output.jsonl
{
  "id": "2",
  "output": { ... },
  "scores": {
    "text1": 88.2,
    "code1": 91.5
  }
}
```

## Troubleshooting

### Issue: "OpenAI client not available"

**Solution**: Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key"
```

### Issue: "DOVER directory not found"

**Solution**: Video evaluation is optional. If you don't need it, the script will skip video files. To enable:
```bash
cd /path/to/eval/
git clone https://github.com/VQAssessment/DOVER.git
# Follow DOVER setup instructions
```

### Issue: "Tesseract not found"

**Solution**: Install Tesseract OCR:
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract
```

### Issue: "File not found: image/xxx.jpg"

**Solution**: Ensure file paths in JSONL are relative to the JSONL file's directory:
```
project/
├── data.jsonl
└── image/
    └── xxx.jpg
```

## Performance Notes

- **Image/Audio/3D**: Fast (pure algorithmic evaluation)
- **Text/Code/Document**: Slower (requires API calls to GPT-4o)
- **Video**: Slowest (requires deep learning model inference)

For large datasets, consider:
- Batching API requests
- Using GPU for video evaluation
- Running in parallel for independent files

## Advanced Usage

### Custom API Endpoint

Modify the `get_openai_client()` function in `unified_eval.py` to use custom endpoints:

```python
def get_openai_client():
    from openai import OpenAI
    return OpenAI(
        api_key=os.environ.get('OPENAI_API_KEY'),
        base_url="https://your-custom-endpoint.com/v1"
    )
```

### Adjusting Evaluation Parameters

Each evaluation function can be customized:

- **Image**: Modify `evaluate_image()` to adjust BRISQUE/NIQE weights
- **Audio**: Adjust thresholds in `audio.py` for SNR, LUFS ranges
- **Text/Code**: Modify GPT prompts in `evaluate_text()`/`evaluate_code()`

## File Structure

```
generate_quality/
├── unified_eval.py          # Main script
├── README.md                # This file
├── audio.py                 # Audio evaluation module
├── code.py                  # Code evaluation module
├── document.py              # Document evaluation module
├── image.py                 # Image evaluation module
├── text.py                  # Text evaluation module
├── threeD.py                # 3D model evaluation module
├── video.py                 # Video evaluation module
├── input.jsonl              # Example input
└── output_example.jsonl     # Example output
```

## License

See project root for license information.

## Contributing

For bug reports and feature requests, please open an issue in the project repository.

## Citation

If you use this tool in your research, please cite the original papers for each evaluation method:

- **BRISQUE**: Mittal et al., "No-Reference Image Quality Assessment in the Spatial Domain"
- **DOVER**: Wu et al., "Exploring Video Quality Assessment on User Generated Contents from Aesthetic and Technical Perspectives"
- **Audio Quality**: Custom statistical analysis based on ITU-R BS.1387

## Contact

For questions and support, please refer to the project documentation or open an issue.

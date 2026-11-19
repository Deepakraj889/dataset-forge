# ⚡ DatasetForge

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-brightgreen.svg)](https://www.python.org/)
[![Model Dialect](https://img.shields.io/badge/Model%20Dialect-Illustrious%20(SDXL)-orange.svg)](https://civitai.com/)
[![CI](https://github.com/Deepakraj889/dataset-forge/actions/workflows/ci.yml/badge.svg)](https://github.com/Deepakraj889/dataset-forge/actions)

> **High-performance dataset curation, smart aspect-ratio bucketing, resolution-adaptive watermark removal, and vision auto-tagging for SDXL & Illustrious diffusion training.**

---

## 🎯 Overview

Curating high-quality datasets for **Illustrious (SDXL)** and LoRA fine-tuning is traditionally painful and error-prone:
* Arbitrary resizing distorts aspect ratios or introduces latent artifacts.
* Naive center-crops slice off heads, hair, or essential character accessories.
* Artist signatures, timestamps, and watermarks pollute training weights.
* Missing or unformatted captions require bouncing across multiple separate tools.
* Near-duplicate images (burst shots, slight angle shifts) trigger severe model overfitting.

**DatasetForge** solves this with an end-to-end, multi-core automated pipeline engineered specifically for the 1024px latent geometry and tag dialect of **Illustrious (SDXL)**.

```
Raw Images (Un-captioned & Watermarked)
                   │
                   ▼
       ┌───────────────────────┐
       │ Perceptual Deduplication │ ──► pHash Hamming screening (prunes burst shots)
       └───────────┬───────────┘
                   │
                   ▼
       ┌───────────────────────┐
       │ Watermark Inpainting  │ ──► Resolution-adaptive Telea/Navier-Stokes inpainting
       └───────────┬───────────┘
                   │
                   ▼
       ┌───────────────────────┐
       │ Smart Aspect Cropping │ ──► Face & subject-aware focal cropping into SDXL buckets
       └───────────┬───────────┘
                   │
                   ▼
       ┌───────────────────────┐
       │   WD14 Vision Tagger  │ ──► On-demand ConvNeXt auto-captioning + Illustrious ordering
       └───────────┬───────────┘
                   │
                   ▼
       ┌───────────────────────┐
       │ Standard LoRA Export  │ ──► Ready-to-train {repeats}_{concept}/ + metadata.jsonl
       └───────────────────────┘
```

---

## ✨ Features

- **🚀 Multi-Core CPU Parallelism**: Dynamically scales across all available hardware threads (e.g. 8, 12, 24, 64 cores) using process pools to churn through 4K datasets in seconds.
- **🏷️ Automated WD14 Vision Tagger**: Reads raw image pixels using the gold-standard ConvNeXt WD14 model to generate Danbooru tags. Models are downloaded on-demand into local cache, keeping the repository light and zero-bloat.
- **🩹 Resolution-Adaptive Watermark Inpainter**: Automatically detects and seamlessly inpaints artist signatures, timestamps, and logos, and trims solid letterbox bars.
- **🎯 Exact SDXL 1024 Bucketing**: Pre-configured with the standard SDXL / Illustrious aspect ratio buckets (1:1, 9:7, 7:9, 3:2, 2:3, 16:9, 9:16, 21:9) strictly divisible by 64.
- **👁️ Face & Subject-Aware Smart Crop**: Automatically identifies face bounding boxes to ensure portraits preserve full head, hair, and torso composition during cropping.
- **🔍 Perceptual Deduplication (`pHash`)**: Computes 64-bit perceptual hashes to detect identical images, crop duplicates, and burst shots using customizable Hamming distance thresholds.
- **🏷️ Illustrious Tag Engineering**:
  - Automatically converts Danbooru underscores to natural spaces (`black_hair` ➔ `black hair`).
  - Guarantees concept trigger words are placed first (`leafa, 1girl, ...`).
  - Automatically appends Illustrious aesthetic tags (`masterpiece, best quality, very aesthetic, absurdres`).
  - Strips watermarks, artist signatures, and low-quality prompt noise.
- **📦 One-Command Standard Export**: Organizes prepared datasets into canonical `{repeats}_{concept}/` folders alongside standard `metadata.jsonl` (compatible with Kohya-ss, SimpleTuner, and other LoRA trainers).

---

## 📦 Installation

```bash
git clone https://github.com/Deepakraj889/dataset-forge.git
cd dataset-forge
pip install -e .
```

---

## ⚡ Quickstart

### 1. Run the Full End-to-End Pipeline
Takes raw, un-captioned, watermarked images and produces a ready-to-train LoRA dataset in one command:
```bash
dataset-forge prepare "path/to/raw_images" --output "./training_dir" --concept leafa --repeats 10 --trigger "leafa" --watermark-clean
```

### 2. Auto-Tag Images with WD14 Vision Tagger
Automatically generate Danbooru captions from raw pixels:
```bash
dataset-forge autotag "path/to/images" --trigger "leafa" --threshold 0.35
```

### 3. Clean Watermarks & Signatures
Inpaint corner signatures, logos, or banner strips:
```bash
dataset-forge clean-watermarks "path/to/images" --output "./cleaned_images" --region corners
```

### 4. Smart-Crop to SDXL Buckets
Crop and resize images to their ideal SDXL 1024 aspect ratio bucket with face-aware centering:
```bash
dataset-forge crop "path/to/images" --output "./cropped_dataset" --face-detect
```

### 5. Scan & Quarantine Near-Duplicates
Find near-duplicate images and optionally isolate them:
```bash
dataset-forge dedupe "path/to/images" --threshold 5 --quarantine "./duplicates_bin"
```

---

## 📐 Supported SDXL Buckets (1024 Base)

| Aspect Ratio | Resolution | Category | Use Case |
| :--- | :--- | :--- | :--- |
| **1:1** | `1024 x 1024` | Square | Avatars, standard portraits |
| **9:7** | `1152 x 896` | Landscape | Standard wide shots |
| **7:9** | `896 x 1152` | Portrait | Full / 3/4 character poses |
| **3:2** | `1216 x 832` | Landscape | Scenic landscapes |
| **2:3** | `832 x 1216` | Portrait | Full body standing |
| **16:9** | `1344 x 768` | Wide | Cinematic banners |
| **9:16** | `768 x 1344` | Tall | Full vertical splash art |
| **21:9** | `1536 x 640` | Ultra-Wide | Panoramic / multi-subject |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 👤 Author

**Deepak Raj**
* GitHub: [@Deepakraj889](https://github.com/Deepakraj889)
* Email: deepakraj247200@gmail.com

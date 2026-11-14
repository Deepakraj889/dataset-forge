# ⚡ DatasetForge

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-brightgreen.svg)](https://www.python.org/)
[![Model Dialect](https://img.shields.io/badge/Model%20Dialect-Illustrious%20(SDXL)-orange.svg)](https://civitai.com/)
[![CI](https://github.com/Deepakraj889/dataset-forge/actions/workflows/ci.yml/badge.svg)](https://github.com/Deepakraj889/dataset-forge/actions)

> **High-performance dataset curation, smart aspect-ratio bucketing, and Danbooru tag engineering for SDXL & Illustrious diffusion training.**

---

## 🎯 Overview

Curating high-quality datasets for **Illustrious (SDXL)** and LoRA fine-tuning is traditionally painful and error-prone:
* Arbitrary resizing distorts aspect ratios or introduces latent artifacts.
* Naive center-crops slice off heads, hair, or essential character accessories.
* Near-duplicate images (burst shots, slight angle shifts) trigger severe model overfitting.
* Raw Danbooru tags contain noisy watermarks, unescaped syntax, and inconsistent tag hierarchies.

**DatasetForge** solves this with an end-to-end, automated pipeline engineered specifically for the 1024px latent geometry and tag dialect of **Illustrious (SDXL)**.

```
Raw Images + Captions
         │
         ▼
 ┌───────────────┐
 │ Deduplication │ ──► Perceptual pHash (prunes burst & identical shots)
 └───────┬───────┘
         │
         ▼
 ┌───────────────┐
 │ Quality Check │ ──► Laplacian variance blur filter & dynamic range check
 └───────┬───────┘
         │
         ▼
 ┌───────────────┐
 │ Smart Cropping│ ──► Face & subject-aware focal cropping into SDXL buckets
 └───────┬───────┘
         │
         ▼
 ┌───────────────┐
 │ Tag Engineer  │ ──► Danbooru normalization, trigger injection & quality tags
 └───────┬───────┘
         │
         ▼
 ┌───────────────┐
 │ Kohya Export  │ ──► Ready-to-train {repeats}_{concept}/ + metadata.jsonl
 └───────────────┘
```

---

## ✨ Features

- **🎯 Exact SDXL 1024 Bucketing**: Pre-configured with the standard SDXL / Illustrious aspect ratio buckets (1:1, 9:7, 7:9, 3:2, 2:3, 16:9, 9:16, 21:9) strictly divisible by 64 to prevent VAE shape mismatch errors.
- **👁️ Face & Subject-Aware Smart Crop**: Automatically identifies face and anatomical focus points to ensure portraits preserve full head, hair, and torso composition during cropping.
- **🔍 Perceptual Deduplication (`pHash`)**: Computes 64-bit perceptual hashes to detect identical images, crop duplicates, and burst shots using customizable Hamming distance thresholds.
- **🔬 Blur & Artifact Screening**: Uses OpenCV Laplacian variance scoring to flag low-resolution or blurry source images before they pollute training weights.
- **🏷️ Illustrious Tag Engineering**:
  - Automatically converts Danbooru underscores to natural spaces (`black_hair` ➔ `black hair`).
  - Cleans escaped syntax (`\(medium\)` ➔ `(medium)`).
  - Pre-pends unique concept trigger words and character identifiers.
  - Automatically appends Illustrious aesthetic tags (`masterpiece, best quality, very aesthetic, absurdres`).
  - Strips watermarks, artist signatures, and low-quality prompt noise.
- **🚀 One-Command Kohya Export**: Organizes prepared datasets into the canonical `10_concept/` folder convention alongside Hugging Face / Kohya `metadata.jsonl`.

---

## 📦 Installation

### From Source
```bash
git clone https://github.com/Deepakraj889/dataset-forge.git
cd dataset-forge
pip install -e .
```

---

## ⚡ Quickstart

### 1. Run the Full End-to-End Pipeline
Prepare a raw folder of images and text files into a ready-to-train Kohya LoRA dataset in one command:
```bash
dataset-forge prepare path/to/raw_images --output ./training_dir --concept makima --repeats 10 --trigger "makima_csm"
```

### 2. Analyze Dataset Quality & Blur
Inspect resolutions, aspect ratios, and blur metrics across your dataset:
```bash
dataset-forge analyze path/to/dataset --dialect illustrious
```

### 3. Scan & Quarantine Near-Duplicates
Find near-duplicate images with Hamming distance $\le 5$ and optionally isolate them:
```bash
dataset-forge dedupe path/to/dataset --threshold 5 --quarantine ./duplicates_bin
```

### 4. Smart-Crop to SDXL Buckets
Crop and resize images to their ideal SDXL 1024 aspect ratio bucket with face-aware centering:
```bash
dataset-forge crop path/to/dataset --output ./cropped_dataset --face-detect
```

### 5. Normalize & Clean Danbooru Tags
Batch process all `.txt` caption files in your dataset:
```bash
dataset-forge tag path/to/dataset --trigger "hinata" --character "hyuuga hinata" --quality
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

## 🗺️ Roadmap

- [x] **v0.1.0**: Illustrious (SDXL) dialect profile, smart bucketing, face-aware crop, pHash deduplication, Kohya export.
- [ ] **v0.2.0**: **Pony Diffusion** dialect profile (`score_9, score_8_up...` score tag ranking).
- [ ] **v0.3.0**: **Flux.1** natural language caption generator & resolution bucketing.
- [ ] **v0.4.0**: Interactive Web UI (FastAPI preview dashboard) for one-click manual approval of borderline crops.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 👤 Author

**Deepak Raj**
* GitHub: [@Deepakraj889](https://github.com/Deepakraj889)
* Email: deepakraj247200@gmail.com

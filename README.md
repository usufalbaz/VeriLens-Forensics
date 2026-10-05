# 🔬 VeriLens Forensics
> Production-Grade Multi-Engine Forensic Framework for Detecting Synthetic Media, Facial Deepfakes, and Digital Tampering.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Architecture-Hybrid%20Forensics-red.svg)]()
[![Status](https://img.shields.io/badge/Release-v1.0--Alpha-orange.svg)]()

---

## 📌 Executive Summary
Standard deepfake and AI-generated image detectors frequently fail when confronted with compression artifacts, modern diffusion architectures (Midjourney, Flux, Stable Diffusion), or localized Photoshop splicing.

**VeriLens Forensics** addresses this vulnerability by deploying a **Tri-Pillar Hybrid Forensic Architecture**:
1. **Frequency-Domain Spectral Profiling (2D-FFT):** Identifies anomalous high-frequency grid residue and unnatural intra-band variance introduced by generative upsampling layers.
2. **Error Level Analysis (ELA):** Localizes spatial tampering, copy-move operations, and compression discrepancies across image planes.
3. **Deep Vision Backbone:** Evaluates semantic facial coherence, boundary blending, and microscopic visual artifacts.

---

## 🧠 System Architecture & Mathematical Foundations

### 1. 2D Fast Fourier Transform (FFT) Decomposition
Input imagery $f(x, y)$ of dimensions $M \times N$ is mapped to the spatial frequency domain:

$$F(u, v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x, y) \cdot e^{-j 2\pi \left(\frac{ux}{M} + \frac{vy}{N}\right)}$$

The centered magnitude spectrum in decibels is extracted via:

$$S(u, v) = 20 \cdot \log_{10}(|F_{\text{shift}}(u, v)| + 10^{-8})$$

### 2. Concentric Radial Band Extraction
Rather than evaluating global averages, the engine segments the spectrum into $K$ concentric rings of radial Euclidean distance $r = \sqrt{(x - c_x)^2 + (y - c_y)^2}$ to compute:
* **Intra-band Spectral Variance ($\sigma_r$):** Detects synthetic asymmetric noise.
* **High-Frequency Energy Ratio ($HF_{\text{ratio}}$):** Captures unnatural high-frequency energy dispersion characteristic of modern GAN/Diffusion upscaling.

---

## 📂 Repository Layout

```text
├── core/
│   ├── frequency.py      # 2D-FFT Spectral Engine & Radial Decomposition
│   ├── forensics.py      # Error Level Analysis (ELA) & Tampering Masking [Next]
│   └── metadata.py       # EXIF & Generative Software Fingerprinting [Upcoming]
├── models/               # Serialized Model Weights & ONNX Artifacts
├── api/                  # High-Performance FastAPI Inference Service
├── frontend/             # Next.js / Tailwind CSS Web Application
└── requirements.txt      # Core Dependencies
```

---

## 🗺️ Project Roadmap & Milestones

- [x] **Phase 1: Frequency Forensics Engine**
  - [x] Dynamic spatial scaling & 2D-FFT extraction.
  - [x] Azimuthal radial integration & high-frequency ratio benchmarking.
- [ ] **Phase 2: Spatial & Compression Forensics**
  - [ ] Error Level Analysis (ELA) with dynamic scale quantization.
  - [ ] Forensic heatmap mask generation for spatial tampering localization.
- [ ] **Phase 3: Metadata & Provenance Engine**
  - [ ] EXIF tag validation & AI tool signature parsing (C2PA / Generative chunks).
- [ ] **Phase 4: Unified Classifier Training**
  - [ ] Benchmark training on 140k real & synthetic facial samples.
  - [ ] Export to quantized ONNX for ultra-low latency inference.
- [ ] **Phase 5: Production Deployment**
  - [ ] Containerized FastAPI backend microservice.
  - [ ] Interactive Next.js web application deployed to Vercel.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

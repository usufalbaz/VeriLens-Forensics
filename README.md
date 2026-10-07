# 🔬 VeriLens Forensics
> Production-Grade Multi-Engine Forensic Framework for Detecting Synthetic Media, Facial Deepfakes, and Digital Tampering.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Architecture-Hybrid%20Forensics-red.svg)]()
[![Status](https://img.shields.io/badge/Release-v1.0--Production-emerald.svg)]()
[![CI Pipeline](https://github.com/usufalbaz/VeriLens-Forensics/actions/workflows/ci.yml/badge.svg)](https://github.com/usufalbaz/VeriLens-Forensics/actions)

---

## 📌 Executive Summary
Standard deepfake and AI-generated image detectors frequently fail when confronted with compression artifacts, modern diffusion architectures (Midjourney, Flux, Stable Diffusion), or localized Photoshop splicing.

**VeriLens Forensics** addresses this vulnerability by deploying a **Multi-Engine Hybrid Forensic Architecture**:
1. **Frequency-Domain Spectral Profiling (2D-FFT):** Identifies anomalous high-frequency grid residue and unnatural intra-band variance introduced by generative upsampling layers.
2. **Error Level Analysis (ELA):** Localizes spatial tampering, copy-move operations, and compression discrepancies across image planes.
3. **Biometric Landmark & Corneal Reflection Verification:** Analyzes ocular specular reflections, pupil light consistency, and facial boundary blending gradients.
4. **Digital Provenance & Metadata Engine:** Validates camera sensor profiles, detects editing software tags (Photoshop, GIMP), and inspects C2PA / generative prompts.

---

## 🧠 System Architecture & Mathematical Foundations

### 1. 2D Fast Fourier Transform (FFT) Decomposition
Input imagery $f(x, y)$ of dimensions $M \times N$ is mapped to the spatial frequency domain:

$$F(u, v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x, y) \cdot e^{-j 2\pi \left(\frac{ux}{M} + \frac{vy}{N}\right)}$$

The centered magnitude spectrum in decibels is extracted via:

$$S(u, v) = 20 \cdot \log_{10}(|F_{\text{shift}}(u, v)| + 10^{-8})$$

### 2. Error Level Analysis (ELA) Quantization
Analyzes the absolute reconstruction difference against a predetermined compression baseline ($Q = 90$):

$$\Delta(x, y) = |I_{\text{original}}(x, y) - I_{\text{recompressed}}(x, y)|$$

Extrema amplification normalizes localized compression anomalies, exposing non-uniform compression grids indicative of spatial tampering.

---

## 📂 Repository Layout

```text
├── core/
│   ├── frequency.py      # 2D-FFT Spectral Engine & Radial Decomposition
│   ├── forensics.py      # Error Level Analysis (ELA) & Tampering Masking
│   ├── biometrics.py     # Facial Localization & Corneal Reflection Engine
│   ├── metadata.py       # EXIF & Generative Software Fingerprinting
│   └── pipeline.py       # Consolidated Multi-Engine Forensic Orchestrator
├── api/
│   └── index.py          # High-Performance Serverless Inference Endpoint
├── public/
│   └── index.html        # Enterprise Dark-Mode Forensic Dashboard
├── vercel.json           # Serverless Routing & Static Asset Configuration
├── requirements.txt      # Production Serverless Dependency Manifest
└── README.md             # Theoretical Architecture & Documentation
```

---

## 🗺️ Project Roadmap & Completed Milestones

- [x] **Phase 1: Frequency Forensics Engine**
  - [x] Dynamic spatial scaling & 2D-FFT extraction.
  - [x] Azimuthal radial integration & high-frequency ratio benchmarking.
- [x] **Phase 2: Spatial & Compression Forensics**
  - [x] Error Level Analysis (ELA) with dynamic scale quantization.
  - [x] Forensic heatmap mask generation for spatial tampering localization.
- [x] **Phase 3: Metadata & Provenance Engine**
  - [x] EXIF tag validation & AI tool signature parsing (C2PA / Generative chunks).
- [x] **Phase 4: Biometric & Multi-Engine Pipeline**
  - [x] Haar cascade facial bounding box localization & corneal lighting reflection verification.
  - [x] Unified forensic decision pipeline synthesizing composite risk scores.
- [x] **Phase 5: Cloud Deployment & User Interface**
  - [x] Serverless FastAPI backend microservice with Base64 telemetry streaming.
  - [x] Responsive cybersecurity dashboard deployed for public access.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

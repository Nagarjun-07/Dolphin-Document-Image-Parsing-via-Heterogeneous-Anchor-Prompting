<div align="center">
  <img src="./assets/dolphin.png" width="300">
</div>

<div align="center">
  <a href="https://arxiv.org/abs/2505.14059">
    <img src="https://img.shields.io/badge/Paper-arXiv-red">
  </a>
  <a href="https://huggingface.co/ByteDance/Dolphin">
    <img src="https://img.shields.io/badge/HuggingFace-Dolphin-yellow">
  </a>
  <a href="http://115.190.42.15:8888/dolphin/">
    <img src="https://img.shields.io/badge/Demo-Dolphin-blue">
  </a>
  <a href="https://github.com/bytedance/Dolphin">
    <img src="https://img.shields.io/badge/Code-GitHub-green">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-lightgrey">
  </a>
</div>

<br/>

<div align="center">
  <img src="./assets/demo.gif" width="800">
</div>

---

# 🐬 Dolphin: Document Image Parsing via Heterogeneous Anchor Prompting

**Dolphin** (**Do**cument Image Parsing via **H**eterogeneous Anchor Prompt**in**g) is a novel multimodal parsing framework introducing a two-stage, analyze-then-parse paradigm for accurate and efficient document understanding.

This repository includes demo code, pretrained models, and scripts for inference and evaluation.

---

## 📌 Table of Contents

- [🧠 Overview](#-overview)
- [🚀 Demo](#-demo)
- [📅 Changelog](#-changelog)
- [🛠️ Installation](#️-installation)
- [⚡ Inference](#-inference)
- [🌟 Key Features](#-key-features)
- [📮 Notice](#-notice)
- [💖 Acknowledgement](#-acknowledgement)
- [📝 Citation](#-citation)
- [⭐ Star History](#-star-history)

---

## 🧠 Overview

Document image parsing poses significant challenges due to the coexistence of various elements like **text**, **figures**, **tables**, and **formulas**. Dolphin addresses this with:

### ✅ Two-Stage Framework

1. **Stage 1**: Page-level layout analysis with element sequence in natural reading order.
2. **Stage 2**: Parallel parsing using **heterogeneous anchor prompting** and **task-specific prompts**.

<div align="center">
  <img src="./assets/framework.png" width="680">
</div>

Dolphin achieves **state-of-the-art accuracy**, supports **parallel decoding**, and integrates seamlessly with the **Hugging Face ecosystem**.

---
![WhatsApp Image 2025-06-07 at 01 17 15_ea78c99a](https://github.com/user-attachments/assets/9484ff87-a696-4c8f-8b39-391da60901be)

![WhatsApp Image 2025-06-07 at 01 04 23_50b9f647](https://github.com/user-attachments/assets/052d51c7-46af-45c7-9bac-ab8374301b54)
![WhatsApp Image 2025-06-07 at 01 19 16_687f1850](https://github.com/user-attachments/assets/b13ffe07-7809-4851-82d5-b029151b6975)

## 🚀 Demo

🖥️ Try Dolphin live: [**Click here for the demo**](http://115.190.42.15:8888/dolphin/)

---

## 📅 Changelog

- 🧠 **2025-05-21**: 🎉 Public Demo Released
- 🧠 **2025-05-20**: 🧪 Model & Inference Code Released
- 🧠 **2025-05-16**: 📄 Paper accepted at ACL 2025 [[arXiv](https://arxiv.org/abs/2505.14059)]

---

## 🛠️ Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/ByteDance/Dolphin.git
cd Dolphin
---

## Installation

Install the dependencies:
pip install -r requirements.txt

```
# Download the model from Hugging Face Hub
git lfs install
git clone https://huggingface.co/ByteDance/Dolphin ./hf_model

# Or use the Hugging Face CLI
huggingface-cli download ByteDance/Dolphin --local-dir ./hf_model

![image](https://github.com/user-attachments/assets/359d47f4-3220-4e55-a559-f2a11034edc1)

**
---

If you want, I can help you customize or add a table of contents and badges too!
**

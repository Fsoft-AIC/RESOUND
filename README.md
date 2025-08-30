<div align="center">
  
# 【Interspeech'2025 🎧】RESOUND: Speech Reconstruction from Silent Videos via Acoustic–Semantic Decomposed Modeling
  
[![Paper](https://img.shields.io/badge/Paper-arXiv:2505.22024v1-FF6B6B.svg)](https://arxiv.org/abs/2505.22024v1)
[![License](https://img.shields.io/badge/License-TBD-4D96FF.svg)](#-license)
</div>

Official repository for **RESOUND**, which reconstructs intelligible, expressive speech from **silent talking-face videos** via **acoustic–semantic decomposed modeling**.

---

## 📌 Citation
If you find this useful, please star 🌟 the repo and cite 📑:

```bibtex
@article{resound2025,
  title   = {RESOUND: Speech Reconstruction from Silent Videos via Acoustic–Semantic Decomposed Modeling},
  author  = {Pham, Long-Khanh and Tran, Thanh V. T. and Pham, Minh-Tan and Nguyen, Van},
  journal = {Interspeech 2025},
  year    = {2025},
  url     = {https://arxiv.org/abs/2505.22024v1}
}
```

---

## 📕 Overview
RESOUND separates **acoustic** (prosody/timbre from a short speaker prompt) and **semantic** (linguistic content from visual cues) paths, then decodes **mel-spectrograms + discrete units** before vocoding to waveform. This disentanglement improves naturalness and intelligibility.

---

## ⚙️ Setup code environment
```bash
conda create -n resound python=3.10 -y
conda activate resound
pip install -r requirements.txt
```

---

## 📂 Data Preparation

Please follow the official pipeline from **lip2speech-unit**:  
https://github.com/choijeongsoo/lip2speech-unit

This repository reuses the same directory structure, manifests, and features. No additional instructions are provided here.

---

## 🚀 Train
```bash
bash encoder/scripts/lrs3/train_avhubert_lrs3.sh
```

---

## 🔊 Inference
```bash
bash encoder/scripts/lrs3/inference_avhubert_lrs3-sh
bash vocoder/scripts/lrs3/inference.sh
```

---

## 🎗️ Acknowledgments
This repository is built using [Fairseq](https://github.com/pytorch/fairseq), [AV-HuBERT](https://https://github.com/facebookresearch/av_hubert), [ESPnet](https://github.com/espnet/espnet), [speech-resynthesis](https://github.com/facebookresearch/speech-resynthesis). We appreciate the open source of the projects.




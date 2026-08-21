# Study Notes Generator 🎓

An automated, reusable Python pipeline that transforms YouTube video lecture playlists into high-fidelity Markdown study notes, paired interactive Jupyter Notebooks (`.ipynb`), and publication-quality XeLaTeX PDFs using local Ollama LLMs and computer vision slide extraction.

---

## 🌟 Featured Course Showcase

The complete 18-lecture study notes for **Stanford CS231N (Spring 2025)** generated with this tool are live at:

- 📖 **Interactive Online Textbook:** 👉 **[https://yellowlab9.github.io/lecture-notes-stanford-cs231n-2025/](https://yellowlab9.github.io/lecture-notes-stanford-cs231n-2025/)**
- 📂 **Course Notes Repository:** 👉 **[https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025)**
- 📥 **Download All 18 14pt PDFs:** 👉 **[Release v2.0.0 Bundle](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025/releases/tag/v2.0.0)**

---

## Key Features

- **🎥 High-Fidelity Slide Detection and Extraction:** Decodes frames via PyAV and detects slide transitions using OpenCV and Kernel Density Estimation (KDE) mode-image extraction.
- **🤖 Minimal-Difference LLM Formatting:** Cleans raw speech into clear paragraphs and standard LaTeX mathematical formulas using local Ollama LLMs (e.g. `gemma4:latest`, `qwen2.5`) while strictly preserving verbatim explanations and examples.
- **📓 Two-Way Jupytext Pairing (`.md` $\leftrightarrow$ `.ipynb`):** Encapsulates each slide and its corresponding explanation into explicit cell regions (`<!-- #region -->`), allowing seamless two-way editing in MarkText, Obsidian, VS Code, or JupyterLab.
- **📄 Publication-Quality PDFs:** Converts Markdown notes directly to XeLaTeX PDFs using Pandoc with native LaTeX captions, non-floating `[H]` figures, and configurable font sizes (14pt default via `extarticle`).
- **⚡ Self-Recovering Cache:** Skips repeated video downloads and slide extraction when cached assets exist.
- **🌐 Decoupled Architecture:** Output notes and slide images are directed cleanly into dedicated course repositories with zero code clutter.

---

## 📥 Download Compiled PDFs (Releases)

Pre-compiled, publication-quality 14pt PDFs with high-resolution slide captures are available on the **[Course Releases Page](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025/releases)**:

- 📄 **[Release v2.0.0: Complete Course (Lectures 1 to 18)](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025/releases/tag/v2.0.0)** *(Full 18-lecture bundle)*

---

## 📚 Course Curriculum & Study Guides (Stanford CS231N 2025)

The complete notes, notebooks, and slide images generated with this pipeline are hosted in the dedicated [lecture-notes-stanford-cs231n-2025](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025) repository:

| Lecture | Topic | Read Online | Google Colab / GitHub | Download Notebook & PDF |
| :---: | :--- | :---: | :---: | :---: |
| **01** | Introduction to Deep Learning for Computer Vision | [📖 Read Notes](https://yellowlab9.github.io/lecture-notes-stanford-cs231n-2025/lectures/lecture_01_notes_introduction.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yellowlab9/lecture-notes-stanford-cs231n-2025/blob/master/docs/lectures/lecture_01_notes_introduction.ipynb) [GitHub](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025/blob/master/docs/lectures/lecture_01_notes_introduction.ipynb) | [📥 .ipynb](https://raw.githubusercontent.com/yellowlab9/lecture-notes-stanford-cs231n-2025/master/docs/lectures/lecture_01_notes_introduction.ipynb) · [📄 PDF](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025/releases/download/v2.0.0/lecture_01_notes_introduction.pdf) |
| **02** | Image Classification with Linear Classifiers | [📖 Read Notes](https://yellowlab9.github.io/lecture-notes-stanford-cs231n-2025/lectures/lecture_02_notes_image_classification_with_linear_classifiers.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yellowlab9/lecture-notes-stanford-cs231n-2025/blob/master/docs/lectures/lecture_02_notes_image_classification_with_linear_classifiers.ipynb) [GitHub](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025/blob/master/docs/lectures/lecture_02_notes_image_classification_with_linear_classifiers.ipynb) | [📥 .ipynb](https://raw.githubusercontent.com/yellowlab9/lecture-notes-stanford-cs231n-2025/master/docs/lectures/lecture_02_notes_image_classification_with_linear_classifiers.ipynb) · [📄 PDF](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025/releases/download/v2.0.0/lecture_02_notes_image_classification_with_linear_classifiers.pdf) |
| **03** | Regularization and Optimization | [📖 Read Notes](https://yellowlab9.github.io/lecture-notes-stanford-cs231n-2025/lectures/lecture_03_notes_regularization_and_optimization.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yellowlab9/lecture-notes-stanford-cs231n-2025/blob/master/docs/lectures/lecture_03_notes_regularization_and_optimization.ipynb) [GitHub](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025/blob/master/docs/lectures/lecture_03_notes_regularization_and_optimization.ipynb) | [📥 .ipynb](https://raw.githubusercontent.com/yellowlab9/lecture-notes-stanford-cs231n-2025/master/docs/lectures/lecture_03_notes_regularization_and_optimization.ipynb) · [📄 PDF](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025/releases/download/v2.0.0/lecture_03_notes_regularization_and_optimization.pdf) |
| **04** | Neural Networks and Backpropagation | [📖 Read Notes](https://yellowlab9.github.io/lecture-notes-stanford-cs231n-2025/lectures/lecture_04_notes_neural_networks_and_backpropagation.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yellowlab9/lecture-notes-stanford-cs231n-2025/blob/master/docs/lectures/lecture_04_notes_neural_networks_and_backpropagation.ipynb) [GitHub](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025/blob/master/docs/lectures/lecture_04_notes_neural_networks_and_backpropagation.ipynb) | [📥 .ipynb](https://raw.githubusercontent.com/yellowlab9/lecture-notes-stanford-cs231n-2025/master/docs/lectures/lecture_04_notes_neural_networks_and_backpropagation.ipynb) · [📄 PDF](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025/releases/download/v2.0.0/lecture_04_notes_neural_networks_and_backpropagation.pdf) |
| **05** | Image Classification with CNNs | [📖 Read Notes](https://yellowlab9.github.io/lecture-notes-stanford-cs231n-2025/lectures/lecture_05_notes_image_classification_with_cnns.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yellowlab9/lecture-notes-stanford-cs231n-2025/blob/master/docs/lectures/lecture_05_notes_image_classification_with_cnns.ipynb) [GitHub](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025/blob/master/docs/lectures/lecture_05_notes_image_classification_with_cnns.ipynb) | [📥 .ipynb](https://raw.githubusercontent.com/yellowlab9/lecture-notes-stanford-cs231n-2025/master/docs/lectures/lecture_05_notes_image_classification_with_cnns.ipynb) · [📄 PDF](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025/releases/download/v2.0.0/lecture_05_notes_image_classification_with_cnns.pdf) |
| **06** | CNN Architectures | [📖 Read Notes](https://yellowlab9.github.io/lecture-notes-stanford-cs231n-2025/lectures/lecture_06_notes_cnn_architectures.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yellowlab9/lecture-notes-stanford-cs231n-2025/blob/master/docs/lectures/lecture_06_notes_cnn_architectures.ipynb) [GitHub](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025/blob/master/docs/lectures/lecture_06_notes_cnn_architectures.ipynb) | [📥 .ipynb](https://raw.githubusercontent.com/yellowlab9/lecture-notes-stanford-cs231n-2025/master/docs/lectures/lecture_06_notes_cnn_architectures.ipynb) · [📄 PDF](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025/releases/download/v2.0.0/lecture_06_notes_cnn_architectures.pdf) |
| **07** | Recurrent Neural Networks (RNNs) | [📖 Read Notes](https://yellowlab9.github.io/lecture-notes-stanford-cs231n-2025/lectures/lecture_07_notes_recurrent_neural_networks.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yellowlab9/lecture-notes-stanford-cs231n-2025/blob/master/docs/lectures/lecture_07_notes_recurrent_neural_networks.ipynb) [GitHub](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025/blob/master/docs/lectures/lecture_07_notes_recurrent_neural_networks.ipynb) | [📥 .ipynb](https://raw.githubusercontent.com/yellowlab9/lecture-notes-stanford-cs231n-2025/master/docs/lectures/lecture_07_notes_recurrent_neural_networks.ipynb) · [📄 PDF](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025/releases/download/v2.0.0/lecture_07_notes_recurrent_neural_networks.pdf) |
| **08** | Attention and Transformers | [📖 Read Notes](https://yellowlab9.github.io/lecture-notes-stanford-cs231n-2025/lectures/lecture_08_notes_attention_and_transformers.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yellowlab9/lecture-notes-stanford-cs231n-2025/blob/master/docs/lectures/lecture_08_notes_attention_and_transformers.ipynb) [GitHub](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025/blob/master/docs/lectures/lecture_08_notes_attention_and_transformers.ipynb) | [📥 .ipynb](https://raw.githubusercontent.com/yellowlab9/lecture-notes-stanford-cs231n-2025/master/docs/lectures/lecture_08_notes_attention_and_transformers.ipynb) · [📄 PDF](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025/releases/download/v2.0.0/lecture_08_notes_attention_and_transformers.pdf) |
| **09** | Object Detection, Image Segmentation & Visualizing | [📖 Read Notes](https://yellowlab9.github.io/lecture-notes-stanford-cs231n-2025/lectures/lecture_09_notes_object_detection_image_segmentation_visualizing.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yellowlab9/lecture-notes-stanford-cs231n-2025/blob/master/docs/lectures/lecture_09_notes_object_detection_image_segmentation_visualizing.ipynb) [GitHub](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025/blob/master/docs/lectures/lecture_09_notes_object_detection_image_segmentation_visualizing.ipynb) | [📥 .ipynb](https://raw.githubusercontent.com/yellowlab9/lecture-notes-stanford-cs231n-2025/master/docs/lectures/lecture_09_notes_object_detection_image_segmentation_visualizing.ipynb) · [📄 PDF](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025/releases/download/v2.0.0/lecture_09_notes_object_detection_image_segmentation_visualizing.pdf) |
| **10** | Video Understanding | [📖 Read Notes](https://yellowlab9.github.io/lecture-notes-stanford-cs231n-2025/lectures/lecture_10_notes_video_understanding.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yellowlab9/lecture-notes-stanford-cs231n-2025/blob/master/docs/lectures/lecture_10_notes_video_understanding.ipynb) [GitHub](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025/blob/master/docs/lectures/lecture_10_notes_video_understanding.ipynb) | [📥 .ipynb](https://raw.githubusercontent.com/yellowlab9/lecture-notes-stanford-cs231n-2025/master/docs/lectures/lecture_10_notes_video_understanding.ipynb) · [📄 PDF](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025/releases/download/v2.0.0/lecture_10_notes_video_understanding.pdf) |
| **11** | Large-Scale Distributed Training | [📖 Read Notes](https://yellowlab9.github.io/lecture-notes-stanford-cs231n-2025/lectures/lecture_11_notes_large_scale_distributed_training.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yellowlab9/lecture-notes-stanford-cs231n-2025/blob/master/docs/lectures/lecture_11_notes_large_scale_distributed_training.ipynb) [GitHub](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025/blob/master/docs/lectures/lecture_11_notes_large_scale_distributed_training.ipynb) | [📥 .ipynb](https://raw.githubusercontent.com/yellowlab9/lecture-notes-stanford-cs231n-2025/master/docs/lectures/lecture_11_notes_large_scale_distributed_training.ipynb) · [📄 PDF](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025/releases/download/v2.0.0/lecture_11_notes_large_scale_distributed_training.pdf) |
| **12** | Self-Supervised Learning | [📖 Read Notes](https://yellowlab9.github.io/lecture-notes-stanford-cs231n-2025/lectures/lecture_12_notes_self_supervised_learning.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yellowlab9/lecture-notes-stanford-cs231n-2025/blob/master/docs/lectures/lecture_12_notes_self_supervised_learning.ipynb) [GitHub](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025/blob/master/docs/lectures/lecture_12_notes_self_supervised_learning.ipynb) | [📥 .ipynb](https://raw.githubusercontent.com/yellowlab9/lecture-notes-stanford-cs231n-2025/master/docs/lectures/lecture_12_notes_self_supervised_learning.ipynb) · [📄 PDF](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025/releases/download/v2.0.0/lecture_12_notes_self_supervised_learning.pdf) |
| **13** | Generative Models (Part 1) | [📖 Read Notes](https://yellowlab9.github.io/lecture-notes-stanford-cs231n-2025/lectures/lecture_13_notes_generative_models_1.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yellowlab9/lecture-notes-stanford-cs231n-2025/blob/master/docs/lectures/lecture_13_notes_generative_models_1.ipynb) [GitHub](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025/blob/master/docs/lectures/lecture_13_notes_generative_models_1.ipynb) | [📥 .ipynb](https://raw.githubusercontent.com/yellowlab9/lecture-notes-stanford-cs231n-2025/master/docs/lectures/lecture_13_notes_generative_models_1.ipynb) · [📄 PDF](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025/releases/download/v2.0.0/lecture_13_notes_generative_models_1.pdf) |
| **14** | Generative Models (Part 2) | [📖 Read Notes](https://yellowlab9.github.io/lecture-notes-stanford-cs231n-2025/lectures/lecture_14_notes_generative_models_2.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yellowlab9/lecture-notes-stanford-cs231n-2025/blob/master/docs/lectures/lecture_14_notes_generative_models_2.ipynb) [GitHub](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025/blob/master/docs/lectures/lecture_14_notes_generative_models_2.ipynb) | [📥 .ipynb](https://raw.githubusercontent.com/yellowlab9/lecture-notes-stanford-cs231n-2025/master/docs/lectures/lecture_14_notes_generative_models_2.ipynb) · [📄 PDF](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025/releases/download/v2.0.0/lecture_14_notes_generative_models_2.pdf) |
| **15** | 3D Vision | [📖 Read Notes](https://yellowlab9.github.io/lecture-notes-stanford-cs231n-2025/lectures/lecture_15_notes_3d_vision.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yellowlab9/lecture-notes-stanford-cs231n-2025/blob/master/docs/lectures/lecture_15_notes_3d_vision.ipynb) [GitHub](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025/blob/master/docs/lectures/lecture_15_notes_3d_vision.ipynb) | [📥 .ipynb](https://raw.githubusercontent.com/yellowlab9/lecture-notes-stanford-cs231n-2025/master/docs/lectures/lecture_15_notes_3d_vision.ipynb) · [📄 PDF](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025/releases/download/v2.0.0/lecture_15_notes_3d_vision.pdf) |
| **16** | Vision and Language | [📖 Read Notes](https://yellowlab9.github.io/lecture-notes-stanford-cs231n-2025/lectures/lecture_16_notes_vision_and_language.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yellowlab9/lecture-notes-stanford-cs231n-2025/blob/master/docs/lectures/lecture_16_notes_vision_and_language.ipynb) [GitHub](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025/blob/master/docs/lectures/lecture_16_notes_vision_and_language.ipynb) | [📥 .ipynb](https://raw.githubusercontent.com/yellowlab9/lecture-notes-stanford-cs231n-2025/master/docs/lectures/lecture_16_notes_vision_and_language.ipynb) · [📄 PDF](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025/releases/download/v2.0.0/lecture_16_notes_vision_and_language.pdf) |
| **17** | Robot Learning | [📖 Read Notes](https://yellowlab9.github.io/lecture-notes-stanford-cs231n-2025/lectures/lecture_17_notes_robot_learning.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yellowlab9/lecture-notes-stanford-cs231n-2025/blob/master/docs/lectures/lecture_17_notes_robot_learning.ipynb) [GitHub](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025/blob/master/docs/lectures/lecture_17_notes_robot_learning.ipynb) | [📥 .ipynb](https://raw.githubusercontent.com/yellowlab9/lecture-notes-stanford-cs231n-2025/master/docs/lectures/lecture_17_notes_robot_learning.ipynb) · [📄 PDF](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025/releases/download/v2.0.0/lecture_17_notes_robot_learning.pdf) |
| **18** | Human-Centered AI | [📖 Read Notes](https://yellowlab9.github.io/lecture-notes-stanford-cs231n-2025/lectures/lecture_18_notes_human_centered_ai.html) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yellowlab9/lecture-notes-stanford-cs231n-2025/blob/master/docs/lectures/lecture_18_notes_human_centered_ai.ipynb) [GitHub](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025/blob/master/docs/lectures/lecture_18_notes_human_centered_ai.ipynb) | [📥 .ipynb](https://raw.githubusercontent.com/yellowlab9/lecture-notes-stanford-cs231n-2025/master/docs/lectures/lecture_18_notes_human_centered_ai.ipynb) · [📄 PDF](https://github.com/yellowlab9/lecture-notes-stanford-cs231n-2025/releases/download/v2.0.0/lecture_18_notes_human_centered_ai.pdf) |

---

## 📂 Decoupled Directory Architecture

The system cleanly separates the **reusable Python generator engine** from each **course notes & website repository**:

```text
Z:\auyeung\ml2026\
│
├── cs231n-lecture-notes-generator/        <-- [THIS REPOSITORY: Python Generator Engine]
│   ├── study_notes_generator/             <-- Core Python package (CV, LLM, PDF modules)
│   ├── generate_notes.py                  <-- Entrypoint CLI
│   ├── Makefile                           <-- Automated commands with COURSE_DIR support
│   ├── requirements.txt                   <-- Python dependencies
│   └── README.md
│
└── lecture-notes-stanford-cs231n-2025/    <-- [COURSE NOTES REPOSITORY & WEBSITE]
    ├── mkdocs.yml                         <-- Material website navigation & MathJax config
    ├── README.md                          <-- Public course index, syllabus & disclaimer
    ├── lectures/                          <-- [TRACKED] 18 .md notes, .ipynb notebooks & slides
    ├── lectures_pdf/                      <-- [LOCAL ONLY] 18 compiled 14pt PDFs
    └── lectures_cache/                    <-- [LOCAL ONLY] Video downloads & logs
```

---

## Prerequisites

1. **Python 3.10+** (Conda recommended)
2. **[Ollama](https://ollama.com/)** with a local model:
   ```bash  
   ollama pull gemma4:latest
   ```
3. **[Pandoc](https://pandoc.org/)** & **XeLaTeX** (TeX Live or MiKTeX) for PDF generation.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yellowlab9/cs231n-lecture-notes-generator.git
cd cs231n-lecture-notes-generator

# Install Python dependencies
pip install -r requirements.txt
```

---

## Quickstart & Usage

### 1. Generate Notes & Slides
Generate Markdown study notes and extract slides for a specific lecture:
```bash
# Default Lecture 1
make notes LECTURE=1

# Or specify a model
make notes LECTURE=2 MODEL=gemma4:latest
```

### 2. Compile to PDF
Compile the Markdown notes into a 14pt PDF:
```bash
# Compile a specific lecture (default 14pt)
make pdf LECTURE=1

# Compile all new or modified lecture notes (skips up-to-date PDFs):
make pdf-all

# Force re-compile all notes with custom font size:
make pdf-all FORCE=1 FONT_SIZE=12pt
```

### 3. Pair with Jupyter Notebook (.ipynb)
Generate or pair the .ipynb notebook with the Markdown note:
```bash
make ipynb LECTURE=1
```

### 4. Synchronize Edits (.md $\leftrightarrow$ .ipynb)
Synchronize changes between Markdown notes and Jupyter Notebooks across the entire workspace:
```bash
make sync
``` 

---

## CLI Reference

You can also run the generator directly via Python:

```bash
python generate_notes.py --help
```

Key arguments:
- --video_list_url: YouTube playlist or video URL.
- --index: 1-based index of the lecture in the playlist.
- --model: Local Ollama model (default: gemma4:latest).
- --fontsize: Font size for PDF compilation (default: 14pt).
- --img_width: Slide image width in notes (default: 75%).
- --display: Open live OpenCV preview windows during slide detection.

---

## License
MIT License

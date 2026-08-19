# Study Notes Generator 🎓

An automated, end-to-end pipeline that converts video lecture playlists (YouTube) into high-fidelity Markdown study notes, paired interactive Jupyter Notebooks (.ipynb), and publication-quality XeLaTeX PDFs. 

This generator is applicable to lecture video with talking heads interleved with lecture slides. Example videos include the lectures of Stanford CS231N Deep Learning for Computer Vision I 2025, with playlist url: https://www.youtube.com/playlist?list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16. 

---

## Key Features

- **🎥 High-Fidelity Slide Detection and Extraction:** Decodes frames via PyAV and detects slide transitions using OpenCV and Kernel Density Estimation (KDE) mode-image extraction.
- **🤖 Minimal-Difference LLM Formatting:** Cleans raw speech into clear paragraphs and standard LaTeX mathematical formulas using local Ollama LLMs (e.g. gemma4:latest, qwen2.5) while strictly preserving verbatim explanations and examples.
- **📓 Two-Way Jupytext Pairing (.md $\leftrightarrow$ .ipynb):** Encapsulates each slide and its corresponding explanation into explicit cell regions (<!-- #region -->), allowing seamless two-way editing in MarkText, Obsidian, VS Code, or JupyterLab.
- **📄 Publication-Quality PDFs:** Converts Markdown notes directly to XeLaTeX PDFs using Pandoc with native LaTeX captions, non-floating [H] figures, and configurable font sizes (14pt default via extarticle for viewing on screen). If you need to print it out, you can change the font size to 12pt.
- **⚡ Self-Recovering Cache:** Skips repeated video downloads and slide extraction when cached assets exist.

---

## Directory Structure

Generated course assets follow a clean, flattened layout:

```bash
Stanford_CS231N_Deep_Learning_for_Computer_Vision_I_2025/
├── lectures/
│   ├── lecture_01_notes_introduction.md
│   ├── lecture_01_notes_introduction.ipynb    <-- Paired Jupyter Notebook
│   ├── lecture_01_slides/                     <-- Extracted Slide Images
│   │   ├── slide_4_00-00-00.133.jpg
│   │   └── ...
│   ├── lecture_02_notes_image_classification_with_linear_classifiers.md
│   ├── lecture_02_notes_image_classification_with_linear_classifiers.ipynb
│   └── lecture_02_slides/
│
├── lectures_pdf/                              <-- Compiled XeLaTeX PDFs (14pt)
│   ├── lecture_01_notes_introduction.pdf
│   └── lecture_02_notes_image_classification_with_linear_classifiers.pdf
│
└── lectures_cache/                            <-- Intermediate Cache
    ├── media/
    ├── slide_csv/
    └── logs/
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
# Default 14pt
make pdf LECTURE=1

# Custom font size on the fly:
make pdf LECTURE=1 FONT_SIZE=12pt
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

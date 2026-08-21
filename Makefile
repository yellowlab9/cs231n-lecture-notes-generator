# Default variables (Override these from the command line)
LECTURE ?= 1
INDEX ?= $(LECTURE)
PLAYLIST_URL ?= https://www.youtube.com/playlist?list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16
COURSE_DIR ?= ../lecture-notes-stanford-cs231n-2025
MODEL ?= gemma4:latest

# Toggles for logging and visual monitor
VERBOSE_FLAG ?= --verbose
DISPLAY_FLAG ?= --display
DEBUG_FLAG   ?= --debug

IMG_WIDTH ?= 75%
FONT_SIZE ?= 14pt
PDF_ENGINE ?= xelatex
PREFIX_FLAG ?= $(if $(OUTPUT_PREFIX),--output_prefix $(OUTPUT_PREFIX),)
COURSE_FLAG ?= $(if $(COURSE_DIR),--playlist_dir "$(COURSE_DIR)",)

# Phony targets
.PHONY: notes pdf pdf-all ipynb sync release clean clean_slides clean_cache clean_pdf debug-args vscode-launch

PYTHON ?= python
TAG ?= v2.0.0
TITLE ?= Stanford CS231N Study Notes ($(TAG))

# Default target
notes:
	@echo "=========================================================="
	@echo "Generating Notes for Lecture $(LECTURE) (Index $(INDEX))"
	@echo "Course Directory: $(COURSE_DIR)"
	@echo "Playlist URL: $(PLAYLIST_URL)"
	@echo "Model: $(MODEL)"
	@echo "Image Width: $(IMG_WIDTH)"
	@echo "Font Size: $(FONT_SIZE)"
	@echo "=========================================================="
	"$(PYTHON)" generate_notes.py --video_list_url "$(PLAYLIST_URL)" --index $(INDEX) $(COURSE_FLAG) $(PREFIX_FLAG) --model $(MODEL) --img_width $(IMG_WIDTH) --fontsize $(FONT_SIZE) $(VERBOSE_FLAG) $(DISPLAY_FLAG) $(DEBUG_FLAG)

# Convert a single Markdown study guide to PDF via Pandoc
pdf:
	@echo "=========================================================="
	@echo "Converting Study Guide to PDF for Lecture $(LECTURE) via Pandoc ($(PDF_ENGINE), $(FONT_SIZE))"
	@echo "=========================================================="
	@"$(PYTHON)" -m study_notes_generator.compile_pdf --lecture $(LECTURE) --engine $(PDF_ENGINE) --fontsize $(FONT_SIZE)

# Convert all new or modified Markdown study guides to PDF
pdf-all:
	@echo "=========================================================="
	@echo "Checking and Compiling All New / Modified Notes to PDF ($(PDF_ENGINE), $(FONT_SIZE))"
	@echo "=========================================================="
	@"$(PYTHON)" -m study_notes_generator.compile_pdf --all --engine $(PDF_ENGINE) --fontsize $(FONT_SIZE) $(if $(FORCE),--force,)

# Convert/Pair Markdown study guide to Jupyter Notebook (.ipynb) via Jupytext
ipynb:
	@echo "=========================================================="
	@echo "Pairing / Converting Lecture $(LECTURE) to Jupyter Notebook (.ipynb)"
	@echo "=========================================================="
	@"$(PYTHON)" -c "import glob, subprocess, sys; lec = '$(LECTURE)'.replace('lecture_', ''); num = f'{int(lec):02d}' if lec.isdigit() else lec; matches = glob.glob(f'**/lectures/lecture_{num}_notes_*.md', recursive=True); sys.exit(subprocess.run(['$(PYTHON)', '-m', 'jupytext', '--to', 'ipynb', matches[0]]).returncode) if matches else print(f'No notes found for lecture {lec}')"

# Synchronize all paired Markdown and Jupyter Notebooks in the workspace
sync:
	@echo "=========================================================="
	@echo "Synchronizing all paired .md and .ipynb notes via Jupytext"
	@echo "=========================================================="
	@"$(PYTHON)" -c "import glob, subprocess; files = glob.glob('**/lectures/lecture_*_notes_*.md', recursive=True); [subprocess.run(['$(PYTHON)', '-m', 'jupytext', '--sync', f]) for f in files]"

# Publish/Upload compiled PDFs to GitHub Releases via gh CLI
release:
	@echo "=========================================================="
	@echo "Publishing GitHub Release $(TAG) with compiled PDFs"
	@echo "=========================================================="
	@"$(PYTHON)" -c "import glob, subprocess; pdfs = sorted(glob.glob('**/lectures_pdf/lecture_*_notes_*.pdf', recursive=True)); cmd = ['gh', 'release', 'create', '$(TAG)'] + pdfs + ['-R', 'yellowlab9/cs231n-lecture-notes-generator', '--title', '$(TITLE)', '--notes', 'High-fidelity lecture study guides (14pt XeLaTeX PDFs) with slide captures.', '--verify-tag']; subprocess.run(cmd)"

# Clean up local environment
clean:
	@echo "Cleaning up lectures, pdfs, and cache..."
	rm -rf lectures_cache/ */lectures_cache/
	rm -rf lectures/ */lectures/
	rm -rf lectures_pdf/ */lectures_pdf/
	rm -f *.mp4 *.m4a *.vtt *.srt *.part *.pdf *.md *.csv *.txt

clean_cache:
	@echo "Cleaning up temporary media and slide caches..."
	rm -rf lectures_cache/ */lectures_cache/

clean_pdf:
	@echo "Cleaning up compiled PDFs..."
	rm -rf lectures_pdf/ */lectures_pdf/
	
clean_slides:
	@echo "Cleaning up slides..."
	mkdir -p slides
	rm -f slides/*

# Helper to generate args for VSCode launch.json
debug-args:
	@echo "Copy the following into your launch.json 'args' array:"
	@echo "--video_list_url", "\"$(PLAYLIST_URL)\"", "--index", "$(INDEX)", "--output_prefix", "$(OUTPUT_PREFIX)", "--model", "$(MODEL)", "$(VERBOSE_FLAG)", "$(DISPLAY_FLAG)", "$(DEBUG_FLAG)" | sed 's/--[^ ]*/"&",/g' | sed 's/, *$$//'

# Generate VSCode launch.json file for debugging
vscode-launch:
	@echo "Generating .vscode/launch.json..."
	@mkdir -p .vscode && \
	printf '{\n' > .vscode/launch.json && \
	printf '    "version": "0.2.0",\n' >> .vscode/launch.json && \
	printf '    "configurations": [\n' >> .vscode/launch.json && \
	printf '        {\n' >> .vscode/launch.json && \
	printf '            "name": "Python: Debug Notes Generator (from Makefile)",\n' >> .vscode/launch.json && \
	printf '            "type": "debugpy",\n' >> .vscode/launch.json && \
	printf '            "request": "launch",\n' >> .vscode/launch.json && \
	printf '            "program": "$${workspaceFolder}/generate_notes.py",\n' >> .vscode/launch.json && \
	printf '            "console": "integratedTerminal",\n' >> .vscode/launch.json && \
	printf '            "args": [\n' >> .vscode/launch.json && \
	printf '                "--video_list_url",\n' >> .vscode/launch.json && \
	printf '                "$(PLAYLIST_URL)",\n' >> .vscode/launch.json && \
	printf '                "--index",\n' >> .vscode/launch.json && \
	printf '                "$(INDEX)",\n' >> .vscode/launch.json && \
	printf '                "--output_prefix",\n' >> .vscode/launch.json && \
	printf '                "$(OUTPUT_PREFIX)",\n' >> .vscode/launch.json && \
	printf '                "--model",\n' >> .vscode/launch.json && \
	printf '                "$(MODEL)",\n' >> .vscode/launch.json && \
	printf '                "$(VERBOSE_FLAG)",\n' >> .vscode/launch.json && \
	printf '                "$(DISPLAY_FLAG)",\n' >> .vscode/launch.json && \
	printf '                "$(DEBUG_FLAG)"\n' >> .vscode/launch.json && \
	printf '            ]\n' >> .vscode/launch.json && \
	printf '        }\n' >> .vscode/launch.json && \
	printf '    ]\n' >> .vscode/launch.json && \
	printf '}\n' >> .vscode/launch.json
	@echo "Done. You can now use the 'Run and Debug' panel in VSCode."

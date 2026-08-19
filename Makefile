# Default variables (Override these from the command line)
LECTURE ?= 3
INDEX ?= $(LECTURE)
PLAYLIST_URL ?= https://www.youtube.com/playlist?list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16
MODEL ?= gemma4:latest

# Toggles for logging and visual monitor
VERBOSE_FLAG ?= --verbose
DISPLAY_FLAG ?= --display
DEBUG_FLAG   ?= --debug

IMG_WIDTH ?= 75%
FONT_SIZE ?= 14pt
PDF_ENGINE ?= xelatex
PREFIX_FLAG ?= $(if $(OUTPUT_PREFIX),--output_prefix $(OUTPUT_PREFIX),)

# Phony targets
.PHONY: notes pdf clean clean_slides clean_cache clean_pdf debug-args vscode-launch

PYTHON ?= python

# Default target
notes:
	@echo "=========================================================="
	@echo "Generating Notes for Lecture $(LECTURE) (Index $(INDEX))"
	@echo "Playlist URL: $(PLAYLIST_URL)"
	@echo "Model: $(MODEL)"
	@echo "Image Width: $(IMG_WIDTH)"
	@echo "Font Size: $(FONT_SIZE)"
	@echo "=========================================================="
	"$(PYTHON)" generate_notes.py --video_list_url "$(PLAYLIST_URL)" --index $(INDEX) $(PREFIX_FLAG) --model $(MODEL) --img_width $(IMG_WIDTH) --fontsize $(FONT_SIZE) $(VERBOSE_FLAG) $(DISPLAY_FLAG) $(DEBUG_FLAG)

# Convert Markdown study guide to PDF via Pandoc
pdf:
	@echo "=========================================================="
	@echo "Converting Study Guide to PDF for Lecture $(LECTURE) via Pandoc ($(PDF_ENGINE), $(FONT_SIZE))"
	@echo "=========================================================="
	@"$(PYTHON)" -m study_notes_generator.compile_pdf --lecture $(LECTURE) --engine $(PDF_ENGINE) --fontsize $(FONT_SIZE)

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

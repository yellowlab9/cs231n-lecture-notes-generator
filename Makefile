# Default variables (Override these from the command line)
LECTURE ?= 3
INDEX ?= $(LECTURE)
PLAYLIST_URL ?= https://www.youtube.com/playlist?list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16
MODEL ?= gemma4:latest

# Toggles for logging and visual monitor
VERBOSE_FLAG ?= --verbose
DISPLAY_FLAG ?= --display
DEBUG_FLAG   ?= --debug

LECTURE_NUM := $(shell printf "%02d" $(LECTURE) 2>/dev/null || echo $(LECTURE))
OUTPUT_PREFIX ?= lecture_$(LECTURE_NUM)
IMG_WIDTH ?= 75%
PDF_ENGINE ?= xelatex

LECTURE_DIR ?= lectures/$(OUTPUT_PREFIX)
MD_FILE ?= $(LECTURE_DIR)/$(OUTPUT_PREFIX)_study_guide.md
PDF_FILE ?= $(LECTURE_DIR)/$(OUTPUT_PREFIX)_study_guide.pdf

# Phony targets
.PHONY: notes pdf clean clean_slides clean_cache debug-args vscode-launch

PYTHON ?= python

# Default target
notes:
	@echo "=========================================================="
	@echo "Generating Notes for Lecture $(LECTURE) (Index $(INDEX))"
	@echo "Playlist URL: $(PLAYLIST_URL)"
	@echo "Model: $(MODEL)"
	@echo "Output: $(MD_FILE)"
	@echo "=========================================================="
	$(PYTHON) generate_notes.py --video_list_url "$(PLAYLIST_URL)" --index $(INDEX) --output_prefix $(OUTPUT_PREFIX) --model $(MODEL) --img_width $(IMG_WIDTH) $(VERBOSE_FLAG) $(DISPLAY_FLAG) $(DEBUG_FLAG)

# Convert Markdown study guide to PDF via Pandoc
pdf:
	@echo "=========================================================="
	@echo "Converting $(MD_FILE) -> $(PDF_FILE) via Pandoc ($(PDF_ENGINE))"
	@echo "=========================================================="
	pandoc "$(MD_FILE)" -o "$(PDF_FILE)" --pdf-engine=$(PDF_ENGINE) --resource-path="$(LECTURE_DIR)" --lua-filter=study_guide_generator/html_filter.lua
	@echo "Done! Generated $(PDF_FILE)"

# Clean up local environment
clean:
	@echo "Cleaning up lectures and cache..."
	rm -rf lectures_cache/
	rm -rf lectures/
	rm -f *.mp4 *.m4a *.vtt *.srt *.part *.pdf *.md *.csv *.txt

clean_cache:
	@echo "Cleaning up temporary media and slide caches..."
	rm -rf lectures_cache/
	
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

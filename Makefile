# Default variables (Override these from the command line)
LECTURE ?= 3
VIDEO_ID ?= dyNGd06MWn4
# Changed to latest as requested
MODEL ?= gemma4:latest

# Toggles for logging and visual monitor
VERBOSE_FLAG ?= --verbose
DISPLAY_FLAG ?= --display
DEBUG_FLAG   ?= --debug

# Constructed URLs
VIDEO_URL = https://www.youtube.com/watch?v=$(VIDEO_ID)

PDF_URL = lecture_$(LECTURE).pdf

# Phony targets
.PHONY: notes clean debug-args vscode-launch

# Default target
notes:
	@echo "=========================================================="
	@echo "Generating Notes for Lecture $(LECTURE)"
	@echo "Video ID: $(VIDEO_ID)"
	@echo "Model: $(MODEL)"
	@echo "=========================================================="
	python generate_notes.py --video_url "$(VIDEO_URL)" --pdf $(PDF_URL) --model $(MODEL) $(VERBOSE_FLAG) $(DISPLAY_FLAG) $(DEBUG_FLAG)

# Clean up local environment
clean:
	@echo "Cleaning up media, slides, and markdown files..."
	rm -rf slides/
	rm -f lecture*.mp4 lecture*.vtt lecture*.srt
	rm -f lecture*.pdf lecture_slides.pdf semantic_study_notes.md
	
clean_slides:
	@echo "Cleaning up slides..."
	mkdir -p slides
	rm -f slides/*

# Helper to generate args for VSCode launch.json
debug-args:
	@echo "Copy the following into your launch.json 'args' array:"
	@echo "--video_url", "\"$(VIDEO_URL)\"", "--pdf", "$(PDF_URL)", "--model", "$(MODEL)", "$(VERBOSE_FLAG)", "$(DISPLAY_FLAG)", "$(DEBUG_FLAG)" | sed 's/--[^ ]*/"&",/g' | sed 's/, *$$//'

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
	printf '                "--video_url",\n' >> .vscode/launch.json && \
	printf '                "$(VIDEO_URL)",\n' >> .vscode/launch.json && \
	printf '                "--pdf",\n' >> .vscode/launch.json && \
	printf '                "$(PDF_URL)",\n' >> .vscode/launch.json && \
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

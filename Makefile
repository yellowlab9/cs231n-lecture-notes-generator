# Default variables (Override these from the command line)
LECTURE ?= 3
INDEX ?= $(LECTURE)
PLAYLIST_URL ?= https://www.youtube.com/playlist?list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16
# Changed to latest as requested
MODEL ?= gemma4:latest

# Toggles for logging and visual monitor
VERBOSE_FLAG ?= --verbose
DISPLAY_FLAG ?= --display
DEBUG_FLAG   ?= --debug

OUTPUT_PREFIX ?= lecture_$(LECTURE)
IMG_WIDTH ?= 60%

# Phony targets
.PHONY: notes clean debug-args vscode-launch

# Default target
notes:
	@echo "=========================================================="
	@echo "Generating Notes for Lecture $(LECTURE) (Index $(INDEX))"
	@echo "Playlist URL: $(PLAYLIST_URL)"
	@echo "Model: $(MODEL)"
	@echo "Image Width: $(IMG_WIDTH)"
	@echo "=========================================================="
	python generate_notes.py --video_list_url "$(PLAYLIST_URL)" --index $(INDEX) --output_prefix $(OUTPUT_PREFIX) --model $(MODEL) --img_width $(IMG_WIDTH) $(VERBOSE_FLAG) $(DISPLAY_FLAG) $(DEBUG_FLAG)

# Clean up local environment
clean:
	@echo "Cleaning up media, slides, and markdown files..."
	rm -rf slides/
	rm -f lecture*.mp4 lecture*.m4a lecture*.vtt lecture*.srt lecture*.part
	rm -f lecture*.pdf lecture_slides.pdf semantic_study_notes.md
	
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

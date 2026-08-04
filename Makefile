# Default variables (Override these from the command line)
LECTURE ?= 3
VIDEO_ID ?= dyNGd06MWn4
# Changed to latest as requested
MODEL ?= gemma4:latest

# Toggles for logging and visual monitor
VERBOSE ?= --verbose
DISPLAY ?= --display
DEBUG   ?= --debug

# Constructed URLs
VIDEO_URL = "https://www.youtube.com/watch?v=$(VIDEO_ID)"

PDF_URL = lecture_3.pdf

# Phony targets
.PHONY: notes clean

# Default target
notes:
	@echo "=========================================================="
	@echo "Generating Notes for Lecture $(LECTURE)"
	@echo "Video ID: $(VIDEO_ID)"
	@echo "Model: $(MODEL)"
	@echo "=========================================================="
#	make clean_slides
	python generate_notes.py --video_url $(VIDEO_URL) --pdf $(PDF_URL) --model $(MODEL) $(VERBOSE) $(DISPLAY) $(DEBUG)

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

# 2026-08-13

## Today's Goal
- Setup CLINE to use Gemini for plaining and Gemma4 for action.

## Output from Gemini
I have completed a thorough scan of the codebase. Below is a detailed breakdown of the codebase architecture, components, file structure, and implementation details.

---

### 1. Overview
The codebase is a **Study Guide Generator** designed to process video lectures (such as those from YouTube) and their corresponding PDF slides to produce high-quality, formatted Markdown study guides compatible with MarkText. 

It implements:
*   **Media Downloader**: Fetching the YouTube video and its auto-generated or uploaded English transcript using `yt-dlp`.
*   **Slide Detector**: A custom, high-fidelity frame-by-frame slide extraction module utilizing `PyAV` and `OpenCV` with advanced features like KDE mode-image extraction.
*   **Transcript Parser**: Overlapping chunk parser for `.vtt` and `.srt` subtitles.
*   **LLM Processor**: Cleaning raw transcript text and formatting mathematical expressions into LaTeX via a local Ollama instance (default: `gemma4:latest`).
*   **Note Generator**: Aligning the text paragraphs chronologically with the extracted slides, matching sentences with timestamps using fuzzy string matching (`fuzzywuzzy`), and outputting the final `.md` study guide.

---

### 2. File Structure & Component Map

*   **`generate_notes.py`**: The main entrypoint file that imports and executes the package CLI logic.
*   **`Makefile`**: Facilitates easy execution and cleanup. Features targets like `notes` (to run the pipeline), `clean` (cleanup media/output), and `vscode-launch` (scaffolds `launch.json` for debugging).
*   **`study_guide_generator/`**: The core package directory containing:
    *   **`__init__.py`**: Initializer making the folder a Python package.
    *   **`main.py`**: Handles command-line arguments (using `argparse`) grouped by core parameters, note generation, slide detection, and debugging. Sets up the output directories and triggers the pipeline.
    *   **`downloader.py`**: Downloads PDF files via `requests` and media/subtitles via `yt_dlp` with targeted formats (e.g. up to 1080p, `.mp4`).
    *   **`slide_detector.py`**: Decodes frames using multi-threaded PyAV, filters out B-frames, and samples frames to detect slide changes. Implements a Kernel Density Estimation (KDE) mode estimator (`kde_mode`) via `scipy.stats.mode` to compute a clean, stable representation of each slide. Includes a cache mechanism (`load_slides_from_cache`) using a local slide CSV file and slide file checks.
    *   **`transcript_parser.py`**: Parses `.vtt` or `.srt` files into fine-grained segments and aggregates them into larger, overlapping text chunks of defined durations.
    *   **`llm_processor.py`**: Connects to the local Ollama server at `127.0.0.1:11434` and manages prompt templates that instruct the LLM to format mathematical symbols to LaTeX, clean grammar/fillers without rephrasing, and structure paragraphs.
    *   **`note_generator.py`**: Coordinates LLM calling, executes fuzzy matching of the first sentence of LLM-cleaned paragraphs against raw subtitle segments to find exact/estimated timestamps, aggregates text and slide timeline items, sorts them chronologically, and writes the output `.md` file.
    *   **`utils.py`**: Utility helpers for formatted logging to stdout and file, as well as time-string conversions.

---

### 3. Implementation Steps & Success Criteria for Pipeline Execution
If you wish to execute or run the tool on an actual lecture, here are the steps we will verify:
1.  **Server Readiness**: Ensure the local Ollama server is running.
2.  **Run with Command**: Invoke `make notes LECTURE=3 VIDEO_ID=dyNGd06MWn4` or run the Python script directly.
3.  **Validate Cache**: Verify that if run twice, slide detection is skipped and loaded from the cached `.csv` file.
4.  **Confirm Output**: Ensure the generated `lecture_3_study_guide.md` file correctly combines the LaTeX-cleaned transcript chunks and slides chronologically.

Since this task was to **scan the codebase**, the analysis is complete and no code modifications are necessary. Please let me know how you would like to proceed!


## Output from Gemini
I have completed a comprehensive scan of the codebase. The project is structured into the `study_guide_generator` package and implements a pipeline to transform video lectures and PDF slides into organized Markdown study guides.

### System Architecture Overview:
1. **Orchestration (`main.py`)**: Acts as the central hub, managing the sequence of operations: pre-flight checks (Ollama connectivity), resource downloading, slide detection, transcript processing, and final note generation.
2. **Data Acquisition (`downloader.py`)**: Handles fetching of external resources (PDFs and media) from URLs like YouTube.
3. **Visual Analysis (`slide_detector.py`)**: Analyzes video frames to identify and timestamp slides, providing a structural backbone for the final document.
4. **Text Processing (`transcript_parser.py`)**: Parses `.vtt`/`.srt` files, handles time conversions, and groups transcript segments into overlapping chunks to preserve context for LLM processing.
5. **Content Synthesis (`note_generator.py`)**: The core logic layer that:
    - Processes raw transcripts through an LLM via `llm_processor`.
    - Uses NLTK and fuzzy matching (`fuzzywuzzy`) to link processed text back to accurate timestamps.
    - Merges these "paragraphs" with "slide" data into a unified, chronologically sorted list.
6. **Utility Layer (`utils.py`)**: Provides shared functions for logging (both to console and file) and time formatting.

### Key Features:
- **Hybrid Workflow**: Combines automated CV/media handling with LLM-driven content cleaning.
- **Robust Mapping**: Employs fuzzy matching to ensure that even after LLM "cleaning," text is correctly anchored to the original video's timeline.
- **Modular Design**: Each stage (downloading, parsing, summarizing) is decoupled, making it easy to modify specific parts of the pipeline.

The project is well-structured with clear separation of concerns between data fetching, processing, and formatting.

## Output from Gemini: Run git diff and recommend messages for git commit

Based on the `git diff`, here are the recommended git commit messages categorized by the scope of the changes:

### Option 1: Single Comprehensive Message (Recommended for a single commit)
`feat: enhance LLM robustness, expand CLI options, and improve transcript processing logic`

### Option 2: Granular Messages (Recommended if splitting into separate commits)

**For `llm_processor.py` updates:**
- `feat(llm): implement Ollama client singleton and server connectivity checks`
- `feat(llm): add retry logic and improved prompt structure for MarkText compatibility`
- `refactor(llm): update system prompts for aggressive LaTeX formatting`

**For `main.py` updates:**
- `feat(cli): expand command-line arguments with categorized groups and descriptions`
- `feat(core): implement pre-flight check for Ollama server availability`
- `refactor(core): update main workflow to utilize new configuration parameters`

**For `note_generator.py` updates:**
- `fix(note): deduplicate overlapping content using a processed paragraph tracker`
- `feature(note): introduce fuzzy matching thresholds for sentence alignment`
- `refactor(note): simplify final study guide construction logic`

**For `transcript_parser.py` updates:**
- `feat(parsing): add in-place progress indicator for transcript processing`
- `refactor(parsing): update default chunking and overlap constants`

### Summary of Key Improvements Identified:
*   **Robustness**: Added a singleton client, retry mechanisms, and pre-flight checks to ensure the system handles LLM connectivity issues gracefully.
*   **UX/UI**: Enhanced the CLI experience with grouped parameters and added a real-time progress bar during the long transcript parsing stage.
*   **Formatting**: Significantly upgraded the prompt engineering to specifically target MarkText's LaTeX requirements (e.g., `\mathrm`, proper subscripts, and escaping).
*   **Accuracy**: Improved the mapping between LLM-generated text and original timestamps by using fuzzy scores and better deduplication logic for overlapping chunks.

## GEMINI

Request
I am planning to use this  python script to process multiple lectures, for example, for all lecture videso from  https://www.youtube.com/playlist?list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16 . Suggest a good directory structure to store the lecture notes and changes to the python scripts to make the task easier.

## Gemini Output: Request for changes to process multiple lectures

To process an entire playlist of lectures (like the CS231N Stanford playlist) efficiently without cluttering your workspace, we need to address two aspects:
1. **An Organized Directory Structure** that separates final output markdown files, slide assets, and temporary video/VTT cache files.
2. **Script Changes** to allow batch processing of a YouTube playlist directly.

---

### 1. Recommended Directory Structure

When running a batch process, you will generate gigabytes of temporary MP4s, hundreds of slide screenshots, and final Markdown guides. We should organize them into three clear zones:

```text
workspace/
├── generate_notes.py             # Single video entry point
├── batch_process.py              # New playlist/batch entry point
│
├── notes/                        # FINAL OUTPUTS ONLY (Clean & commit-friendly)
│   ├── lecture_1_introduction.md
│   ├── lecture_2_image_classification.md
│   └── assets/                   # Subfolders for each lecture's slide images
│       ├── lecture_1/
│       │   ├── slide_001.png
│       │   └── slide_002.png
│       └── lecture_2/
│           ├── slide_001.png
│           └── slide_002.png
│
├── temp_cache/                   # INTERMEDIATE CACHE (Excluded from Git / easily deleted)
│   ├── lecture_VIDEO_ID_1/
│   │   ├── video.mp4             # High-res video download
│   │   ├── transcript.en.vtt     # Raw transcript
│   │   ├── slide_data.csv        # Detected slide timestamps
│   │   └── process.log           # Log of the run
│   └── lecture_VIDEO_ID_2/
│       └── ...
│
└── study_guide_generator/        # Python library code
```

#### Why this structure is perfect:
*   **Zero Clutter:** All temporary/large binary assets (MP4s, logs, raw VTTs) go into a `temp_cache/` directory.
*   **Easy Git/Syncing:** You can add `temp_cache/` to `.gitignore` and only commit your high-quality `notes/` directory.
*   **Self-contained Notes:** Inside the markdown files, the slide images are referenced relative to the note (e.g. `![Slide 5](assets/lecture_1/slide_005.png)`), making the `notes/` folder fully portable for rendering in MarkText or Obsidian.

---

### 2. Suggested Script Changes

To achieve this, we can introduce a new entry script called `batch_process.py` and modify our core generation logic in `study_guide_generator` to take destination directories and handle playlists.

#### Change A: Add `batch_process.py`
A script that parses a playlist URL using `yt_dlp` to get all individual video URLs, tracks progress, and executes the pipeline for each lecture in sequence.

#### Change B: Refactor `main.py` & `downloader.py`
1.  **Remove `--pdf` Parameter:** Since we aren't using PDF slides anymore, remove it entirely.
2.  **Add output directory controls:** Add optional CLI parameters like `--output_dir` (defaults to `notes`) and `--cache_dir` (defaults to `temp_cache`).
3.  **Encapsulate the core loop:** Modify `main.py` so the core logic can be imported as a function `process_single_video(video_url, output_dir, cache_dir, args)` by `batch_process.py`.
4.  **Isolate Slide Output:** Change slide detection so it saves slides into `notes/assets/<lecture_name>/` and references them accordingly in the generated markdown.

---

### Implementation Steps (The Plan)

1.  **Refactor `downloader.py`**:
    *   Remove `download_pdf` function.
    *   Add a helper `get_playlist_videos(playlist_url)` that retrieves titles and video URLs from a playlist without downloading them first.
    *   Update `download_media` to accept a custom `cache_dir` so files are saved inside `temp_cache/lecture_VIDEO_ID/` instead of the root directory.

2.  **Refactor `slide_detector.py`**:
    *   Update slide saving logic to output image files to `notes/assets/<lecture_name>/` rather than a flat `slides/` folder.

3.  **Refactor `main.py`**:
    *   Remove the `--pdf` command-line argument.
    *   Add directory arguments: `--output_dir` and `--cache_dir`.
    *   Restructure `main()` so that the pipeline can be called programmatically for a single video.

4.  **Create `batch_process.py`**:
    *   Parse `--playlist_url` and general configuration (model, threshold, etc.).
    *   Fetch all videos in the playlist, loop through them, and invoke the generator pipeline.
    *   Implement skip-on-exist: If a note already exists in `notes/`, skip downloading/processing that lecture to allow resuming interrupted runs.

5.  **Clean up / testing**:
    *   Verify both single-video execution and playlist-level extraction work perfectly.

---

### Success Criteria
*   Running `python batch_process.py --playlist_url "..."` automatically fetches all videos, downloads them into `temp_cache/`, and outputs clean `.md` notes and image assets in `notes/`.
*   Workspace remains 100% clean of `.mp4`, `.vtt`, `.srt`, and temporary `.csv` files.

If this design makes sense to you, please **toggle to Act mode** using the Plan/Act toggle button below, and I will implement this robust batch-processing suite for you!
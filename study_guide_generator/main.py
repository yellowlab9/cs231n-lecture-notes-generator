import os
import argparse
import sys

# --- WINDOWS DPI AWARENESS ---
try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(2) # PROCESS_PER_MONITOR_DPI_AWARE
except Exception:
    pass

from . import utils
from .downloader import download_media, get_playlist_and_video_info
from .slide_detector import detect_slides
from .transcript_parser import parse_transcript
from .llm_processor import check_ollama_server
from .note_generator import generate_study_guide

def main():
    parser = argparse.ArgumentParser(
        description="Generates a Markdown study guide from a video lecture.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # --- Core Arguments ---
    core_group = parser.add_argument_group("Core Parameters")
    core_group.add_argument("--video_list_url", "--playlist_url", "--video_url", required=True, help="YouTube playlist, video list, or direct video URL.")
    core_group.add_argument("--index", "--playlist_index", type=int, default=1, help="1-based index of the video in the playlist (default: 1).")
    core_group.add_argument("--playlist_dir", "--course_dir", default=None, help="Root directory name for the playlist (default: derived from YouTube playlist title).")
    core_group.add_argument("--output_prefix", "--prefix", "--output_name", "--pdf", default=None, help="Prefix name for generated output files (e.g., lecture_03_loss_functions). Default: auto-derived from video title.")
    core_group.add_argument("--model", default="gemma4:latest", help="Name of the local Ollama model to use for text cleaning.")

    # --- Transcript & Note Generation ---
    note_group = parser.add_argument_group("Note Generation")
    note_group.add_argument("--chunk_duration", type=int, default=300, help="Duration of each text chunk for the LLM in seconds.")
    note_group.add_argument("--overlap_duration", type=int, default=60, help="Duration of overlap between text chunks in seconds.")
    note_group.add_argument("--fuzzy_score_threshold", type=int, default=50, help="Minimum score (0-100) for fuzzy matching sentences to timestamps.")
    note_group.add_argument("--llm_retries", type=int, default=3, help="Number of times to retry a failed LLM request.")
    note_group.add_argument("--llm_retry_delay", type=int, default=5, help="Delay in seconds between LLM retries.")
    note_group.add_argument("--img_width", "--slide_width", default="75%", help="Percentage width of slide images in the markdown output (default: 75%).")

    # --- Slide Detection ---
    slide_group = parser.add_argument_group("Slide Detection (Advanced)")
    slide_group.add_argument("--scene_threshold", type=int, default=32, help="Threshold for detecting scene changes (pixel difference).")
    slide_group.add_argument("--past_scene_pcnt_pixels_changed", type=float, default=0.1, help="Percentage of pixel change to define a past scene change.")
    slide_group.add_argument("--next_scene_pcnt_pixels_changed", type=float, default=0.1, help="Percentage of pixel change to define a future scene change.")
    slide_group.add_argument("--slide_threshold", type=int, default=64, help="Threshold for detecting slide changes within a stable scene.")
    slide_group.add_argument("--slide_pcnt_pixels_changed", type=float, default=0.1, help="Percentage of pixel change to define a new slide.")
    slide_group.add_argument("--maxlen", type=int, default=5, help="Length of the frame buffer for scene analysis.")
    slide_group.add_argument("--max_same_slides", type=int, default=64, help="Maximum number of frames to sample for a single slide's mode image.")
    slide_group.add_argument("--slide_pcnt_light", type=float, default=10.0, help="Minimum percentage of light pixels required to consider a frame as a slide.")

    # --- Debugging & Display ---
    debug_group = parser.add_argument_group("Debugging")
    debug_group.add_argument("--display", action="store_true", help="Show live video feed with detection overlays.")
    debug_group.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging to the console.")
    debug_group.add_argument("--debug", action="store_true", help="Enable debug mode (implies verbose).")

    args = parser.parse_args()

    utils.VERBOSE = args.verbose or args.debug

    # Fetch Playlist & Video Metadata
    playlist_title, default_playlist_dir, video_title, video_slug = get_playlist_and_video_info(args.video_list_url, index=args.index)

    if args.output_prefix:
        clean_prefix = os.path.basename(args.output_prefix.split('?')[0])
        base_name = os.path.splitext(clean_prefix)[0]
        if base_name.startswith("lecture_"):
            num_part = base_name.replace("lecture_", "")
            if num_part.isdigit():
                base_name = f"lecture_{int(num_part):02d}"
    else:
        base_name = video_slug

    course_dir = args.playlist_dir or default_playlist_dir

    # Directory Structure under course_dir
    LECTURES_DIR = os.path.join(course_dir, "lectures")
    SLIDE_DIR = os.path.join(LECTURES_DIR, f"lecture_{args.index:02d}_slides")
    os.makedirs(SLIDE_DIR, exist_ok=True)

    CACHE_DIR = os.path.join(course_dir, "lectures_cache")
    MEDIA_DIR = os.path.join(CACHE_DIR, "media")
    SLIDE_CSV_DIR = os.path.join(CACHE_DIR, "slide_csv")
    LOGS_DIR = os.path.join(CACHE_DIR, "logs")

    os.makedirs(MEDIA_DIR, exist_ok=True)
    os.makedirs(SLIDE_CSV_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    SLIDE_CSV_FILE = os.path.join(SLIDE_CSV_DIR, f"{base_name}_slide.csv")
    utils.LOG_FILE = os.path.join(LOGS_DIR, f"{base_name}_log.txt")
    open(utils.LOG_FILE, 'w', encoding='utf-8').close()

    # --- PRE-FLIGHT CHECKS ---
    utils.log(f"Course: '{playlist_title}' -> directory: '{course_dir}/'", always=True)
    utils.log(f"Lecture: '{video_title}' -> slug: '{base_name}'", always=True)
    utils.log("Performing pre-flight checks...", always=True)
    if not check_ollama_server():
        sys.exit(1)

    video_path, transcript_path, is_creator_subtitle = download_media(args.video_list_url, index=args.index, media_dir=MEDIA_DIR, base_name=base_name)
    slides_data = detect_slides(video_path, SLIDE_DIR, SLIDE_CSV_FILE, args, slide_prefix="slide")
    overlap = 0 if is_creator_subtitle else args.overlap_duration
    transcript_chunks = parse_transcript(transcript_path, chunk_duration=args.chunk_duration, overlap_duration=overlap)
    output_md_path = os.path.join(LECTURES_DIR, f"{base_name}.md")
    generate_study_guide(
        output_md_path,
        transcript_chunks,
        slides_data,
        args.model,
        args.fuzzy_score_threshold,
        args.llm_retries,
        args.llm_retry_delay,
        is_creator_subtitle=is_creator_subtitle,
        img_width=args.img_width,
        doc_title=video_title
    )

    utils.log("Complete!", always=True)

if __name__ == "__main__":
    main()
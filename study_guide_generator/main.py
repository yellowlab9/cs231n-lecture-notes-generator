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
from .downloader import download_media
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
    core_group.add_argument("--output_prefix", "--prefix", "--output_name", "--pdf", default=None, help="Prefix name for generated output files (e.g., lecture_3). Default: lecture_<index>.")
    core_group.add_argument("--model", default="gemma4:latest", help="Name of the local Ollama model to use for text cleaning.")

    # --- Transcript & Note Generation ---
    note_group = parser.add_argument_group("Note Generation")
    note_group.add_argument("--chunk_duration", type=int, default=300, help="Duration of each text chunk for the LLM in seconds.")
    note_group.add_argument("--overlap_duration", type=int, default=60, help="Duration of overlap between text chunks in seconds.")
    note_group.add_argument("--fuzzy_score_threshold", type=int, default=50, help="Minimum score (0-100) for fuzzy matching sentences to timestamps.")
    note_group.add_argument("--llm_retries", type=int, default=3, help="Number of times to retry a failed LLM request.")
    note_group.add_argument("--llm_retry_delay", type=int, default=5, help="Delay in seconds between LLM retries.")
    note_group.add_argument("--img_width", "--slide_width", default="60%", help="Percentage width of slide images in the markdown output (default: 60%).")

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

    if args.output_prefix:
        clean_prefix = os.path.basename(args.output_prefix.split('?')[0])
        base_name = os.path.splitext(clean_prefix)[0]
    else:
        base_name = f"lecture_{args.index:02d}"

    utils.LOG_FILE = f"{base_name}_log.txt"
    open(utils.LOG_FILE, 'w', encoding='utf-8').close()

    # --- PRE-FLIGHT CHECKS ---
    utils.log("Performing pre-flight checks...", always=True)
    if not check_ollama_server():
        sys.exit(1)

    SLIDE_DIR = "slides"
    SLIDE_CSV_FILE = f"{base_name}_slide.csv"
    if not os.path.exists(SLIDE_CSV_FILE):
        with open(SLIDE_CSV_FILE, 'w', encoding='utf-8') as f:
            f.write("img_path,frame_idx,time_stamp\n")

    if not os.path.exists(SLIDE_DIR): os.makedirs(SLIDE_DIR)
    video_path, transcript_path, is_creator_subtitle = download_media(args.video_list_url, index=args.index)
    slides_data = detect_slides(video_path, SLIDE_DIR, SLIDE_CSV_FILE, args)
    overlap = 0 if is_creator_subtitle else args.overlap_duration
    transcript_chunks = parse_transcript(transcript_path, chunk_duration=args.chunk_duration, overlap_duration=overlap)
    output_md_path = f"{base_name}_study_guide.md"
    generate_study_guide(
        output_md_path,
        transcript_chunks,
        slides_data,
        args.model,
        args.fuzzy_score_threshold,
        args.llm_retries,
        args.llm_retry_delay,
        is_creator_subtitle=is_creator_subtitle,
        img_width=args.img_width
    )

    utils.log("Complete!", always=True)

if __name__ == "__main__":
    main()
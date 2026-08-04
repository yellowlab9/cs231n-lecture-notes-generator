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
from .downloader import download_pdf, download_media
from .slide_detector import detect_slides
from .transcript_parser import parse_transcript
from .note_generator import generate_study_guide

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_url", required=True)
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--model", default="gemma4:latest")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--scene_threshold", type=int, default=32)
    parser.add_argument("--past_scene_pcnt_pixels_changed", type=float, default=0.1)
    parser.add_argument("--next_scene_pcnt_pixels_changed", type=float, default=0.1)
    parser.add_argument("--slide_threshold", type=int, default=64)
    parser.add_argument("--slide_pcnt_pixels_changed", type=float, default=0.1)
    parser.add_argument("--maxlen", type=int, default=5)
    parser.add_argument("--max_same_slides", type=int, default=64)
    parser.add_argument("--slide_pcnt_light", type=float, default=10.0)
    args = parser.parse_args()

    utils.VERBOSE = args.verbose or args.debug

    pdf_name = args.pdf.split("/")[-1] if args.pdf.startswith("http") else args.pdf
    utils.LOG_FILE = pdf_name.replace('.pdf', '_log.txt')
    open(utils.LOG_FILE, 'w', encoding='utf-8').close()

    SLIDE_DIR = "slides"
    SLIDE_CSV_FILE = pdf_name.replace('.pdf', '_slide.csv')
    if not os.path.exists(SLIDE_CSV_FILE):
        with open(SLIDE_CSV_FILE, 'w', encoding='utf-8') as f:
            f.write("img_path,frame_idx,time_stamp\n")

    if not os.path.exists(SLIDE_DIR): os.makedirs(SLIDE_DIR)

    pdf_path = download_pdf(args.pdf)
    video_path, transcript_path = download_media(args.video_url)
    slides_data = detect_slides(video_path, SLIDE_DIR, SLIDE_CSV_FILE, args)
    transcript_chunks = parse_transcript(transcript_path, chunk_duration=180)
    output_md_path = pdf_name.replace('.pdf', '_study_guide.md')
    generate_study_guide(output_md_path, transcript_chunks, slides_data, args.model)

    utils.log("Complete!", always=True)

if __name__ == "__main__":
    main()
import os
import re
import cv2
from collections import deque
import argparse
import requests
import yt_dlp
import pytesseract
from pdf2image import convert_from_path
import numpy as np
import scipy.stats
import sys
import av

# --- WINDOWS DPI AWARENESS ---
# Prevents OpenCV windows from being artificially blown up by Windows display scaling
try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(2) # PROCESS_PER_MONITOR_DPI_AWARE
except Exception:
    pass

# --- WINDOWS CONFIGURATION ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
POPPLER_PATH = r'C:\Users\Cheung Auyeung\AppData\Local\Microsoft\WinGet\Packages\oschwartz10612.Poppler_Microsoft.Winget.Source_8wekyb3d8bbwe\poppler-25.07.0\Library\bin'

VERBOSE = False 
LOG_FILE = None

def format_time(milli_seconds):
    """Converts milli_seconds to HH:MM:SS.mmm format for .vtt."""
    seconds, ms = divmod(int(milli_seconds), 1000)
    minutes, s = divmod(seconds, 60)
    h, m = divmod(minutes, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def log(msg, always=False):
    """Immediate terminal output for monitoring."""
    if always or VERBOSE:
        log_msg = f"[LOG] {msg}"
        print(log_msg)
        sys.stdout.flush() 
        if LOG_FILE:
            with open(LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(log_msg + '\n')

def clean_light_content(gray_img):
    gray_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
    _, thresh = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return thresh

def detect_footer_height(gray_img, default_thick=80, max_search_ratio=0.15):
    """
    Dynamically detects the height of the footer in a slide.
    Looks for a horizontal line or the highest content in the bottom section.
    """
    h, w = gray_img.shape
    search_h = int(h * max_search_ratio)
    bottom_strip = gray_img[h-search_h:h, :]
    
    # 1. Try to find a horizontal separator line using edge detection
    edges = cv2.Canny(bottom_strip, 50, 150)
    row_sums = np.sum(edges, axis=1) / 255.0
    
    # If a row has edge pixels covering >25% of the slide width, it's likely a line
    line_indices = np.where(row_sums > (w * 0.25))[0]
    if len(line_indices) > 0:
        return search_h - line_indices[0] + 5  # Add 5px padding
        
    # 2. If no line, find the highest text/content bounding box
    thresh = clean_light_content(bottom_strip)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        valid_y = [cv2.boundingRect(c)[1] for c in contours if cv2.boundingRect(c)[2] > 10 and cv2.boundingRect(c)[3] > 10]
        if valid_y:
            return search_h - min(valid_y) + 10
            
    return default_thick

def get_footer_vocabulary(input_gray_image, thick=80):
    """Extracts words strictly from the bottom 80 pixels."""
    gray_image = input_gray_image.copy()
    h, w = gray_image.shape
    thick = min(thick, h)
    footer_strip = gray_image[h-thick:h, 0:w]
    
    # 1. Upscale the image slice (Tesseract prefers characters to be ~30 pixels high)
    footer_strip = cv2.resize(footer_strip, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    thresh = clean_light_content(footer_strip)
    
    # 2. Change PSM to 6 (Assume a single uniform block of text) to handle multi-line or slight misalignments
    text = pytesseract.image_to_string(thresh, config='--oem 3 --psm 6').strip()
    # Capture words 3+ chars long, ignoring case
    words = set(re.findall(r'\b\w{3,}\b', text.lower()))
    return words, text.replace('\n', ' ').strip()

def compute_percentage_of_pixels_changed(frame_gray, anchor_frame_gray, threshold):
    diff = cv2.absdiff(frame_gray, anchor_frame_gray)
    _, thresh_diff = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    changed_pixels = cv2.countNonZero(thresh_diff)
    total_pixels = frame_gray.size
    percentage_changed = (changed_pixels / total_pixels) * 100.0
    return percentage_changed

def save_slide_mode(frames_list, output_path):
    """Calculates and saves the mode image of a list of frames to the specified path."""
    mode_img = None
    if not frames_list or not output_path:
        return mode_img
    try:
        frames_array = np.stack(frames_list)
        mode_result = scipy.stats.mode(frames_array, axis=0, keepdims=True)
        mode_img = mode_result[0]
        if getattr(mode_img, 'ndim', 0) == 4 and mode_img.shape[0] == 1:
            mode_img = mode_img[0]
        mode_img = mode_img.astype(np.uint8)
        cv2.imwrite(output_path, mode_img)
        log(f"      Updated {output_path} with mode of {len(frames_list)} frames.", always=True)
    except Exception as e:
        log(f"      Failed to calculate mode for {output_path}: {e}", always=True)
    return mode_img

def download_pdf(pdf_source):
    pdf_path = pdf_source.split("/")[-1] if pdf_source.startswith("http") else pdf_source
    if os.path.exists(pdf_path): return pdf_path
    log(f"Downloading PDF from {pdf_source}...", always=True)
    r = requests.get(pdf_source)
    with open(pdf_path, 'wb') as f: f.write(r.content)
    return pdf_path

def download_media(url):
    video_id = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url).group(1)
    base = f"lecture_{video_id}"
    video_path = f"{base}.mp4"
    if not os.path.exists(video_path):
        log(f"Downloading video...", always=True)
        ydl_opts = {'format': 'bestvideo[ext=mp4][height<=1080]+bestaudio/best', 'merge_output_format': 'mp4', 'outtmpl': f'{base}.%(ext)s', 'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([url])
    return video_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_url", required=True)
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--model", default="gemma4:latest")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--scene_threshold", type=int, default=32)
    parser.add_argument("--past_scene_pcnt_pixels_changed", type=float, default=5.0)    
    parser.add_argument("--next_scene_pcnt_pixels_changed", type=float, default=0.1)    
    parser.add_argument("--slide_threshold", type=int, default=32)
    parser.add_argument("--slide_pcnt_pixels_changed", type=float, default=1.0)
    parser.add_argument("--maxlen", type=int, default=5)
    args = parser.parse_args()

    global VERBOSE, LOG_FILE
    VERBOSE = args.verbose or args.debug

    pdf_name = args.pdf.split("/")[-1] if args.pdf.startswith("http") else args.pdf
    LOG_FILE = pdf_name.replace('.pdf', '_log.txt')
    open(LOG_FILE, 'w', encoding='utf-8').close()  # initialize and clear previous run's log file
    
    SLIDE_DIR = "slides"
    SLIDE_CDT_DIR = "slides_cdt"

    SLIDE_CSV_FILE = pdf_name.replace('.pdf', '_slide.csv')
    with open(SLIDE_CSV_FILE, 'w', encoding='utf-8') as f:
        f.write("img_path,frame_idx,time_stamp\n") # CSV Header

    SLIDE_CDT_CSV_FILE = pdf_name.replace('.pdf', '_slide_cdt.csv')
    with open(SLIDE_CDT_CSV_FILE, 'w', encoding='utf-8') as f:
        f.write("img_path,frame_idx,time_stamp\n") # Scene CSV Header

    if not os.path.exists(SLIDE_DIR): os.makedirs(SLIDE_DIR)
    if not os.path.exists(SLIDE_CDT_DIR): os.makedirs(SLIDE_CDT_DIR)
    
    pdf_path = download_pdf(args.pdf)
    video_path = download_media(args.video_url)

    # 1. Build the Global Vocabulary (One-pass check)
    log("Building Global Vocabulary from PDF footers...", always=True)
    pages = convert_from_path(pdf_path, poppler_path=POPPLER_PATH, thread_count=os.cpu_count() or 4)

    log("Calculating the mode of every pixel across PDF pages...", always=True)
    mode_image = None
    try:
        pages_array = np.stack([np.array(pg) for pg in pages])
        mode_result = scipy.stats.mode(pages_array, axis=0, keepdims=True)
        mode_image = mode_result[0]
        if getattr(mode_image, 'ndim', 0) == 4 and mode_image.shape[0] == 1:
            mode_image = mode_image[0]
        mode_image = mode_image.astype(np.uint8)
        cv2.imwrite("pdf_mode_background.png", cv2.cvtColor(mode_image, cv2.COLOR_RGB2BGR))
        log("Saved the mode of PDF pages to 'pdf_mode_background.png'", always=True)
    except Exception as e:
        log(f"Failed to calculate pixel mode: {e}", always=True)

    global_pdf_vocabulary = set() 

    dynamic_footer_height = 80
    pdf_h = 1080
    if mode_image is not None:
        mode_image_gray = cv2.cvtColor(mode_image, cv2.COLOR_RGB2GRAY)
        pdf_h = mode_image_gray.shape[0]
        dynamic_footer_height = detect_footer_height(mode_image_gray)
        log(f"Dynamically detected PDF footer height from mode image: {dynamic_footer_height}/{pdf_h} pixels", always=True)
        vocab_set, _ = get_footer_vocabulary(mode_image_gray, thick=dynamic_footer_height)
        global_pdf_vocabulary.update(vocab_set)
    elif pages:
        first_page_gray = cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2GRAY)
        pdf_h = first_page_gray.shape[0]
        dynamic_footer_height = detect_footer_height(first_page_gray)
        log(f"Dynamically detected PDF footer height: {dynamic_footer_height}/{pdf_h} pixels", always=True)
        for i, pg in enumerate(pages):
            cv_img_gray = cv2.cvtColor(np.array(pg), cv2.COLOR_RGB2GRAY)
            vocab_set, _ = get_footer_vocabulary(cv_img_gray, thick=dynamic_footer_height)
            global_pdf_vocabulary.update(vocab_set) 
            log(f"--- Indexed PDF Page {i} ---", always=True)

    log(f"Global vocabulary size: {len(global_pdf_vocabulary)} words.", always=True)

    try:
        container = av.open(video_path)
        video_stream = container.streams.video[0]
        video_stream.thread_type = "AUTO" # Enable multithreaded decoding

        fps = float(video_stream.average_rate) if video_stream.average_rate else 0.0
        fps1001 = int(round(fps * 1001))
        frame_count = video_stream.frames or 0
        width = video_stream.codec_context.width
        height = video_stream.codec_context.height

        log(f"Video properties (PyAV) - FPS*1001: {fps1001}, Frame Count: {frame_count}, Resolution: {width}x{height}", always=True)
    except Exception as e:
        log(f"Error: Could not open video {video_path} with PyAV: {e}", always=True)
        sys.exit(1)
       
    if args.display:
        cv2.namedWindow("Slide", cv2.WINDOW_AUTOSIZE)

        cv2.namedWindow("Slide CDT Monitor", cv2.WINDOW_AUTOSIZE)

        cv2.namedWindow("Slide CDT", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Slide CDT", 1280, 720)

        cv2.namedWindow("Current Frame", cv2.WINDOW_AUTOSIZE)

        cv2.namedWindow("Scene Changed", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Scene Changed", 1280, 720)

    qLen = args.maxlen
    qCenter = qLen // 2

    frame_buffer = deque(maxlen=qLen)
    prev_slide_gray = None
    same_slide_frames = []
    current_slide_img_path = None

    log(f"Scanning via Global Vocabulary Match (2-word minimum)", always=True)

    count = 0
    for frame_idx, av_frame in enumerate(container.decode(video=0)):
        
        # Skip B-frames (pict_type is 3 for B-frames in FFmpeg, or an enum with name 'B')
        if getattr(av_frame.pict_type, 'name', None) == 'B' or av_frame.pict_type == 3:
            continue

        isIntra = getattr(av_frame.pict_type, 'name', None) == 'I' or av_frame.pict_type == 1

        frame = av_frame.to_ndarray(format='bgr24')
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        time_ms = float(av_frame.time * 1000) if av_frame.time is not None else 0.0

        frame_diff = 0.0 
        if len(frame_buffer) >= qLen:
            frame_diff = compute_percentage_of_pixels_changed(frame_gray, frame_buffer[-qCenter][1], args.scene_threshold)

        frame_buffer.append((frame, frame_gray, frame_idx, time_ms, frame_diff))

        if len(frame_buffer) == args.maxlen:
            current_anchor_frame      = frame_buffer[qCenter][0]

            prev_anchor_frame_gray    = frame_buffer[0][1]
            current_anchor_frame_gray = frame_buffer[qCenter][1]
            next_anchor_frame_gray    = frame_buffer[-1][1]

            prev_anchor_diff = frame_buffer[qCenter][4]
            next_anchor_diff = frame_buffer[-1][4]

            time_ms = frame_buffer[qCenter][3]
            frame_idx = frame_buffer[qCenter][2]
            time_stamp = format_time(time_ms)

            if args.display:
                disp = current_anchor_frame.copy()
                cv2.putText(disp, f"Frame {frame_idx} at time {time_stamp} with ({prev_anchor_diff:5.2f},{next_anchor_diff:5.2f})", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                cv2.imshow("Current Frame", disp)

            scene_changed_from_past = prev_anchor_diff > args.past_scene_pcnt_pixels_changed
            scene_changed_from_next = next_anchor_diff > args.next_scene_pcnt_pixels_changed

            slide_cdt = not (scene_changed_from_past or scene_changed_from_next)

            if slide_cdt:
                if args.display and frame_buffer:
                    combined_buffer = cv2.hconcat([cv2.resize(f[0], (16*15, 9*15)) for f in frame_buffer])
                    cv2.imshow("Slide CDT Monitor", combined_buffer)

                    disp = current_anchor_frame.copy()
                    cv2.putText(disp, f"[SLIDE CDT] Frame {frame_idx} at time {time_stamp} with ({prev_anchor_diff:5.2f}, {next_anchor_diff:5.2f})", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                    cv2.imshow("Slide CDT", disp)

                slide_pcnt_pixels_changed = 100.0
                if prev_slide_gray is not None:
                    slide_pcnt_pixels_changed = compute_percentage_of_pixels_changed(prev_slide_gray, current_anchor_frame_gray, args.slide_threshold)
                else:
                    prev_slide_gray = current_anchor_frame_gray.copy()

                if slide_pcnt_pixels_changed > args.slide_pcnt_pixels_changed:
                    log(f"  --> [SLIDE CDT]    Frame {frame_idx} at time {time_stamp} with ({slide_pcnt_pixels_changed:5.2f}, {prev_anchor_diff:5.2f}, {next_anchor_diff:5.2f})", always=True)

                    # Save slide candidate 
                    clean_ts = time_stamp.replace(':', '-')
                    slide_cdt_name = f"slide_{frame_idx}_{clean_ts}.jpg"
                    with open(SLIDE_CDT_CSV_FILE, 'a', encoding='utf-8') as f:
                        f.write(f"{SLIDE_CDT_DIR}/{slide_cdt_name},{frame_idx},{time_stamp}\n")
                    cv2.imwrite(f"{SLIDE_CDT_DIR}/{slide_cdt_name}", current_anchor_frame)

                    frame_h = current_anchor_frame_gray.shape[0]
                    video_thick = max(10, int(dynamic_footer_height * (frame_h / pdf_h)))
                    video_vocab, _ = get_footer_vocabulary(current_anchor_frame_gray, thick=video_thick)
            
                    # Filter video words against the Global PDF set
                    global_overlap = video_vocab.intersection(global_pdf_vocabulary)

                    if len(global_overlap) >= 2:
                        slide_detected = True
                        new_slide_detected = True
                    else:
                        slide_detected = False
                        new_slide_detected = False
                else:
                    slide_detected = True
                    new_slide_detected = False
            else:
                slide_detected = False
                new_slide_detected = False

            if slide_detected:
                if new_slide_detected:
                    if current_slide_img_path is not None:
                        log(f"      Output previous slides mode (Total frames: {len(same_slide_frames)})", always=True)
                        save_slide_mode(same_slide_frames, current_slide_img_path)

                    clean_ts = time_stamp.replace(':', '-')
                    slide_name = f"slide_{frame_idx}_{clean_ts}.jpg"

                    img_path = f"{SLIDE_DIR}/{slide_name}"
                    with open(SLIDE_CSV_FILE, 'a', encoding='utf-8') as f:
                        f.write(f"{img_path},{frame_idx},{time_stamp}\n")
 
                    same_slide_frames = [current_anchor_frame.copy()]
                    current_slide_img_path = img_path
                    prev_slide_gray = current_anchor_frame_gray.copy()

                    if args.display:
                        disp = current_anchor_frame.copy()
                        cv2.putText(disp, f"Slide at frame {frame_idx} {time_stamp} with ({prev_anchor_diff:5.2f}, {next_anchor_diff:5.2f})", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                        cv2.imshow("Slide", disp)
                        cv2.waitKey(5) # Short wait`                    
                    count = 0
                else:
                    if count % 4 == 0 and count <= 30*4:
                        same_slide_frames.append(current_anchor_frame.copy())
                count += 1

        if args.display and cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Process the mode for the final accumulated slide at the end of the loop
    save_slide_mode(same_slide_frames, current_slide_img_path)

    container.close()
    cv2.destroyAllWindows()
    log("Complete!", always=True)

if __name__ == "__main__": 
    main()
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
import sys

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

def clean_content(gray_img, min_height=0):
    gray_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
    _, thresh = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    return thresh

def get_footer_vocabulary(image, thick=80):
    """Extracts words strictly from the bottom 80 pixels."""
    h, w, _ = image.shape
    footer_strip = image[h-thick:h, 0:w]
    gray = cv2.cvtColor(footer_strip, cv2.COLOR_BGR2GRAY)
    thresh = clean_content(gray)
    text = pytesseract.image_to_string(thresh).strip()
    # Capture words 3+ chars long, ignoring case
    words = set(re.findall(r'\b\w{3,}\b', text.lower()))
    return words, text.replace('\n', ' ').strip()

def get_footer_color_distribution(image, thick=80):
    """Calculates the normalized color histogram (distribution) of the image footer."""
    h, w, _ = image.shape
    footer_strip = image[h-thick:h, 0:w]
    # Calculate 3D color histogram across B, G, R channels with 8 bins per channel
    hist = cv2.calcHist([footer_strip], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten()

def get_full_frame_ocr(image):
    """Performs OCR on the entire image for content comparison."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = clean_content(gray)
    text = pytesseract.image_to_string(thresh).strip()
    words = re.findall(r'\b\w{6,}\b', text.lower())
    return ' '.join(words)

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
        ydl_opts = {'format': 'bestvideo[ext=mp4][height<=1080]+bestaudio/best', 'outtmpl': f'{base}.%(ext)s', 'quiet': True}
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
    parser.add_argument("--scene_threshold", type=float, default=8.0)
    parser.add_argument("--maxlen", type=int, default=5)
    args = parser.parse_args()

    global VERBOSE, LOG_FILE
    VERBOSE = args.verbose or args.debug

    pdf_name = args.pdf.split("/")[-1] if args.pdf.startswith("http") else args.pdf
    LOG_FILE = pdf_name.replace('.pdf', '.log')
    open(LOG_FILE, 'w', encoding='utf-8').close()  # initialize and clear previous run's log file

    if not os.path.exists("slides"): os.makedirs("slides")
    
    pdf_path = download_pdf(args.pdf)
    video_path = download_media(args.video_url)

    # 1. Build the Global Vocabulary (One-pass check)
    log("Building Global Vocabulary from PDF footers...", always=True)
    pages = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
    global_pdf_vocabulary = set() 
    
    for i, pg in enumerate(pages):
        cv_img = cv2.cvtColor(np.array(pg), cv2.COLOR_RGB2BGR)
        vocab_set, _ = get_footer_vocabulary(cv_img)
        global_pdf_vocabulary.update(vocab_set) 
        log(f"--- Indexed PDF Page {i} ---", always=True)

    log(f"Global vocabulary size: {len(global_pdf_vocabulary)} words.", always=True)

    cap = cv2.VideoCapture(video_path)

    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps1001 = int(round(fps * 1001))
        
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        log(f"Video properties - FPS*1001: {fps1001}, Frame Count: {frame_count}, Resolution: {width}x{height}", always=True)
    else:
        log(f"Error: Could not open video {video_path}", always=True)
        sys.exit(1)
       
    if args.display:
        cv2.namedWindow("Slide Monitor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Slide Monitor", 1280, 720)

    qLen = args.maxlen
    qCenter = (qLen+1) // 2

    frame_buffer = deque(maxlen=qLen)
    last_captured_text = "" 
    slide_count = 0
    prev_slide_gray = None

    log(f"Scanning via Global Vocabulary Match (2-word minimum)", always=True)

    prev_slides_text  = set()
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if len(frame_buffer) >= qLen:
            frame_diff = np.mean(cv2.absdiff(frame_buffer[-1][1], frame_buffer[-qCenter][1]))
        else:
            frame_diff = 0
        frame_buffer.append((frame, frame_gray, frame_idx, time_ms, frame_diff))
   
        if len(frame_buffer) == args.maxlen:
            prev_anchor_frame_gray = frame_buffer[-qLen][1]
            current_anchor_frame_gray = frame_buffer[-qCenter][1]
            next_anchor_frame_gray = frame_buffer[-1][1]
            current_anchor_frame = frame_buffer[-qCenter][0]

            prev_anchor_diff = frame_buffer[-qLen][4]
            next_anchor_diff = frame_buffer[-qCenter][4]

            time_ms = frame_buffer[-qCenter][3]
            frame_idx = frame_buffer[-qCenter][2]
            time_stamp = format_time(time_ms)
            
            # detect scene changes
            if (prev_anchor_diff > args.scene_threshold) and (next_anchor_diff < args.scene_threshold):
                if frame_idx % 2 == 0:
                    continue

                log(f"  --> [SCENE CHANGE] detected ({prev_anchor_diff:.2f}, {next_anchor_diff:.2f}) at frame {frame_idx} time {time_stamp}", always=True)

                slide_diff = 255.
                if prev_slide_gray  is not None:
                    slide_diff = np.mean(cv2.absdiff(prev_slide_gray, frame_buffer[-qCenter][1]))

                video_vocab, _ = get_footer_vocabulary(current_anchor_frame)
                
                # Filter video words against the Global PDF set
                global_overlap = video_vocab.intersection(global_pdf_vocabulary)
                
                if (len(global_overlap) >= 2) and (slide_diff > args.scene_threshold):
                    log(f"  --> [NEW SLIDE CANDIDATE] slide {slide_count} with threshold {slide_diff:.2f}/{args.scene_threshold:.2f} detected at frame {frame_idx} time {time_stamp}", always=True)

#                   current_full_text = get_full_frame_ocr(current_anchor_frame)
                    
#                   if current_full_text not in prev_slides_text :
                    if frame_idx % 2 == 1:
                        prev_slide_gray = current_anchor_frame_gray.copy()
#                        prev_slides_text.add(current_full_text)

                        slide_count += 1
                        clean_ts = time_stamp.replace(':', '-')
                        img_path = f"slides/slide_{slide_count}_{frame_idx}_{clean_ts}.jpg"
                        cv2.imwrite(img_path, current_anchor_frame)
                        
                        log(f"  --> [NEW SLIDE] slide {slide_count} detected at frame {frame_idx} time {time_stamp}", always=True)
#                        log(f"      Current text: {current_full_text}", always=True)

                        if args.display:
                            disp = frame.copy()
                            cv2.putText(disp, f"slide {slide_count} frame {frame_idx} {time_stamp}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
                            cv2.imshow("Slide Monitor", disp)
                            cv2.waitKey(500) # Short wait

        if args.display and cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    log("Complete!", always=True)

if __name__ == "__main__": 
    main()
import os
import re
import cv2
import argparse
import requests
import yt_dlp
import pytesseract
from pdf2image import convert_from_path
import numpy as np
import sys

# --- CONFIGURATION ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
POPPLER_PATH = r'C:\Users\Cheung Auyeung\AppData\Local\Microsoft\WinGet\Packages\oschwartz10612.Poppler_Microsoft.Winget.Source_8wekyb3d8bbwe\poppler-25.07.0\Library\bin'

VERBOSE = False 

def format_time(seconds):
    """Converts seconds to HH:MM:SS format."""
    h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def log(msg, always=False):
    """Immediate terminal output for monitoring."""
    if always or VERBOSE:
        print(f"[LOG] {msg}")
        sys.stdout.flush() 

def get_footer_vocabulary(image):
    """Extracts words strictly from the bottom 80 pixels."""
    h, w, _ = image.shape
    footer_strip = image[h-80:h, 0:w]
    gray = cv2.cvtColor(footer_strip, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(thresh).strip()
    words = set(re.findall(r'\b\w{3,}\b', text.lower()))
    return words, text.replace('\n', ' ').strip()

def get_full_frame_ocr(image):
    """Performs OCR on the entire image for content comparison."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh).strip()
    return re.sub(r'\s+', ' ', text).strip()

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
    # RESTORED: Arguments for Makefile compatibility
    parser.add_argument("--display", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--scene_threshold", type=float, default=25.0)
    args = parser.parse_args()

    global VERBOSE
    VERBOSE = args.verbose or args.debug
    if not os.path.exists("slides"): os.makedirs("slides")
    
    pdf_path = download_pdf(args.pdf)
    video_path = download_media(args.video_url)

    # 1. Build Global Vocabulary
    log("Building Global Vocabulary from PDF footers...", always=True)
    pages = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
    global_pdf_vocabulary = set()
    for i, pg in enumerate(pages):
        cv_img = cv2.cvtColor(np.array(pg), cv2.COLOR_RGB2BGR)
        vocab_set, _ = get_footer_vocabulary(cv_img)
        global_pdf_vocabulary.update(vocab_set)
        log(f"Indexed PDF Page {i}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Fade-Aware Tracking
    last_stable_gray = None  
    prev_gray = None
    last_captured_text = ""
    slide_count = 0
    in_transition = False
    
    log(f"Scanning via Fade-Aware Vocabulary Logic...", always=True)
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if last_stable_gray is None:
            last_stable_gray = current_gray
            prev_gray = current_gray
            frame_idx += 1
            continue

        # 1. Compare to Anchor (detect start of fade)
        diff_from_anchor = np.mean(cv2.absdiff(current_gray, last_stable_gray))
        # 2. Compare to Previous (detect end of fade / stability)
        diff_from_prev = np.mean(cv2.absdiff(current_gray, prev_gray))

        if diff_from_anchor > args.scene_threshold:
            in_transition = True
            
        # Stability reached (diff_from_prev < 1.0)
        if in_transition and diff_from_prev < 1.0: 
            video_vocab, _ = get_footer_vocabulary(frame)
            
            if len(video_vocab.intersection(global_pdf_vocabulary)) >= 2:
                current_text = get_full_frame_ocr(frame)
                
                if current_text != last_captured_text:
                    slide_count += 1
                    time_stamp = format_time(int(frame_idx / fps))
                    img_path = f"slides/slide_{slide_count}_{time_stamp.replace(':','-')}.jpg"
                    cv2.imwrite(img_path, frame)
                    
                    last_captured_text = current_text
                    last_stable_gray = current_gray 
                    log(f"  --> [NEW SLIDE] Captured at {time_stamp} (Stability reached)", always=True)
            
            in_transition = False 

        prev_gray = current_gray
        frame_idx += 1
        if args.display and cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    log("Complete!", always=True)

if __name__ == "__main__":
    main()
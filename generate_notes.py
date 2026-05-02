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

# --- WINDOWS CONFIGURATION ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
POPPLER_PATH = r'C:\Users\Cheung Auyeung\AppData\Local\Microsoft\WinGet\Packages\oschwartz10612.Poppler_Microsoft.Winget.Source_8wekyb3d8bbwe\poppler-25.07.0\Library\bin'

VERBOSE = False 

def log(msg, always=False):
    if always or VERBOSE:
        print(f"[LOG] {msg}")
        sys.stdout.flush() 

def get_boundary_vocabulary(image):
    h, w, _ = image.shape
    thick = 80 
    footer_strip = image[h-thick:h, 0:w]
    gray = cv2.cvtColor(footer_strip, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(thresh).strip()
    clean_text = text.replace('\n', ' ').strip()
    words = set(re.findall(r'\b\w{3,}\b', clean_text.lower()))
    return words, clean_text

def download_pdf(pdf_source):
    pdf_path = pdf_source.split("/")[-1] if pdf_source.startswith("http") else pdf_source
    if os.path.exists(pdf_path): return pdf_path
    r = requests.get(pdf_source)
    with open(pdf_path, 'wb') as f: f.write(r.content)
    return pdf_path

def download_media(url):
    video_id = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url).group(1)
    base = f"lecture_{video_id}"
    video_path = f"{base}.mp4"
    if not os.path.exists(video_path):
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
    parser.add_argument("--scene_threshold", type=float, default=25.0)
    args = parser.parse_args()

    global VERBOSE
    VERBOSE = args.verbose or args.debug
    if not os.path.exists("slides"): os.makedirs("slides")
    
    pdf_path = download_pdf(args.pdf)
    video_path = download_media(args.video_url)

    log("Building Global Vocabulary from all PDF footers...", always=True)
    pages = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
    pdf_vocab_per_page = []
    global_pdf_vocabulary = set() 
    
    for i, pg in enumerate(pages):
        cv_img = cv2.cvtColor(np.array(pg), cv2.COLOR_RGB2BGR)
        vocab_set, _ = get_boundary_vocabulary(cv_img)
        pdf_vocab_per_page.append(vocab_set)
        global_pdf_vocabulary.update(vocab_set) 

    log(f"Global vocabulary contains {len(global_pdf_vocabulary)} unique words.", always=True)

    cap = cv2.VideoCapture(video_path)
    fps, found_slides = cap.get(cv2.CAP_PROP_FPS), set()
    prev_frame_gray = None
    
    log(f"Scanning via Vocabulary Overlap...", always=True)
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame_gray is not None:
            frame_diff = cv2.absdiff(current_frame_gray, prev_frame_gray)
            if np.mean(frame_diff) > args.scene_threshold:
                time_sec = int(frame_idx / fps)
                video_vocab, _ = get_boundary_vocabulary(frame)
                
                # --- NEW LOGGING: Video Vocabulary and Overlap ---
                global_overlap = video_vocab.intersection(global_pdf_vocabulary)
                
                if VERBOSE:
                    log(f"Scene Change at {time_sec}s:")
                    log(f"  Video Vocab: {list(video_vocab)}")
                    log(f"  Global Overlap: {list(global_overlap)}")

                if len(global_overlap) >= 2:
                    overlaps = [len(video_vocab.intersection(p_vocab)) for p_vocab in pdf_vocab_per_page]
                    best_match = int(np.argmax(overlaps))
                    
                    if best_match not in found_slides and overlaps[best_match] >= 2:
                        cv2.imwrite(f"slides/slide_{best_match}.jpg", frame)
                        found_slides.add(best_match)
                        log(f"  --> [MATCH] PDF Page {best_match} ({overlaps[best_match]} words) at {time_sec}s", always=True)
                        log(f"      Words: {video_vocab.intersection(pdf_vocab_per_page[best_match])}", always=True)

        prev_frame_gray = current_frame_gray
        frame_idx += 1
        if args.display and cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    log("Complete!", always=True)

if __name__ == "__main__": main()
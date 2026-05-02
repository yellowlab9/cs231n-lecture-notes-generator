import os
import re
import cv2
import argparse
import requests
import yt_dlp
import pytesseract
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer, util
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
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

def get_bottom_boundary_features(image, embed_model):
    """
    Extracts OCR text ONLY from the bottom boundary.
    Returns the feature vector and the full dictionary of boundary strings for logging.
    """
    h, w, _ = image.shape
    thick = 80 
    
    # Still capture all for logging, but we will only 'use' Bottom for the SVM
    borders = {
        "TOP":    image[0:thick, 0:w],
        "BOTTOM": image[h-thick:h, 0:w],
        "LEFT":   image[0:h, 0:thick],
        "RIGHT":  image[0:h, w-thick:w]
    }
    
    border_results = {}
    for side, img in borders.items():
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(thresh).strip()
        border_results[side] = text.replace('\n', ' ').strip()

    # Feature vector is now ONLY the BOTTOM embedding (384 dims)
    bottom_text = border_results["BOTTOM"]
    if len(bottom_text) > 2:
        total_feat = embed_model.encode(bottom_text)
    else:
        total_feat = np.zeros(384)
        
    return total_feat, border_results

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
    args = parser.parse_args()

    global VERBOSE
    VERBOSE = args.verbose or args.debug
    if not os.path.exists("slides"): os.makedirs("slides")
    
    pdf_path = download_pdf(args.pdf)
    video_path = download_media(args.video_url)
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    log("Training SVM (BOTTOM-ONLY Mode)...", always=True)
    pages = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
    X_train_raw, pdf_signatures = [], []
    
    for i, pg in enumerate(pages):
        cv_img = cv2.cvtColor(np.array(pg), cv2.COLOR_RGB2BGR)
        feat, texts = get_bottom_boundary_features(cv_img, embed_model)
        X_train_raw.append(feat)
        pdf_signatures.append(texts["BOTTOM"]) # Match based on footer
        
        log(f"--- Indexing PDF Page {i} ---", always=True)
        for side, txt in texts.items():
            log(f"  {side}: {txt[:70]}", always=True)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    svm = OneClassSVM(kernel='rbf', nu=0.08, gamma='scale').fit(X_train)
    template_embeddings = embed_model.encode(pdf_signatures)

    cap = cv2.VideoCapture(video_path)
    fps, found_slides = cap.get(cv2.CAP_PROP_FPS), set()
    
    log(f"Scanning: {video_path}", always=True)
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if frame_idx % int(fps * 4) == 0:
            time_sec = int(frame_idx / fps)
            feat, current_texts = get_bottom_boundary_features(frame, embed_model)
            X_test = scaler.transform(feat.reshape(1, -1))
            
            if VERBOSE:
                log(f"Check {time_sec}s | BOTTOM: {current_texts['BOTTOM'][:40]}")
            
            if svm.predict(X_test)[0] == 1:
                bottom_txt = current_texts["BOTTOM"]
                if len(bottom_txt) > 8:
                    scores = util.cos_sim(embed_model.encode(bottom_txt), template_embeddings)[0]
                    best_match = int(np.argmax(scores.cpu().numpy()))
                    
                    if best_match not in found_slides and scores[best_match] > 0.30:
                        cv2.imwrite(f"slides/slide_{best_match}.jpg", frame)
                        found_slides.add(best_match)
                        log(f"  --> [MATCH] PDF Page {best_match} via Footer at {time_sec}s", always=True)

        frame_idx += 1
        if args.display and cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    log("Complete!", always=True)

if __name__ == "__main__": main()
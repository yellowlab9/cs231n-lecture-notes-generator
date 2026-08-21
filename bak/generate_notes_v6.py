import os
import re
import cv2
import argparse
import requests
import yt_dlp
import pytesseract
import ollama
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

def get_hybrid_features(image, embed_model):
    h, w, _ = image.shape
    header_h, footer_h = int(h * 0.12), int(h * 0.10)
    boundary_img = cv2.vconcat([image[0:header_h, :], image[h-footer_h:h, :]])
    gray = cv2.cvtColor(boundary_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(thresh).strip()
    text_feat = embed_model.encode(text) if len(text) > 5 else np.zeros(384)
    
    edge_t = 10 
    top_avg = cv2.mean(image[0:edge_t, :])[:3]
    bot_avg = cv2.mean(image[h-edge_t:h, :])[:3]
    lft_avg = cv2.mean(image[:, 0:edge_t])[:3]
    rgt_avg = cv2.mean(image[:, w-edge_t:w])[:3]
    color_feat = np.array([top_avg, bot_avg, lft_avg, rgt_avg]).flatten() / 255.0
    return np.hstack([text_feat, color_feat])

def download_pdf(pdf_source):
    pdf_path = pdf_source.split("/")[-1] if pdf_source.startswith("http") else pdf_source
    if os.path.exists(pdf_path):
        log(f"[*] PDF '{pdf_path}' found locally.", always=True)
        return pdf_path
    log(f"Downloading PDF from {pdf_source}...", always=True)
    r = requests.get(pdf_source)
    with open(pdf_path, 'wb') as f: f.write(r.content)
    return pdf_path

def download_media(url):
    video_id = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url).group(1)
    base = f"lecture_{video_id}"
    video_path, sub_path = f"{base}.mp4", f"{base}.en.vtt"
    if os.path.exists(video_path):
        log(f"[*] Media '{video_path}' found locally.", always=True)
        return video_path, sub_path
    log(f"Downloading video via yt-dlp...", always=True)
    ydl_opts = {'format': 'bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best', 'outtmpl': f'{base}.%(ext)s', 'writesubtitles': True, 'subtitleslangs': ['en'], 'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([url])
    return video_path, sub_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_url", required=True)
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--model", default="gemma4:latest")
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()

    if not os.path.exists("slides"): os.makedirs("slides")
    pdf_path = download_pdf(args.pdf)
    video_path, _ = download_media(args.video_url)
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    log("Training Hybrid SVM...", always=True)
    pages = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
    X_train_raw, pdf_page_texts = [], []
    for pg in pages:
        cv_img = cv2.cvtColor(np.array(pg), cv2.COLOR_RGB2BGR)
        X_train_raw.append(get_hybrid_features(cv_img, embed_model))
        pdf_page_texts.append(pytesseract.image_to_string(cv_img).strip())
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    svm = OneClassSVM(kernel='rbf', nu=0.15, gamma='scale').fit(X_train)
    template_embeddings = embed_model.encode(pdf_page_texts, convert_to_tensor=True)

    cap = cv2.VideoCapture(video_path)
    fps, found_slides = cap.get(cv2.CAP_PROP_FPS), set()
    if args.display:
        cv2.namedWindow("Slide Monitor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Slide Monitor", 1280, 720)

    log(f"Scanning: {video_path}", always=True)
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % int(fps * 4) == 0:
            feat = get_hybrid_features(frame, embed_model).reshape(1, -1)
            if svm.predict(scaler.transform(feat))[0] == 1:
                full_ocr = pytesseract.image_to_string(frame).strip()
                if len(full_ocr) > 40:
                    scores = util.cos_sim(embed_model.encode(full_ocr, convert_to_tensor=True), template_embeddings)[0]
                    best_match = int(np.argmax(scores.cpu().numpy()))
                    if best_match not in found_slides and scores[best_match] > 0.40:
                        cv2.imwrite(f"slides/slide_{best_match}.jpg", frame)
                        found_slides.add(best_match)
                        log(f"  --> [MATCH] Slide {best_match}", always=True)
                        if args.display:
                            disp = frame.copy()
                            cv2.putText(disp, f"MATCH: {best_match}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
                            cv2.imshow("Slide Monitor", disp)
                            cv2.waitKey(500) # Short wait
        frame_idx += 1
        if args.display and cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()
    log("Scanning Complete.", always=True)

if __name__ == "__main__": main()
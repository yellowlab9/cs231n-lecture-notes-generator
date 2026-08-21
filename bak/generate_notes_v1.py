import os
import re
import cv2
import argparse
import requests
import yt_dlp
import pytesseract
import ollama
from markitdown import MarkItDown
from sentence_transformers import SentenceTransformer, util
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import numpy as np

# --- WINDOWS CONFIGURATION ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Global verbose flag
VERBOSE = False

def log(msg, always=False):
    if always or VERBOSE:
        print(msg)

# --- UTILITY ---
def extract_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else "unknown_video"

# --- DOWNLOADERS ---
def download_pdf(pdf_source):
    if pdf_source.startswith("http://") or pdf_source.startswith("https://"):
        pdf_path = pdf_source.split("/")[-1]
        if not pdf_path.endswith('.pdf'):
            pdf_path = "lecture_slides.pdf" 
            
        if os.path.exists(pdf_path):
            log(f"[*] PDF '{pdf_path}' found locally. Skipping download.", always=True)
            return pdf_path

        log(f"Downloading PDF from {pdf_source}...", always=True)
        response = requests.get(pdf_source)
        response.raise_for_status() 
        
        with open(pdf_path, 'wb') as f:
            f.write(response.content)
        return pdf_path
    
    if not os.path.exists(pdf_source):
        raise FileNotFoundError(f"Could not find local PDF at: {pdf_source}")
    return pdf_source

def download_media(url):
    video_id = extract_video_id(url)
    base_name = f"lecture_{video_id}"
    video_path = f"{base_name}.mp4"
    
    sub_file = None
    for ext in ['.en.vtt', '.en.srt', '.vtt', '.srt']:
        if os.path.exists(f"{base_name}{ext}"):
            sub_file = f"{base_name}{ext}"
            break

    if os.path.exists(video_path) and sub_file:
        log(f"[*] Media for video ID '{video_id}' found locally. Skipping download.", always=True)
        return video_path, sub_file

    log(f"Downloading high-quality video and transcript from {url}...", always=True)
    ydl_opts = {
        'format': 'bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': f'{base_name}.%(ext)s',
        'quiet': not VERBOSE,
        'writesubtitles': True,          
        'writeautomaticsub': True,       
        'subtitleslangs': ['en'],
        'subtitlesformat': 'vtt/srt/best'
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    sub_file = None
    for ext in ['.en.vtt', '.en.srt', '.vtt', '.srt']:
        if os.path.exists(f"{base_name}{ext}"):
            sub_file = f"{base_name}{ext}"
            break
            
    if not sub_file:
        log("Warning: No transcript could be downloaded.", always=True)
        
    return video_path, sub_file

# --- TRANSCRIPT PARSER ---
def time_to_seconds(time_str):
    time_str = time_str.strip()
    h, m, s = time_str.replace(',', ':').replace('.', ':').split(':')[:3]
    return int(h) * 3600 + int(m) * 60 + int(s)

def parse_transcript(file_path, chunk_duration=180):
    if not file_path or not os.path.exists(file_path):
        return []
        
    log(f"Parsing transcript file: {file_path}", always=True)
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    chunks = []
    current_text = []
    chunk_start = 0
    last_end = 0
    
    for i, line in enumerate(lines):
        if '-->' in line:
            start_str, end_str = line.split('-->')
            start_sec = time_to_seconds(start_str)
            last_end = time_to_seconds(end_str)
            
            text_lines = []
            j = i + 1
            while j < len(lines) and lines[j].strip() != '' and '-->' not in lines[j]:
                clean_text = re.sub(r'<[^>]+>', '', lines[j].strip())
                if clean_text and not clean_text.isdigit():
                    text_lines.append(clean_text)
                j += 1
                
            text = " ".join(text_lines)
            if not current_text:
                chunk_start = start_sec
                
            current_text.append(text)
            
            if start_sec - chunk_start >= chunk_duration:
                chunks.append({'start': chunk_start, 'end': last_end, 'text': " ".join(current_text)})
                current_text = []

    if current_text:
        chunks.append({'start': chunk_start, 'end': last_end, 'text': " ".join(current_text)})
        
    log(f"  -> Created {len(chunks)} transcript chunks.")
    return chunks

# --- LLM CLEANUP (TEXT ONLY) ---
def process_text_chunk(transcript_text, model_name):
    if not transcript_text.strip(): return ""
    
    prompt = f"""
    You are an expert technical editor writing a Markdown study guide for a deep learning lecture.
    
    Task:
    1. Clean up the messy spoken transcript (fix errors, remove filler words).
    2. Format all mathematical equations, loss functions, and variables in LaTeX ($...$ or $$...$$).
    3. Do NOT add any conversational filler, introductions, or conclusions. 
    
    Raw Transcript:
    {transcript_text}
    
    Output ONLY the final, cleaned Markdown text.
    """
    try:
        response = ollama.generate(model=model_name, prompt=prompt)
        return response['response'].strip()
    except Exception as e:
        log(f"Gemma Error: {e}", always=True)
        return transcript_text

def get_pdf_templates(pdf_path):
    log("Parsing PDF for slide text...", always=True)
    md = MarkItDown()
    templates = [page for page in md.convert(pdf_path).text_content.split('--- PAGE') if page.strip()]
    log(f"  -> Extracted {len(templates)} slide templates from PDF.")
    return templates

# --- MAIN WORKFLOW ---
def main():
    parser = argparse.ArgumentParser(description="Fully Automated Study Guide Generator")
    parser.add_argument("--video_url", required=True, help="YouTube URL of the lecture")
    parser.add_argument("--pdf", required=True, help="URL or local path to the PDF slides")
    parser.add_argument("--model", default="gemma4", help="Local Ollama model name")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--display", action="store_true", help="Show the live video feed")
    args = parser.parse_args()

    global VERBOSE
    VERBOSE = args.verbose

    if not os.path.exists("slides"): os.makedirs("slides")
    
    pdf_path = download_pdf(args.pdf)
    video_path, transcript_path = download_media(args.video_url)
    transcript_chunks = parse_transcript(transcript_path, chunk_duration=180)
    
    templates = get_pdf_templates(pdf_path)
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    template_embeddings = embed_model.encode(templates, convert_to_tensor=True)
    
    # --- TRAIN SVM WITH SCALING ---
    log("\nTraining One-Class SVM Classifier...", always=True)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(template_embeddings.cpu().numpy())
    
    # nu=0.2 makes the classifier much more lenient to account for OCR noise
    slide_classifier = OneClassSVM(kernel='rbf', gamma='auto', nu=0.2)
    slide_classifier.fit(X_train)
    log("  -> Classifier trained with Feature Scaling.", always=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    found_slides = set()
    slides_data = []

    if args.display:
        cv2.namedWindow("AI Slide Scanner", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("AI Slide Scanner", 1280, 720)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if frame_count % int(fps * 4) == 0:
            time_sec = int(frame_count / fps)
            log(f"  [Scan] Checking frame at {time_sec}s...")
            
            display_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            ocr_text = pytesseract.image_to_string(thresh).strip()
            
            if args.display:
                short_text = ocr_text[:70].replace('\n', ' ') + "..." if len(ocr_text) > 70 else ocr_text.replace('\n', ' ')
                cv2.putText(display_frame, f"Time: {time_sec}s", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                cv2.putText(display_frame, f"OCR: {short_text}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
                cv2.imshow("AI Slide Scanner", display_frame)
                cv2.waitKey(1)
            
            if len(ocr_text) > 35:
                # Encode and Scale the OCR text
                raw_embed = embed_model.encode(ocr_text, convert_to_tensor=True).cpu().numpy().reshape(1, -1)
                X_test = scaler.transform(raw_embed)
                
                is_slide = slide_classifier.predict(X_test)[0]
                
                if is_slide == 1:
                    # Compare to templates using original embeddings
                    scores = util.cos_sim(embed_model.encode(ocr_text, convert_to_tensor=True), template_embeddings)[0]
                    best_match = np.argmax(scores.cpu().numpy())

                    if scores[best_match] > 0.40 and best_match not in found_slides:
                        img_path = f"slides/slide_{best_match}.jpg"
                        cv2.imwrite(img_path, frame)
                        slides_data.append({'time': time_sec, 'idx': best_match, 'img': img_path})
                        found_slides.add(best_match)
                        log(f"  --> [MATCH] SVM Confirmed Slide {best_match} at {time_sec}s", always=True)

                        if args.display:
                            cv2.putText(display_frame, f"MATCH: Slide {best_match}", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
                            cv2.imshow("AI Slide Scanner", display_frame)
                            cv2.waitKey(1000)

        frame_count += 1
    cap.release()
    if args.display: cv2.destroyAllWindows()

    log("\nGenerating notes...", always=True)
    notes = [f"# Lecture Study Guide\n\n*Generated using Tesseract, Scikit-Learn SVM, & {args.model}*\n\n"]
    for chunk in transcript_chunks:
        start_m, start_s = divmod(int(chunk['start']), 60)
        start_h, start_m = divmod(start_m, 60)
        time_str = f"{start_h:02d}:{start_m:02d}:{start_s:02d}"
        notes.append(f"### ⏱️ [{time_str}]\n\n")
        chunk_slides = [s for s in slides_data if chunk['start'] <= s['time'] <= chunk['end']]
        for s in chunk_slides:
            notes.append(f"![Slide {s['idx']}]({s['img'].replace('\\', '/')})\n\n")
        notes.append(process_text_chunk(chunk['text'], args.model) + "\n\n---\n\n")

    with open("semantic_study_notes.md", "w", encoding="utf-8") as f:
        f.writelines(notes)
    log("\nComplete!", always=True)

if __name__ == "__main__":
    main()
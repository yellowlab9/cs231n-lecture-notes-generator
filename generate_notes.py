import os
import re
import cv2
from collections import deque
import argparse
import requests
import yt_dlp
from pdf2image import convert_from_path
import numpy as np
import scipy.stats
import sys
import av
import ollama

# --- WINDOWS DPI AWARENESS ---
# Prevents OpenCV windows from being artificially blown up by Windows display scaling
try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(2) # PROCESS_PER_MONITOR_DPI_AWARE
except Exception:
    pass

# --- WINDOWS CONFIGURATION ---
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

def compute_percentage_of_pixels_changed(frame_gray, anchor_frame_gray, threshold):
    diff = cv2.absdiff(frame_gray, anchor_frame_gray)
    _, thresh_diff = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    changed_pixels = cv2.countNonZero(thresh_diff)
    total_pixels = frame_gray.size
    percentage_changed = (changed_pixels / total_pixels) * 100.0
    return percentage_changed

def compute_percentage_of_light_pixels(frame, threshold=220):
    lower_bound = (threshold + 1, threshold + 1, threshold + 1)
    upper_bound = (255, 255, 255)
    mask = cv2.inRange(frame, lower_bound, upper_bound)
    light_pixels = cv2.countNonZero(mask)
    total_pixels = frame.shape[0] * frame.shape[1]
    percentage_light = (light_pixels / total_pixels) * 100.0
    return percentage_light

def kde_mode(frames_list, bandwidth=20.0, max_samples=128):
    """
    Estimates the mode of pixels across frames using a fast subsampled mode.
    frames_list: list of numpy arrays (H, W, C) or a single stacked array in uint8.
    """
    if isinstance(frames_list, np.ndarray):
        frames_array = frames_list
        N = frames_array.shape[0]
    else:
        N = len(frames_list)
        if N == 0:
            return None
        frames_array = np.stack(frames_list)

    if N == 1:
        return frames_array[0].copy()

    if N <= 16:
        var = 2 * (bandwidth ** 2)
        lut = np.exp(-(np.arange(256, dtype=np.float32)**2) / var)

        max_density = np.zeros(frames_array.shape[1:], dtype=np.float32) - 1.0
        best_mode = np.zeros(frames_array.shape[1:], dtype=np.uint8)

        for i in range(N):
            frame_i = frames_array[i]
            density_i = np.zeros(frames_array.shape[1:], dtype=np.float32)
            for j in range(N):
                diff = cv2.absdiff(frames_array[j], frame_i)
                density_i += lut[diff]

            mask = density_i > max_density
            max_density[mask] = density_i[mask]
            best_mode[mask] = frame_i[mask]

        return best_mode

    if N > max_samples:
        indices = np.linspace(0, N - 1, max_samples, dtype=int)
        sample_frames = frames_array[indices]
    else:
        sample_frames = frames_array

    # Flatten spatial dimensions to drastically speed up scipy.stats.mode computation
    original_shape = sample_frames.shape[1:]
    sample_frames_flat = sample_frames.reshape(sample_frames.shape[0], -1)

    try:
        # The keepdims arg is deprecated in newer scipy and behavior is default
        mode_result = scipy.stats.mode(sample_frames_flat, axis=0)
    except TypeError:
        mode_result = scipy.stats.mode(sample_frames_flat, axis=0)

    return mode_result.mode.reshape(original_shape).astype(np.uint8)

def save_slide_mode(frames_list, output_path):
    """Calculates and saves the mode image of a list of frames to the specified path."""
    mode_img = None
    if not frames_list or not output_path:
        return mode_img
    try:
        mode_img = kde_mode(frames_list)
        cv2.imwrite(output_path, mode_img)
        log(f"                     Updated {output_path} with mode of {len(frames_list)} frames.", always=True)
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

    sub_file = None
    for ext in ['.en.vtt', '.en.srt', '.vtt', '.srt']:
        if os.path.exists(f"{base}{ext}"):
            sub_file = f"{base}{ext}"
            break

    if not os.path.exists(video_path) or not sub_file:
        log(f"Downloading video and transcript from {url}...", always=True)
        
        ydl_opts = {
            'format': 'bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'merge_output_format': 'mp4',
            'outtmpl': f'{base}.%(ext)s',
            'quiet': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],
            'subtitlesformat': 'vtt/srt/best'
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([url])

        for ext in ['.en.vtt', '.en.srt', '.vtt', '.srt']:
            if os.path.exists(f"{base}{ext}"):
                sub_file = f"{base}{ext}"
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
    4. Keep the Markdown text *extra close* to the exact wording, vocabulary, and flow of the original transcript. 
    
    Raw Transcript:
    {transcript_text}
    
    Output ONLY the final, cleaned Markdown text.
    """
    try:
        client = ollama.Client(host="http://127.0.0.1:11434")
        response = client.generate(model=model_name, prompt=prompt)
        return response['response'].strip()
    except Exception as e:
        log(f"Ollama Error: {e}", always=True)
        return transcript_text

def detect_slides(video_path, SLIDE_DIR, SLIDE_CSV_FILE, args):
    """
    Detects and extracts slides from a video file.

    Args:
        video_path (str): Path to the video file.
        SLIDE_DIR (str): Directory to save slide images.
        SLIDE_CSV_FILE (str): Path to the CSV file for logging slide info.
        args (argparse.Namespace): Command-line arguments.

    Returns:
        list: A list of dictionaries, where each dictionary contains information about a detected slide.
    """
    # --- CACHE CHECK ---
    if os.path.exists(SLIDE_CSV_FILE) and os.path.getsize(SLIDE_CSV_FILE) > 0:
        log(f"Found existing slide data file: {SLIDE_CSV_FILE}. Verifying contents...", always=True)
        
        slides_from_csv = []
        all_files_exist = True
        
        with open(SLIDE_CSV_FILE, 'r', encoding='utf-8') as f:
            next(f, None)  # Skip header
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 3:
                    img_path, _, _ = parts
                    slides_from_csv.append(img_path)
                    if not os.path.exists(img_path):
                        all_files_exist = False
                        log(f"  -> Mismatch: File '{img_path}' from CSV not found on disk.", always=True)
                        break
                else:
                    all_files_exist = False
                    log(f"  -> Mismatch: Malformed line in CSV: {line.strip()}", always=True)
                    break

        if all_files_exist:
            files_in_dir = {os.path.join(SLIDE_DIR, f) for f in os.listdir(SLIDE_DIR) if f.endswith('.jpg')}
            if files_in_dir == set(slides_from_csv):
                log("  -> Verification successful. Loading slides from cache.", always=True)
                slides_data = []
                with open(SLIDE_CSV_FILE, 'r', encoding='utf-8') as f:
                    next(f, None) # Skip header
                    for line in f:
                        img_path, frame_idx, time_stamp = line.strip().split(',')
                        time_ms = sum(x * int(t) for x, t in zip([3600000, 60000, 1000], time_stamp.split('.')[0].split(':'))) + int(time_stamp.split('.')[1])
                        slides_data.append({'img': img_path, 'time': time_ms / 1000, 'idx': int(frame_idx)})
                return slides_data
            else:
                log("  -> Mismatch: Files in directory do not match CSV. Re-running detection.", always=True)

    try:
        slides_data = []
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
        cv2.namedWindow("Slide CDT", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Current Frame", cv2.WINDOW_AUTOSIZE)

    qLen = args.maxlen
    qCenter = qLen // 2

    frame_buffer = deque(maxlen=qLen)
    prev_slide_gray = None
    same_slide_frames = []
    same_slide_frames_idx = []

    same_slide_frame_first_idx = None
    same_slide_frame_first_time = None
    same_slide_frame_first_img_path = None

    same_slide_frame_last_idx = None
    same_slide_frame_last_time = None

    prev_slide_mode_gray = None
    new_slide_mode = None
    new_slide_mode_gray = None
    slide_detected = False

    log(f"Scanning via Global Vocabulary Match (2-word minimum)", always=True)

    count = 0
    frame_count = 0
    for frame_idx, av_frame in enumerate(container.decode(video=0)):

        # Skip B-frames (pict_type is 3 for B-frames in FFmpeg, or an enum with name 'B')
        if (getattr(av_frame.pict_type, 'name', None) == 'B' or av_frame.pict_type == 3):
            continue
        
        frame_count += 1
        if (frame_count % 2) != 1:
            continue

        frame = av_frame.to_ndarray(format='bgr24')
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        time_ms = float(av_frame.time * 1000) if av_frame.time is not None else 0.0

        frame_diff = 0.0
        if len(frame_buffer) >= qCenter:
            frame_diff = compute_percentage_of_pixels_changed(frame_gray, frame_buffer[-qCenter][1], args.scene_threshold)

        percent_light_pixels = compute_percentage_of_light_pixels(frame)

        frame_buffer.append((frame, frame_gray, frame_idx, time_ms, frame_diff, percent_light_pixels))

        if len(frame_buffer) == args.maxlen:
            current_anchor_frame      = frame_buffer[qCenter][0]

            prev_anchor_frame_gray    = frame_buffer[0][1]
            current_anchor_frame_gray = frame_buffer[qCenter][1]
            next_anchor_frame_gray    = frame_buffer[-1][1]

            prev_anchor_diff = frame_buffer[qCenter][4]
            next_anchor_diff = frame_buffer[-1][4]
            current_anchor_frame_pcnt_light = frame_buffer[qCenter][5]


            time_ms = frame_buffer[qCenter][3]
            frame_idx = frame_buffer[qCenter][2]
            time_stamp = format_time(time_ms)

            if args.display:
                disp = current_anchor_frame.copy()
                cv2.putText(disp, f"Frame {frame_idx} at time {time_stamp} with ({prev_anchor_diff:5.2f}, {next_anchor_diff:5.2f}, {current_anchor_frame_pcnt_light:5.2f}%)", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                cv2.imshow("Current Frame", disp)

            scene_changed_from_past = prev_anchor_diff > args.past_scene_pcnt_pixels_changed
            scene_changed_from_next = next_anchor_diff > args.next_scene_pcnt_pixels_changed

            slide_cdt = not (scene_changed_from_past or scene_changed_from_next)
            slide_cdt = slide_cdt and (current_anchor_frame_pcnt_light >= args.slide_pcnt_light)
 
            if slide_cdt:
                slide_cdt_gray_pcnt_pixels_changed = 100.0
                if prev_slide_mode_gray is not None:
                    slide_cdt_gray_pcnt_pixels_changed = compute_percentage_of_pixels_changed(prev_slide_mode_gray, current_anchor_frame_gray, args.slide_threshold)

                slide_pcnt_pixels_changed = 100.0
                if prev_slide_gray is not None:
                    slide_pcnt_pixels_changed = compute_percentage_of_pixels_changed(prev_slide_gray, current_anchor_frame_gray, args.slide_threshold)
                    if args.display and frame_buffer:
                        disp = current_anchor_frame.copy()
                        cv2.putText(disp, f"[CDT] Frame {frame_idx} at {time_stamp} with ({prev_anchor_diff:5.2f}, {next_anchor_diff:5.2f}, {current_anchor_frame_pcnt_light:5.2f}%, {slide_cdt_gray_pcnt_pixels_changed:5.2f})", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                        cv2.imshow("Slide CDT", disp)
                else:
                    prev_slide_gray = current_anchor_frame_gray.copy()

                slide_detected = True
                if slide_pcnt_pixels_changed > args.slide_pcnt_pixels_changed:
                    log(f"  --> [SLIDE CDT]    Frame {frame_idx} at time {time_stamp} with ({prev_anchor_diff:5.2f}, {next_anchor_diff:5.2f}, {current_anchor_frame_pcnt_light:5.2f}%, {slide_pcnt_pixels_changed:5.2f})", always=True)
                    new_slide_detected = True
                else:
                    new_slide_detected = False
            else:
                slide_detected = False
                new_slide_detected = False

            if slide_detected:
                if new_slide_detected:
                    if same_slide_frame_first_img_path is not None:
                        log(f"  --> [SLIDE]        Frame {same_slide_frame_first_idx} to {same_slide_frame_last_idx} with total frames: {len(same_slide_frames)}", always=True)
                        log(f"                     {same_slide_frames_idx}", always=True)
                        new_slide_mode = save_slide_mode(same_slide_frames, same_slide_frame_first_img_path)

                        if new_slide_mode is not None:
                            new_slide_mode_gray = cv2.cvtColor(new_slide_mode, cv2.COLOR_BGR2GRAY)
                            slide_mode_gray_pcnt_pixels_changed = 100.0
                            if prev_slide_mode_gray is not None:
                                slide_mode_gray_pcnt_pixels_changed = compute_percentage_of_pixels_changed(prev_slide_mode_gray, new_slide_mode_gray, args.slide_threshold)

                            prev_slide_mode_gray = new_slide_mode_gray.copy()

                        if args.display:
                            disp = new_slide_mode.copy()
                            cv2.putText(disp, f"[Slide] Frame {same_slide_frame_first_idx} at {same_slide_frame_first_time} with ({slide_mode_gray_pcnt_pixels_changed:5.2f}%)", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                            cv2.imshow("Slide", disp)
                            cv2.waitKey(5) # Short wait`
                    clean_ts = time_stamp.replace(':', '-')
                    slide_name = f"slide_{frame_idx}_{clean_ts}.jpg"

                    img_path = f"{SLIDE_DIR}/{slide_name}"
                    with open(SLIDE_CSV_FILE, 'a', encoding='utf-8') as f:
                        f.write(f"{img_path},{frame_idx},{time_stamp}\n")
                    slides_data.append({'img': img_path, 'time': time_ms / 1000, 'idx': frame_idx})

                    same_slide_frames = [current_anchor_frame.copy()]
                    same_slide_frame_first_img_path = img_path                    
                    same_slide_frame_first_time  = time_stamp
                    same_slide_frame_first_idx = frame_idx

                    same_slide_frames_idx = [frame_idx]

                    prev_slide_gray = current_anchor_frame_gray.copy()


                    count = 0
                else:
                    num_same_slides = len(same_slide_frames)
                    if ((count % 4) == 0) and (num_same_slides < args.max_same_slides//4):
                        same_slide_frames.append(current_anchor_frame.copy())
                        same_slide_frames_idx.append(frame_idx)
                    elif ((count % 8) == 0) and (num_same_slides < args.max_same_slides//2) :
                        same_slide_frames.append(current_anchor_frame.copy())
                        same_slide_frames_idx.append(frame_idx)
                    elif ((count % 16) == 0) and (num_same_slides < args.max_same_slides) :
                        same_slide_frames.append(current_anchor_frame.copy())
                        same_slide_frames_idx.append(frame_idx)
                    same_slide_frame_last_idx = frame_idx
                    same_slide_frame_last_time = time_stamp
                count += 1

        if args.display and cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Process the mode for the final accumulated slide at the end of the loop
    save_slide_mode(same_slide_frames, same_slide_frame_first_img_path)

    container.close()
    cv2.destroyAllWindows()
    return slides_data

def generate_study_guide(output_path, transcript_chunks, slides_data, model_name):
    """
    Generates and saves the final Markdown study guide.

    Args:
        output_path (str): The path to save the final .md file.
        transcript_chunks (list): A list of transcript chunk dictionaries.
        slides_data (list): A list of detected slide dictionaries.
        model_name (str): The name of the Ollama model used for processing.
    """
    log("\nGenerating notes...", always=True)
    notes = [f"# Lecture Study Guide\n\n*Generated using {model_name}*\n\n"]
    for chunk in transcript_chunks:
        start_m, start_s = divmod(int(chunk['start']), 60)
        start_h, start_m = divmod(start_m, 60)
        time_str = f"{start_h:02d}:{start_m:02d}:{start_s:02d}"
        notes.append(f"### ⏱️ [{time_str}]\n\n")
        chunk_slides = [s for s in slides_data if chunk['start'] <= s['time'] <= chunk['end']]
        for s in chunk_slides:
            img_path = s['img'].replace('\\', '/')
            notes.append(f"![Slide {s['idx']}]({img_path})\n\n")
        processed_text = process_text_chunk(chunk['text'], model_name)
        notes.append(processed_text + "\n\n---\n\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(notes)

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

    global VERBOSE, LOG_FILE
    VERBOSE = args.verbose or args.debug

    pdf_name = args.pdf.split("/")[-1] if args.pdf.startswith("http") else args.pdf
    LOG_FILE = pdf_name.replace('.pdf', '_log.txt')
    open(LOG_FILE, 'w', encoding='utf-8').close()  # initialize and clear previous run's log file

    SLIDE_DIR = "slides"

    SLIDE_CSV_FILE = pdf_name.replace('.pdf', '_slide.csv')
    if not os.path.exists(SLIDE_CSV_FILE):
        with open(SLIDE_CSV_FILE, 'w', encoding='utf-8') as f:
            f.write("img_path,frame_idx,time_stamp\n") # CSV Header

    if not os.path.exists(SLIDE_DIR): os.makedirs(SLIDE_DIR)

    pdf_path = download_pdf(args.pdf)
    video_path, transcript_path = download_media(args.video_url)


    slides_data = detect_slides(video_path, SLIDE_DIR, SLIDE_CSV_FILE, args)
    
    transcript_chunks = parse_transcript(transcript_path, chunk_duration=180)
    
    output_md_path = pdf_name.replace('.pdf', '_study_guide.md')
    generate_study_guide(output_md_path, transcript_chunks, slides_data, args.model)

    log("Complete!", always=True)

if __name__ == "__main__":
    main()
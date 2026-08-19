import os
import cv2
from collections import deque
import numpy as np
import scipy.stats
import sys
import av
from .utils import log, format_time
from pathlib import Path

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

import shutil

def _migrate_legacy_slides(base_name, SLIDE_DIR, SLIDE_CSV_FILE):
    """
    Migrates legacy slides/ and CSV files from root into lectures/<base_name>/slides/ and lectures_cache/slide_csv/.
    """
    os.makedirs(SLIDE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SLIDE_CSV_FILE), exist_ok=True)

    # Possible legacy CSV files in root: e.g. lecture_3_slide.csv, lecture_03_slide.csv
    legacy_csv_names = [f"{base_name}_slide.csv"]
    if base_name.startswith("lecture_"):
        num_part = base_name.replace("lecture_", "").lstrip("0")
        if num_part:
            legacy_csv_names.append(f"lecture_{num_part}_slide.csv")

    for leg_csv in legacy_csv_names:
        if os.path.exists(leg_csv) and (not os.path.exists(SLIDE_CSV_FILE) or os.path.getsize(SLIDE_CSV_FILE) == 0):
            log(f"[*] Migrating legacy CSV '{leg_csv}' to '{SLIDE_CSV_FILE}'...", always=True)
            with open(leg_csv, 'r', encoding='utf-8') as fin, open(SLIDE_CSV_FILE, 'w', encoding='utf-8') as fout:
                header = fin.readline()
                fout.write(header if header.startswith("img_path") else "img_path,frame_idx,time_stamp\n")
                for line in fin:
                    parts = line.strip().split(',')
                    if len(parts) == 3:
                        old_path, f_idx, ts = parts
                        old_fname = os.path.basename(old_path)
                        # Ensure prefix is applied
                        if not old_fname.startswith(f"{base_name}_"):
                            new_fname = f"{base_name}_{old_fname}"
                        else:
                            new_fname = old_fname
                        fout.write(f"slides/{new_fname},{f_idx},{ts}\n")

    # Migrate legacy slides from root 'slides/' directory
    legacy_slides_dir = "slides"
    if os.path.exists(legacy_slides_dir) and os.path.isdir(legacy_slides_dir) and legacy_slides_dir != SLIDE_DIR:
        legacy_files = [f for f in os.listdir(legacy_slides_dir) if f.endswith('.jpg')]
        if legacy_files and len(os.listdir(SLIDE_DIR)) == 0:
            log(f"[*] Migrating {len(legacy_files)} slide images from '{legacy_slides_dir}/' to '{SLIDE_DIR}/' with '{base_name}_' prefix...", always=True)
            for f in legacy_files:
                src_path = os.path.join(legacy_slides_dir, f)
                new_fname = f if f.startswith(f"{base_name}_") else f"{base_name}_{f}"
                dst_path = os.path.join(SLIDE_DIR, new_fname)
                shutil.copy2(src_path, dst_path)

def load_slides_from_cache(SLIDE_CSV_FILE, SLIDE_DIR):
    """
    Attempts to load slide data from a CSV cache file or disk if valid.
    Returns a list of slide data dictionaries if successful, otherwise None.
    """
    if not os.path.exists(SLIDE_DIR):
        return None

    files_in_dir = sorted([f for f in os.listdir(SLIDE_DIR) if f.endswith('.jpg')])
    if not files_in_dir:
        return None

    # Check if CSV exists and parse entries
    if os.path.exists(SLIDE_CSV_FILE) and os.path.getsize(SLIDE_CSV_FILE) > 0:
        log(f"Found existing slide data file: {SLIDE_CSV_FILE}. Verifying contents...", always=True)
        slides_dict = {}
        with open(SLIDE_CSV_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) != 3:
                    continue
                # Skip header row if present
                if parts[0].strip() == "img_path" or not parts[1].strip().isdigit():
                    continue
                img_path, frame_idx_str, time_stamp = parts
                fname = os.path.basename(img_path)
                full_img_path = os.path.join(SLIDE_DIR, fname)
                if os.path.exists(full_img_path):
                    slides_dict[fname] = (img_path, int(frame_idx_str), time_stamp)

        # If CSV contains all images in SLIDE_DIR, return sorted slides
        if set(slides_dict.keys()) == set(files_in_dir):
            log(f"  -> Verification successful. Loading {len(slides_dict)} slides from cache.", always=True)
            slide_dir_name = os.path.basename(SLIDE_DIR)
            slides_data = []
            for fname in files_in_dir:
                img_path, frame_idx, time_stamp = slides_dict[fname]
                ts_parts = time_stamp.split('.')
                time_hms = ts_parts[0].split(':')
                hms_ms = sum(x * int(t) for x, t in zip([3600000, 60000, 1000], time_hms))
                ms = int(ts_parts[1]) if len(ts_parts) > 1 else 0
                time_ms = hms_ms + ms
                slides_data.append({
                    'img': f"{slide_dir_name}/{fname}",
                    'time': time_ms / 1000.0,
                    'idx': frame_idx,
                    'timestamp': time_stamp
                })
            # Clean and write valid deduplicated CSV
            with open(SLIDE_CSV_FILE, 'w', encoding='utf-8') as f:
                f.write("img_path,frame_idx,time_stamp\n")
                for s in slides_data:
                    f.write(f"{s['img']},{s['idx']},{s['timestamp']}\n")
            return slides_data

    # Self-recovery fallback: parse timestamps directly from slide filenames
    import re
    slide_dir_name = os.path.basename(SLIDE_DIR)
    parsed_slides = []
    for fname in files_in_dir:
        m = re.search(r'slide_(\d+)_(\d{2})-(\d{2})-(\d{2}\.\d+)\.jpg$', fname)
        if m:
            frame_idx = int(m.group(1))
            time_stamp = f"{m.group(2)}:{m.group(3)}:{m.group(4)}"
            time_hms = [int(m.group(2)), int(m.group(3)), int(float(m.group(4)))]
            ms = int(round((float(m.group(4)) - int(float(m.group(4)))) * 1000))
            time_ms = sum(x * t for x, t in zip([3600000, 60000, 1000], time_hms)) + ms
            parsed_slides.append({
                'img': f"{slide_dir_name}/{fname}",
                'time': time_ms / 1000.0,
                'idx': frame_idx,
                'timestamp': time_stamp
            })

    if len(parsed_slides) == len(files_in_dir) and len(parsed_slides) > 0:
        parsed_slides.sort(key=lambda s: s['time'])
        log(f"  -> Recovered {len(parsed_slides)} slides directly from {SLIDE_DIR}.", always=True)
        with open(SLIDE_CSV_FILE, 'w', encoding='utf-8') as f:
            f.write("img_path,frame_idx,time_stamp\n")
            for s in parsed_slides:
                f.write(f"{s['img']},{s['idx']},{s['timestamp']}\n")
        return parsed_slides

    return None

def detect_slides(video_path, SLIDE_DIR, SLIDE_CSV_FILE, args, slide_prefix=""):
    """
    Detects and extracts slides from a video file.

    Args:
        video_path (str): Path to the video file.
        SLIDE_DIR (str): Directory to save slide images.
        SLIDE_CSV_FILE (str): Path to the CSV file for logging slide info.
        args (argparse.Namespace): Command-line arguments.
        slide_prefix (str): Prefix for slide filenames (e.g. 'lecture_03').

    Returns:
        list: A list of dictionaries, where each dictionary contains information about a detected slide.
    """
    base_name = slide_prefix or os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(SLIDE_DIR, exist_ok=True)
    csv_dir = os.path.dirname(SLIDE_CSV_FILE)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    # Migrate any legacy files from root into lectures/<base_name>/slides and lectures_cache/slide_csv
    _migrate_legacy_slides(base_name, SLIDE_DIR, SLIDE_CSV_FILE)

    # --- CACHE CHECK ---
    cached_slides = load_slides_from_cache(SLIDE_CSV_FILE, SLIDE_DIR)
    if cached_slides is not None:
        return cached_slides

    # Initialize / clean CSV file with header before extraction
    with open(SLIDE_CSV_FILE, 'w', encoding='utf-8') as f:
        f.write("img_path,frame_idx,time_stamp\n")

    # --- VIDEO INITIALIZATION ---
    try:
        slides_data = []
        container = av.open(video_path)
        if not container.streams.video:
            log(f"Error: No video stream found in {video_path}", always=True)
            sys.exit(1)
        video_stream = container.streams.video[0]
        video_stream.thread_type = "AUTO" # Enable multithreaded decoding

        fps = float(video_stream.average_rate) if video_stream.average_rate else 0.0
        fps1001 = int(round(fps * 1001))
        frame_count = video_stream.frames or 0
        width = getattr(video_stream, 'width', 0) or getattr(getattr(video_stream, 'codec_context', None), 'width', 0)
        height = getattr(video_stream, 'height', 0) or getattr(getattr(video_stream, 'codec_context', None), 'height', 0)

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
                    if base_name and base_name != "slide":
                        slide_name = f"{base_name}_slide_{frame_idx}_{clean_ts}.jpg"
                    else:
                        slide_name = f"slide_{frame_idx}_{clean_ts}.jpg"
                    img_file_path = os.path.join(SLIDE_DIR, slide_name)
                    slide_dir_name = os.path.basename(SLIDE_DIR)
                    rel_img_path = f"{slide_dir_name}/{slide_name}"

                    with open(SLIDE_CSV_FILE, 'a', encoding='utf-8') as f:
                        f.write(f"{rel_img_path},{frame_idx},{time_stamp}\n")
                    slides_data.append({'img': rel_img_path, 'time': time_ms / 1000, 'idx': frame_idx, 'timestamp': time_stamp})

                    same_slide_frames = [current_anchor_frame.copy()]
                    same_slide_frame_first_img_path = img_file_path                    
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
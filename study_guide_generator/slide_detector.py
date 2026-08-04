import os
import cv2
from collections import deque
import numpy as np
import scipy.stats
import sys
import av
from .utils import log, format_time

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
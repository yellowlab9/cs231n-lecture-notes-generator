import os
import re
import sys
from .utils import log

def time_to_seconds(time_str):
    """Converts a 'HH:MM:SS.ms' or 'HH:MM:SS,ms' string to seconds."""
    time_str = time_str.strip()
    parts = time_str.replace(',', '.').split('.')
    h, m, s = map(int, parts[0].split(':'))
    ms = int(parts[1]) if len(parts) > 1 else 0
    return h * 3600 + m * 60 + s + ms / 1000.0

def parse_transcript(file_path, chunk_duration=300, overlap_duration=60):
    """
    Parses a .vtt or .srt file, returning a list of large text chunks with overlap.
    Each chunk contains the full text for an LLM and a list of fine-grained
    'segments' with precise timing information for timestamping.
    """
    if not file_path or not os.path.exists(file_path):
        return []
        
    # The log() function adds a newline, which interferes with a single-line progress bar.
    # We will handle logging for this specific section manually to create a clean progress indicator.
    log_prefix = f"[LOG] Parsing transcript file: {file_path}"
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_lines = len(lines)
    if total_lines == 0:
        return []

    # 1. First, parse all individual caption segments from the file
    all_segments = []
    i = 0
    while i < len(lines):
        # Simple progress report that updates on one line
        if i % 100 == 0 or i == total_lines - 1: # Update less frequently, but always at the start and end
            progress = (i / total_lines) * 100
            sys.stdout.write(f"\r{log_prefix} ... {progress:.1f}%")
            sys.stdout.flush() # Ensure the update is visible immediately

        if '-->' in lines[i]:
            start_str, end_str_full = lines[i].split('-->')
            # VTT files can have extra metadata after the end timestamp (e.g., "align:start")
            end_str = end_str_full.strip().split(' ')[0]
            start_sec = time_to_seconds(start_str)
            end_sec = time_to_seconds(end_str)

            text_lines = []
            j = i + 1
            while j < len(lines) and lines[j].strip() != '' and '-->' not in lines[j]:
                clean_text = re.sub(r'<[^>]+>', '', lines[j].strip())
                if clean_text and not clean_text.isdigit():
                    text_lines.append(clean_text)
                j += 1
            i = j # Move main loop index forward

            text = " ".join(text_lines)
            if text:
                all_segments.append({'start': start_sec, 'end': end_sec, 'text': text})
        else:
            i += 1

    # Finalize progress bar: overwrite with "Done" and add a newline for subsequent logs.
    # Pad with spaces to ensure the previous, longer message is fully overwritten.
    sys.stdout.write(f"\r{log_prefix} ... Done.      \n")

    if not all_segments:
        return []

    # 2. Now, group these segments into larger chunks with overlap
    chunks = []
    current_idx = 0
    while current_idx < len(all_segments):
        start_time = all_segments[current_idx]['start']

        # Find the end of the chunk (approx. chunk_duration long)
        end_idx = current_idx
        while end_idx < len(all_segments) - 1 and (all_segments[end_idx]['start'] - start_time < chunk_duration):
            end_idx += 1

        chunk_segments = all_segments[current_idx : end_idx + 1]
        if not chunk_segments: break

        full_text = " ".join([s['text'] for s in chunk_segments])
        chunks.append({'start': chunk_segments[0]['start'], 'end': chunk_segments[-1]['end'], 'text': full_text, 'segments': chunk_segments})

        # If we've reached the end, stop.
        if end_idx >= len(all_segments) - 1: break

        # Determine the start of the next chunk by looking back from the end of the current one.
        next_chunk_start_time = chunk_segments[-1]['end'] - overlap_duration
        
        # Find the segment index that is closest to (but not after) the desired start time for the overlap.
        next_start_idx = end_idx
        while next_start_idx > current_idx and all_segments[next_start_idx]['start'] > next_chunk_start_time:
            next_start_idx -= 1
        
        # Ensure we always move forward
        current_idx = next_start_idx if next_start_idx > current_idx else end_idx + 1

    log(f"  -> Created {len(chunks)} overlapping transcript chunks.")
    return chunks
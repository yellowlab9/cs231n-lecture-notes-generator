import os
import re
from .utils import log

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
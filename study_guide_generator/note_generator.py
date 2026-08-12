import re
from .utils import log
from .llm_processor import process_text_chunk
import nltk
from fuzzywuzzy import process as fuzzy_process, fuzz

# --- One-time download for NLTK's sentence tokenizer ---
def ensure_nltk_punkt():
    """Ensures the NLTK 'punkt' tokenizer data is downloaded."""
    try:
        # Check for both the main tokenizer data and the tab-separated resource
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        log("NLTK 'punkt' or 'punkt_tab' resource not found. Downloading necessary resources...", always=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)

def _get_timestamp_from_chunk_linear(cleaned_sentence, original_chunk):
    """Fallback estimator using linear interpolation over a large chunk."""
    original_text = original_chunk['text']
    chunk_start_sec = original_chunk['start']
    chunk_end_sec = original_chunk['end']
    chunk_duration = chunk_end_sec - chunk_start_sec
    if chunk_duration <= 0 or not original_text:
        return chunk_start_sec

    try:
        search_term = cleaned_sentence[:20]
        char_index = original_text.lower().find(search_term.lower())
        if char_index == -1:
            return chunk_start_sec
    except:
        return chunk_start_sec

    progress = char_index / len(original_text)
    estimated_time = chunk_start_sec + (chunk_duration * progress)
    return estimated_time

def get_sentence_timestamp(cleaned_sentence, original_chunk):
    """
    Finds the best timestamp for a cleaned sentence by fuzzy matching it against
    the original transcript's fine-grained segments.
    """
    # If detailed segments are not available, use the old linear method
    if 'segments' not in original_chunk or not original_chunk['segments']:
        return _get_timestamp_from_chunk_linear(cleaned_sentence, original_chunk)

    segments = original_chunk['segments']
    segment_texts = [seg['text'] for seg in segments]
    
    result = fuzzy_process.extractOne(cleaned_sentence, segment_texts, scorer=fuzz.token_sort_ratio)
    
    if not result or result[1] < 50:
        return original_chunk['start']

    best_match_text = result[0]
    segment_index = segment_texts.index(best_match_text)
    best_segment = segments[segment_index]
    
    segment_text = best_segment['text']
    segment_start = best_segment['start']
    segment_duration = best_segment['end'] - segment_start

    if segment_duration <= 0 or not segment_text:
        return segment_start

    try:
        search_term = cleaned_sentence[:30]
        char_index = segment_text.lower().find(search_term.lower())
        if char_index == -1:
            return segment_start
    except:
        return segment_start

    progress = char_index / len(segment_text)
    estimated_time = segment_start + (segment_duration * progress)
    return estimated_time

def generate_study_guide(output_path, transcript_chunks, slides_data, model_name):
    """
    Generates and saves the final Markdown study guide.
    """
    log("\nGenerating notes...", always=True)
    # Ensure NLTK data is ready before processing
    ensure_nltk_punkt()

    # 1. Create a timeline of paragraphs and slides.
    timeline_items = []
    total_chunks = len(transcript_chunks)
    log(f"Processing {total_chunks} transcript chunks with LLM...", always=True)
    for i, chunk in enumerate(transcript_chunks):
        log(f"  -> Processing chunk {i + 1}/{total_chunks}...", always=True)
        cleaned_text_with_paragraphs = process_text_chunk(chunk['text'], model_name)
        
        # Split the LLM output into paragraphs.
        paragraphs = [p.strip() for p in cleaned_text_with_paragraphs.split('\n\n') if p.strip()]

        for p_text in paragraphs:
            # Estimate the time of the paragraph by its first sentence.
            try:
                first_sentence = nltk.sent_tokenize(p_text)[0]
                est_time = get_sentence_timestamp(first_sentence, chunk)
                timeline_items.append({'type': 'paragraph', 'time': est_time, 'text': p_text})
            except IndexError:
                # Handle cases where a "paragraph" might be empty or just whitespace.
                continue

    # Add slides to the timeline
    for slide in slides_data:
        timeline_items.append({'type': 'slide', 'time': slide['time'], 'data': slide})

    # Sort everything chronologically
    timeline_items.sort(key=lambda x: x['time'])

    # 2. Filter the timeline to consolidate consecutive slides.
    # When multiple slides appear in a row, only keep the last one.
    log("Aligning slides and paragraphs...", always=True)
    processed_timeline = []
    i = 0
    while i < len(timeline_items):
        item = timeline_items[i]
        if item['type'] == 'slide':
            # Look ahead to find the end of a consecutive block of slides
            j = i + 1
            while j < len(timeline_items) and timeline_items[j]['type'] == 'slide':
                j += 1
            # Add only the last slide from the block
            processed_timeline.append(timeline_items[j - 1])
            i = j # Move index past the processed block
        else:
            processed_timeline.append(item)
            i += 1

    # 3. Generate the markdown from the processed timeline.
    log("Writing final study guide to file...", always=True)
    notes = [f"# Lecture Study Guide\n\n*Generated using {model_name}*\n\n"]

    for item in processed_timeline:
        if item['type'] == 'slide':
            # Add the slide image.
            slide = item['data']
            img_path = slide['img'].replace('\\', '/')
            notes.append(f"![Slide {slide['idx']}]({img_path})\n\n")
        
        elif item['type'] == 'paragraph':
            # Add the paragraph text.
            notes.append(item['text'] + "\n\n")
        
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(notes)
    log(f"Study guide saved to {output_path}", always=True)
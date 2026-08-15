import os
import re
from itertools import groupby
from .utils import log
from .llm_processor import process_text_chunk
import nltk
from fuzzywuzzy import process as fuzzy_process, fuzz

# --- One-time download for NLTK's sentence tokenizer ---
def ensure_nltk_punkt():
    """Ensures the NLTK 'punkt' tokenizer data is downloaded."""
    for resource in ['tokenizers/punkt', 'tokenizers/punkt_tab']:
        try:
            nltk.data.find(resource)
        except LookupError:
            pkg = resource.split('/')[-1]
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass

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

def get_sentence_timestamp(cleaned_sentence, original_chunk, score_cutoff=50):
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
    
    # If the match score is too low, it's unreliable. Return None to signal failure.
    if not result or result[1] < score_cutoff:
        return None

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

def generate_study_guide(output_path, transcript_chunks, slides_data, model_name, fuzzy_score_threshold=50, llm_retries=3, llm_retry_delay=5):
    """
    Generates and saves the final Markdown study guide.
    """
    log("\nGenerating notes...", always=True)
    # Ensure NLTK data is ready before processing
    ensure_nltk_punkt()

    # 1. Create a timeline of paragraphs and slides.
    timeline_items = []

    # A set to keep track of the first sentence of paragraphs we've already added.
    # This prevents adding duplicate content from overlapping chunks.
    processed_paragraph_starts = set()

    total_chunks = len(transcript_chunks)
    log(f"Processing {total_chunks} transcript chunks with LLM...", always=True)
    for i, chunk in enumerate(transcript_chunks):
        log(f"  -> Processing chunk {i + 1}/{total_chunks}...", always=True)
        cleaned_text_with_paragraphs = process_text_chunk(chunk['text'], model_name, retries=llm_retries, delay=llm_retry_delay)
        
        # Split the LLM output into paragraphs.
        paragraphs = [p.strip() for p in cleaned_text_with_paragraphs.split('\n\n') if p.strip()]

        for p_text in paragraphs:
            # Filter out markdown horizontal rules that the LLM might erroneously add.
            if re.fullmatch(r'[\*\-\_]{3,}', p_text):
                log(f"  -> Skipping horizontal rule artifact from LLM output: '{p_text}'", always=True)
                continue

            # Get a timestamp for the paragraph based on its first sentence.
            try:
                first_sentence = nltk.sent_tokenize(p_text)[0]

                # If we've already processed a paragraph starting with this sentence, skip it.
                if first_sentence in processed_paragraph_starts:
                    continue

                est_time = get_sentence_timestamp(first_sentence, chunk, score_cutoff=fuzzy_score_threshold)

                # If timestamping fails, it means the sentence is likely a hallucination or
                # too heavily modified. Skip it to prevent incorrect ordering.
                if est_time is None:
                    log(f"  -> Could not find reliable timestamp for paragraph. Skipping.", always=True)
                    continue

                # Add the entire paragraph as a single timeline item
                timeline_items.append({'type': 'paragraph', 'time': est_time, 'text': p_text})
                processed_paragraph_starts.add(first_sentence)
            except IndexError:
                continue

    # Add slides to the timeline
    for slide in slides_data:
        timeline_items.append({'type': 'slide', 'time': slide['time'], 'data': slide})

    # Sort everything chronologically
    timeline_items.sort(key=lambda x: x['time'])

    # 2. Filter the timeline to consolidate consecutive slides.
    # NOTE: The original logic to consolidate consecutive slides is disabled here
    # to ensure all detected slides are included in the final output.
    log("Aligning slides and paragraphs...", always=True)
    processed_timeline = timeline_items

    # 3. Generate the markdown from the processed timeline.
    log("Writing final study guide to file...", always=True)
    # Initialize an empty list for notes. The header was removed as it's not part of the original transcript.
    notes = []

    # Iterate through the timeline and add items to the notes.
    for item in processed_timeline:
        if item['type'] == 'paragraph':
            notes.append(item['text'] + "\n\n")
        elif item['type'] == 'slide':
            slide = item['data']
            img_path = slide['img'].replace('\\', '/')
            notes.append(f"![Slide {slide['idx']}]({img_path})\n\n")

    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(notes)
    log(f"Study guide saved to {output_path}", always=True)
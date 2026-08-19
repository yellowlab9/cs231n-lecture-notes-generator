import os
import re
from itertools import groupby
from .utils import log
from .llm_processor import process_text_chunk, process_clean_transcript_chunk
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

def generate_study_guide(output_path, transcript_chunks, slides_data, model_name, fuzzy_score_threshold=50, llm_retries=3, llm_retry_delay=5, is_creator_subtitle=False, img_width="75%", doc_title=None, fontsize="14pt"):
    """
    Generates and saves the final Markdown study guide.
    Guarantees slides are placed strictly BETWEEN complete sentences,
    immediately before the related sentence spoken during/after the slide appearance.
    """
    log("\nGenerating notes...", always=True)
    ensure_nltk_punkt()

    # Format percentage width string cleanly (e.g. 60 or "60" or "60%" -> "60%")
    width_clean = str(img_width).strip().rstrip('%')
    width_str = f"{width_clean}%" if width_clean.isdigit() else str(img_width).strip()

    total_chunks = len(transcript_chunks)
    sentences = []

    # 1. Extract all complete sentences with start and end timestamps
    if is_creator_subtitle:
        log(f"Formatting creator subtitles ({total_chunks} chunks, LaTeX math & prose polishing)...", always=True)
        for i, chunk in enumerate(transcript_chunks):
            log(f"  -> Formatting chunk {i + 1}/{total_chunks} (faithful transcript & LaTeX math)...", always=True)
            # Enhance presentation (LaTeX math & typography) while preserving faithful explanations
            formatted_text = process_clean_transcript_chunk(
                chunk['text'],
                model_name,
                retries=llm_retries,
                delay=llm_retry_delay
            )

            # Split into paragraphs and sentences
            paragraphs = [p.strip() for p in formatted_text.split('\n\n') if p.strip()]
            for p_text in paragraphs:
                p_sentences = nltk.sent_tokenize(p_text)
                for s_text in p_sentences:
                    s_clean = s_text.strip()
                    if not s_clean:
                        continue
                    est_time = get_sentence_timestamp(s_clean, chunk, score_cutoff=fuzzy_score_threshold)
                    if est_time is None:
                        continue
                    sentences.append({
                        'text': s_clean,
                        'start': est_time,
                        'end': est_time + 4.0
                    })
    else:
        log(f"Processing {len(transcript_chunks)} text chunks with LLM ({model_name})...", always=True)
        for i, chunk in enumerate(transcript_chunks):
            log(f"  -> Processing chunk {i+1}/{len(transcript_chunks)}...", always=True)
            cleaned_text = process_text_chunk(
                chunk['text'],
                model_name=model_name,
                retries=llm_retries,
                delay=llm_retry_delay
            )
            paragraphs = [p.strip() for p in cleaned_text.split('\n\n') if p.strip()]
            for p_text in paragraphs:
                p_sentences = nltk.sent_tokenize(p_text)
                for s_text in p_sentences:
                    s_clean = s_text.strip()
                    if not s_clean or s_clean in processed_sentence_starts:
                        continue

                    est_time = get_sentence_timestamp(s_clean, chunk, score_cutoff=fuzzy_score_threshold)
                    if est_time is None:
                        continue

                    sentences.append({
                        'text': s_clean,
                        'start': est_time,
                        'end': est_time + 4.0
                    })
                    processed_sentence_starts.add(s_clean)

    # 2. Interleave slides strictly BETWEEN complete sentences (before related sentence)
    log("Aligning slides between sentence boundaries...", always=True)
    sorted_slides = sorted(slides_data, key=lambda s: s['time'])
    slide_idx = 0
    num_slides = len(sorted_slides)

    fs_val = str(fontsize).strip()
    if fs_val.isdigit():
        fs_val = f"{fs_val}pt"

    notes = [
        "---\n",
        "jupyter:\n",
        "  jupytext:\n",
        "    formats: ipynb,md\n",
        "    text_representation:\n",
        "      extension: .md\n",
        "      format_name: markdown\n",
        "  kernelspec:\n",
        "    display_name: Python 3 (ipykernel)\n",
        "    language: python\n",
        "    name: python3\n",
        "documentclass: extarticle\n",
        "papersize: letter\n",
        'geometry: "margin=0.75in"\n',
        f"fontsize: {fs_val}\n",
        "---\n\n",
    ]
    if doc_title:
        notes.append(f"<!-- #region -->\n# {doc_title}\n<!-- #endregion -->\n\n")

    current_p_sentences = []
    in_region = False

    def close_region():
        nonlocal in_region
        if in_region:
            notes.append("<!-- #endregion -->\n\n")
            in_region = False

    def flush_paragraph():
        nonlocal current_p_sentences
        if current_p_sentences:
            p_str = " ".join(current_p_sentences).strip()
            if p_str:
                notes.append(p_str + "\n\n")
            current_p_sentences = []

    for s_idx, sent in enumerate(sentences):
        sent_text = sent['text'].strip()
        if not sent_text:
            continue
        sent_end = sent['end']

        # Collect any slides that appeared before or during this sentence
        slides_to_insert = []
        while slide_idx < num_slides:
            slide = sorted_slides[slide_idx]
            # Place slide before this sentence if slide time is at or before this sentence's end
            if s_idx == len(sentences) - 1 or slide['time'] <= sent_end:
                slides_to_insert.append(slide)
                slide_idx += 1
            else:
                break

        if slides_to_insert:
            flush_paragraph()
            close_region()
            notes.append("<!-- #region -->\n")
            in_region = True
            for sl in slides_to_insert:
                img_path = sl['img'].replace('\\', '/')
                if not img_path.startswith('./'):
                    img_path = f"./{img_path.lstrip('/')}"
                slide_ts = sl.get('timestamp') or f"{int(sl['time']//3600):02d}:{int((sl['time']%3600)//60):02d}:{int(sl['time']%60):02d}"
                notes.append(f'<p align="center"><img src="{img_path}" width="{width_str}" alt="Lecture Video at {slide_ts}" /></p>\n\n')
        elif not in_region and not notes[-1].endswith("<!-- #region -->\n"):
            notes.append("<!-- #region -->\n")
            in_region = True

        current_p_sentences.append(sent_text)

        # Form readable paragraphs (3-4 sentences or ~350-500 chars)
        combined_p = " ".join(current_p_sentences)
        if len(current_p_sentences) >= 4 or len(combined_p) >= 350:
            flush_paragraph()

    # Flush any remaining sentences
    flush_paragraph()
    close_region()

    # Append any remaining slides after speech ends
    while slide_idx < num_slides:
        sl = sorted_slides[slide_idx]
        img_path = sl['img'].replace('\\', '/')
        if not img_path.startswith('./'):
            img_path = f"./{img_path.lstrip('/')}"
        slide_ts = sl.get('timestamp') or f"{int(sl['time']//3600):02d}:{int((sl['time']%3600)//60):02d}:{int(sl['time']%60):02d}"
        notes.append(f'<!-- #region -->\n<p align="center"><img src="{img_path}" width="{width_str}" alt="Lecture Video at {slide_ts}" /></p>\n<!-- #endregion -->\n\n')
        slide_idx += 1

    # 3. Write final markdown study guide
    log("Writing final study guide to file...", always=True)
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(notes)
    log(f"Study guide saved to {output_path}", always=True)
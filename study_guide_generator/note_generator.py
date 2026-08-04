from .utils import log
from .llm_processor import process_text_chunk

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
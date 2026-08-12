import ollama
from .utils import log

def process_text_chunk(transcript_text, model_name):
    if not transcript_text.strip(): return ""
    
    prompt = f"""
    You are an expert technical editor creating a Markdown study guide compatible with MarkText from a raw lecture transcript. Your primary goal is to clean the text while preserving the speaker's original wording and flow as closely as possible.

    Your tasks are:
    1.  **Clean, Don't Rewrite:** Correct grammar, fix typos, and remove filler words (like "um", "uh"). Do NOT rephrase sentences or alter the core vocabulary. The output should feel like a polished version of the original speech, not a summary.
    2.  **Format LaTeX for MarkText:**
        - Use simple underscores for subscripts (e.g., `$L_2$`, not `$L\_2$`).
        - Use `\mathrm{{...}}` for text in equations, with tildes for spaces (e.g., `$$\mathrm{{Data~Loss}}$$`).
        - Do not escape underscores in commands like `\sum_{...}`.
    3.  **Create Natural Paragraphs:** Group the cleaned sentences into paragraphs. Start a new paragraph (with a double newline) only when there is a clear shift in topic in the original speech.
    4.  **Output Only Markdown:** Produce only the final, cleaned Markdown text. Do not add any headers, introductions, horizontal rules (`---` or `***`), or other text that wasn't in the original transcript.

    Raw Transcript:
    {transcript_text}
    
    Cleaned Markdown:
    """
    try:
        client = ollama.Client(host="http://127.0.0.1:11434")
        response = client.generate(model=model_name, prompt=prompt)
        return response['response'].strip()
    except Exception as e:
        log(f"Ollama Error: {e}", always=True)
        return transcript_text
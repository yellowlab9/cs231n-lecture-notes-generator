import ollama
import sys
import time
import re
from .utils import log

# Create a single, reusable client instance to improve efficiency.
try:
    OLLAMA_CLIENT = ollama.Client(host="http://127.0.0.1:11434")
except Exception as e:
    log(f"Failed to create Ollama client: {e}", always=True)
    OLLAMA_CLIENT = None

def check_ollama_server():
    """Checks if the Ollama server is running and accessible."""
    if OLLAMA_CLIENT is None:
        log("Ollama client could not be initialized.", always=True)
        return False
    try:
        # A lightweight command to check connectivity without fetching large model data
        OLLAMA_CLIENT.list()
        return True
    except Exception:
        log("Ollama server not found or not responding at http://127.0.0.1:11434.", always=True)
        log("Please ensure the Ollama application is running before starting the script.", always=True)
        return False

def process_clean_transcript_chunk(transcript_text, model_name, retries=3, delay=5):
    """
    Minimal-touch formatter for high-quality creator subtitles (.en-US.vtt).
    Strictly preserves verbatim wording, phrasing, and context while formatting
    LaTeX math ($W$, $b$, $x_i$, $L_1$) and structuring clean paragraphs.
    """
    if not transcript_text.strip():
        return ""

    prompt_template = """
    You are a technical editor converting a video lecture transcript into a readable Markdown lecture note.

    Your goal is to make ONLY minimal editorial changes to convert spoken transcript into clean lecture notes while keeping all content, phrasing, and full context intact.

    Guidelines:
    1. **Strictly Preserve Verbatim Text & Context:** Keep the speaker's exact words, sentences, explanations, examples, and phrasing. Do NOT rewrite, summarize, condense, or paraphrase. The text differences between the input and output must be minimal.
    2. **Minimal Editorial Cleanups Only:** Remove verbal filler words ("um", "uh", stuttered duplicate words), fix capitalization at sentence starts, and correct punctuation. Do NOT alter the vocabulary or sentence structure.
    3. **Format All Math and Variables as LaTeX:** Convert all mathematical symbols, variables, indices, loss functions, vectors, matrices, and Greek letters into LaTeX (e.g., $W$, $b$, $x_i$, $y_i$, $\hat{y}$, $\alpha$, $\lambda$, $L_1$, $L_2$, $\sum$, $$...$$).
    4. **Paragraph Structure:** Organize the verbatim sentences into natural, readable paragraphs separated by a blank line (2–4 sentences per paragraph).
    5. **Output Only Markdown:** Output ONLY the edited Markdown text. Do NOT add preamble, commentary, meta-talk, notes, or extra markdown headers.

    Lecture Transcript:
    __TRANSCRIPT_PLACEHOLDER__

    Markdown Lecture Note:
    """
    prompt = prompt_template.replace('__TRANSCRIPT_PLACEHOLDER__', transcript_text)

    for attempt in range(retries):
        try:
            if OLLAMA_CLIENT is None:
                raise ConnectionError("Ollama client is not available.")
            response = OLLAMA_CLIENT.generate(model=model_name, prompt=prompt)
            raw_text = response['response'] if isinstance(response, dict) else getattr(response, 'response', str(response))
            cleaned = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
            if cleaned:
                return cleaned
        except Exception as e:
            log(f"Attempt {attempt + 1}/{retries} failed for chunk. Retrying in {delay}s...", always=True)
            time.sleep(delay)

    # Fallback to verbatim text if LLM call fails
    return transcript_text

def process_text_chunk(transcript_text, model_name, retries=3, delay=5):
    if not transcript_text.strip():
        return ""
    
    prompt_template = """
    You are a technical editor converting a video lecture transcript into a Markdown lecture note compatible with MarkText.

    Your goal is to make ONLY minimal editorial changes to convert spoken transcript into clean lecture notes while keeping all content, phrasing, and full context intact.

    Guidelines:
    1. **Strictly Preserve Verbatim Text & Context:** Keep the speaker's exact words, explanations, examples, and phrasing. Do NOT rewrite, summarize, condense, or paraphrase. The text differences between the input and output must be minimal.
    2. **Minimal Editorial Cleanups Only:** Remove verbal filler words ("um", "uh"), fix capitalization, and correct punctuation. Do NOT alter vocabulary or sentence structure.
    3. **Format All Math as LaTeX:** Convert all variables (e.g., $W$, $b$, $x_i$, $y_i$, $\hat{y}$), Greek letters ($\alpha$, $\beta$, $\lambda$), and equations into inline ($...$) or block ($$...$$) LaTeX.
        - Subscripts: use simple underscore ($f_1$, $L_1$, $L_2$, $x_i$).
        - Text inside math: use `\mathrm{...}` (e.g., $\mathrm{Loss}(W)$).
        - Absolute values: use `\lvert` and `\rvert`.
    4. **Paragraph Structure:** Group the verbatim sentences into readable paragraphs (2–4 sentences each) separated by a blank line.
    5. **Output Only Markdown:** Output ONLY the final edited text without conversational intro/outro or meta commentary.

    Lecture Transcript:
    __TRANSCRIPT_PLACEHOLDER__
    
    Markdown Lecture Note:
    """
    prompt = prompt_template.replace('__TRANSCRIPT_PLACEHOLDER__', transcript_text)

    last_exception = None
    for attempt in range(retries):
        try:
            if OLLAMA_CLIENT is None:
                raise ConnectionError("Ollama client is not available.")
            response = OLLAMA_CLIENT.generate(model=model_name, prompt=prompt)
            raw_text = response['response'] if isinstance(response, dict) else getattr(response, 'response', str(response))
            cleaned = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
            return cleaned
        except Exception as e:
            last_exception = e
            log(f"Attempt {attempt + 1}/{retries} failed for chunk. Retrying in {delay}s...", always=True)
            time.sleep(delay)

    log(f"An error occurred while processing a chunk with Ollama after {retries} attempts: {last_exception}", always=True)
    log("The chunk will be skipped to prevent corrupting the output.", always=True)
    return ""
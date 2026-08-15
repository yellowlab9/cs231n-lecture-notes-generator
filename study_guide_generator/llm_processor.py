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

def process_text_chunk(transcript_text, model_name, retries=3, delay=5):
    if not transcript_text.strip():
        return ""
    
    # Use a placeholder and .replace() to build the prompt. This is the most robust
    # method, as it avoids any and all parsing errors from special characters
    # like backslashes and curly braces that are common in the LaTeX examples.
    prompt_template = """
    You are an expert technical editor creating a Markdown study guide compatible with MarkText from a raw lecture transcript. Your primary goal is to clean the text while preserving the speaker's original wording and flow as closely as possible.

    Your tasks are:
    1.  **Clean, Don't Rewrite:** Correct grammar, fix typos, and remove filler words (like "um", "uh"). Do NOT rephrase sentences or alter the core vocabulary. The output should feel like a polished version of the original speech, not a summary.
    2.  **Aggressively Format All Math as LaTeX:** This is the most critical task. Be extremely vigilant in identifying and formatting anything that looks like a mathematical concept, variable, or equation into LaTeX for MarkText.
        - **Variables and Symbols:** Convert all single-letter variables (like `W`, `b`, `x_i`, `y_hat`), Greek letters (`alpha`, `beta`), and mathematical symbols into inline LaTeX (e.g., $W$, $b$, $x_i$, $\hat{y}$, $\alpha$).
        - **Equations:** Wrap all equations, from simple assignments to complex loss functions, in display LaTeX (`$$...$$`).
        - **MarkText Compatibility:**
            - For subscripts, use a simple underscore. For example, convert "f sub 1" into $f_1$. Common ML terms like "L1" or "L2" should become $L_1$ and $L_2$. Do not write out the word "sub".
            - For text inside equations, always use `\mathrm{...}`, not `\text{...}`. For example: $$\mathrm{Data~Loss}$$.
            - When referring to a loss for a specific variable, prefer functional notation, e.g., $\mathrm{Loss}(W_1)$ instead of $\mathrm{Loss}_{W_1}$.
            - Do not escape underscores in commands like `\sum_{...}`.
            - For absolute value, use `\lvert` and `\rvert` for correct spacing, e.g., $\lvert -5 \rvert$.
    3.  **Create Short, Readable Paragraphs:** As a technical writer, you know that long walls of text are hard to read. Group sentences into short paragraphs of 2-4 sentences each. Start a new paragraph (with a double newline) to break up the text for readability.
    4.  **Output Only Markdown:** Produce only the final, cleaned Markdown text. Do not add any headers, introductions, horizontal rules (`---` or `***`), or other text that wasn't in the original transcript.

    Raw Transcript:
    __TRANSCRIPT_PLACEHOLDER__
    
    Cleaned Markdown:
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
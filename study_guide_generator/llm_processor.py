import ollama
from .utils import log

def process_text_chunk(transcript_text, model_name):
    if not transcript_text.strip(): return ""
    
    prompt = f"""
    You are an expert technical editor writing a Markdown study guide for a deep learning lecture.
    
    Task:
    1. Clean up the messy spoken transcript (fix errors, remove filler words).
    2. Format all mathematical equations, loss functions, and variables in LaTeX ($...$ or $$...$$).
    3. Do NOT add any conversational filler, introductions, or conclusions. 
    4. Keep the Markdown text *extra close* to the exact wording, vocabulary, and flow of the original transcript. 
    
    Raw Transcript:
    {transcript_text}
    
    Output ONLY the final, cleaned Markdown text.
    """
    try:
        client = ollama.Client(host="http://127.0.0.1:11434")
        response = client.generate(model=model_name, prompt=prompt)
        return response['response'].strip()
    except Exception as e:
        log(f"Ollama Error: {e}", always=True)
        return transcript_text
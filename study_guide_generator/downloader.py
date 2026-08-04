import os
import re
import requests
import yt_dlp
from .utils import log

def download_pdf(pdf_source):
    pdf_path = pdf_source.split("/")[-1] if pdf_source.startswith("http") else pdf_source
    if os.path.exists(pdf_path): return pdf_path
    log(f"Downloading PDF from {pdf_source}...", always=True)
    r = requests.get(pdf_source)
    with open(pdf_path, 'wb') as f: f.write(r.content)
    return pdf_path

def download_media(url):
    video_id = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url).group(1)
    base = f"lecture_{video_id}"
    video_path = f"{base}.mp4"

    sub_file = None
    for ext in ['.en.vtt', '.en.srt', '.vtt', '.srt']:
        if os.path.exists(f"{base}{ext}"):
            sub_file = f"{base}{ext}"
            break

    if not os.path.exists(video_path) or not sub_file:
        log(f"Downloading video and transcript from {url}...", always=True)
        
        ydl_opts = {
            'format': 'bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'merge_output_format': 'mp4',
            'outtmpl': f'{base}.%(ext)s',
            'quiet': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],
            'subtitlesformat': 'vtt/srt/best'
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([url])

        for ext in ['.en.vtt', '.en.srt', '.vtt', '.srt']:
            if os.path.exists(f"{base}{ext}"):
                sub_file = f"{base}{ext}"
                break

        if not sub_file:
            log("Warning: No transcript could be downloaded.", always=True)

    return video_path, sub_file
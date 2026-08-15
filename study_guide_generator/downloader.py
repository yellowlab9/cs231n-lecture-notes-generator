import os
import yt_dlp
from .utils import log

def _find_subtitle_file(base):
    for ext in ['.en-US.vtt', '.en.vtt', '.en-orig.vtt', '.en-US.srt', '.en.srt', '.vtt', '.srt']:
        if os.path.exists(f"{base}{ext}"):
            return f"{base}{ext}"
    # Fallback to any subtitle file starting with the base name (e.g. .en-orig.vtt, .en-US.vtt)
    dirname = os.path.dirname(base) or '.'
    if not os.path.exists(dirname):
        return None
    prefix = os.path.basename(base)
    for fname in os.listdir(dirname):
        if fname.startswith(prefix) and (fname.endswith('.vtt') or fname.endswith('.srt')):
            return os.path.join(dirname, fname) if dirname != '.' else fname
    return None

def download_media(playlist_url, index=1):
    """
    Downloads the video and subtitles for a specific item in a playlist.
    Returns (video_path, sub_file_path).
    """
    index = max(1, int(index))
    base = f"lecture_{index:02d}"
    video_path = f"{base}.mp4"

    # Skip download if already cached locally
    sub_file = _find_subtitle_file(base)
    if os.path.exists(video_path) and sub_file:
        log(f"[*] Media for {base} found locally. Skipping download.", always=True)
        return video_path, sub_file

    log(f"Fetching video and transcript (item #{index:02d}) from playlist...", always=True)

    ydl_opts = {
        'format': 'bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'merge_output_format': 'mp4',
        'outtmpl': f'{base}.%(ext)s',
        'playlist_items': str(index),
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en', 'en-US'],
        'subtitlesformat': 'vtt/srt/best',
        'quiet': True,
        'overwrites': False,
        'ignoreerrors': True,
        'nopart': True,
        'windowsfilenames': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(str(playlist_url).strip(), download=True)

    sub_file = _find_subtitle_file(base)

    if not sub_file:
        log("Warning: No transcript could be downloaded.", always=True)

    return video_path, sub_file
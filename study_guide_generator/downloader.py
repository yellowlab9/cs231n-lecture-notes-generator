import os
import yt_dlp
from .utils import log

import shutil

def _find_subtitle_file(base):
    for ext in ['.en-US.vtt', '.en.vtt', '.en-orig.vtt', '.en-US.srt', '.en.srt', '.vtt', '.srt']:
        if os.path.exists(f"{base}{ext}"):
            return f"{base}{ext}"
    # Fallback to any subtitle file starting with the base name
    dirname = os.path.dirname(base) or '.'
    if not os.path.exists(dirname):
        return None
    prefix = os.path.basename(base)
    for fname in os.listdir(dirname):
        if fname.startswith(prefix) and (fname.endswith('.vtt') or fname.endswith('.srt')):
            return os.path.join(dirname, fname) if dirname != '.' else fname
    return None

def _migrate_legacy_media(base_name, media_dir):
    """
    Migrates any legacy media files in the root folder into media_dir.
    """
    os.makedirs(media_dir, exist_ok=True)
    # Check for legacy video in root
    legacy_vid = f"{base_name}.mp4"
    target_vid = os.path.join(media_dir, legacy_vid)
    if os.path.exists(legacy_vid) and not os.path.exists(target_vid):
        log(f"[*] Moving legacy '{legacy_vid}' into '{media_dir}/'...", always=True)
        shutil.move(legacy_vid, target_vid)

    # Check for legacy subtitle files in root
    for ext in ['.en-US.vtt', '.en.vtt', '.en-orig.vtt', '.en-US.srt', '.en.srt', '.vtt', '.srt']:
        legacy_sub = f"{base_name}{ext}"
        target_sub = os.path.join(media_dir, legacy_sub)
        if os.path.exists(legacy_sub) and not os.path.exists(target_sub):
            log(f"[*] Moving legacy '{legacy_sub}' into '{media_dir}/'...", always=True)
            shutil.move(legacy_sub, target_sub)

def download_subtitles(playlist_url, index=1, media_dir=os.path.join("lectures_cache", "media")):
    """
    Downloads subtitles only. If creator-uploaded 'en-US' is available, downloads only 'en-US'.
    Returns (sub_file_path, is_creator_subtitle).
    """
    index = max(1, int(index))
    base_name = f"lecture_{index:02d}"
    os.makedirs(media_dir, exist_ok=True)
    _migrate_legacy_media(base_name, media_dir)
    base = os.path.join(media_dir, base_name)

    # Check local cache in media_dir
    sub_file = _find_subtitle_file(base)
    if sub_file:
        is_creator = sub_file.endswith('.en-US.vtt') or sub_file.endswith('.en-US.srt')
        log(f"[*] Subtitle file '{sub_file}' found locally (Creator uploaded: {is_creator}). Skipping download.", always=True)
        return sub_file, is_creator

    log(f"Checking and downloading subtitles for item #{index:02d} into '{media_dir}'...", always=True)

    # 1. Probe available subtitles
    ydl_opts_probe = {
        'playlist_items': str(index),
        'extract_flat': False,
        'quiet': True,
    }

    sub_lang = 'en'
    write_manual = False
    write_auto = True
    is_creator_subtitle = False

    with yt_dlp.YoutubeDL(ydl_opts_probe) as ydl:
        try:
            info = ydl.extract_info(str(playlist_url).strip(), download=False)
            if info:
                entries = [e for e in (info.get('entries') or []) if e]
                target = entries[0] if entries else info
                manual_subs = target.get('subtitles') or {}
                if 'en-US' in manual_subs:
                    sub_lang = 'en-US'
                    write_manual = True
                    write_auto = False
                    is_creator_subtitle = True
                    log("  -> Creator-uploaded 'en-US' subtitles detected.", always=True)
                elif 'en' in manual_subs:
                    sub_lang = 'en'
                    write_manual = True
                    write_auto = False
                    is_creator_subtitle = True
                    log("  -> Creator-uploaded 'en' subtitles detected.", always=True)
                else:
                    log("  -> Using auto-generated captions.", always=True)
        except Exception as e:
            log(f"Subtitle probe warning: {e}. Falling back to default.", always=True)

    # 2. Download selected subtitle
    ydl_opts_sub = {
        'skip_download': True,
        'outtmpl': f'{base}.%(ext)s',
        'playlist_items': str(index),
        'writesubtitles': write_manual,
        'writeautomaticsub': write_auto,
        'subtitleslangs': [sub_lang],
        'subtitlesformat': 'vtt/srt/best',
        'quiet': True,
        'nocheckcertificate': True,
        'remote_components': ['ejs:github'],
        'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
        'overwrites': False,
        'ignoreerrors': True,
        'windowsfilenames': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts_sub) as ydl:
        ydl.extract_info(str(playlist_url).strip(), download=True)

    sub_file = _find_subtitle_file(base)
    if not sub_file:
        log("Warning: No transcript could be downloaded.", always=True)

    return sub_file, is_creator_subtitle

def download_video(playlist_url, index=1, media_dir=os.path.join("lectures_cache", "media")):
    """
    Downloads only the video and audio stream for a specific playlist item.
    Returns video_path.
    """
    index = max(1, int(index))
    base_name = f"lecture_{index:02d}"
    os.makedirs(media_dir, exist_ok=True)
    _migrate_legacy_media(base_name, media_dir)
    base = os.path.join(media_dir, base_name)
    video_path = f"{base}.mp4"

    if os.path.exists(video_path):
        log(f"[*] Video '{video_path}' found locally. Skipping download.", always=True)
        return video_path

    log(f"Downloading video (item #{index:02d}) into '{media_dir}'...", always=True)

    ydl_opts_vid = {
        'format': 'bestvideo[height<=1080]+bestaudio/best[height<=1080]/best',
        'merge_output_format': 'mp4',
        'outtmpl': f'{base}.%(ext)s',
        'playlist_items': str(index),
        'quiet': False,
        'nocheckcertificate': True,
        'remote_components': ['ejs:github'],
        'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
        'overwrites': False,
        'ignoreerrors': False,
        'nopart': True,
        'windowsfilenames': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts_vid) as ydl:
        ydl.extract_info(str(playlist_url).strip(), download=True)

    return video_path

def download_media(playlist_url, index=1, media_dir=os.path.join("lectures_cache", "media")):
    """
    Downloads subtitles and video separately for a specific playlist item.
    Skips downloading if the video and subtitle files are already cached locally.
    Returns (video_path, sub_file_path, is_creator_subtitle).
    """
    index = max(1, int(index))
    base_name = f"lecture_{index:02d}"
    os.makedirs(media_dir, exist_ok=True)
    _migrate_legacy_media(base_name, media_dir)
    base = os.path.join(media_dir, base_name)
    video_path = f"{base}.mp4"
    sub_file = _find_subtitle_file(base)

    # Fast check: if both video and transcript are already downloaded, skip completely
    if os.path.exists(video_path) and sub_file:
        is_creator = sub_file.endswith('.en-US.vtt') or sub_file.endswith('.en-US.srt')
        log(f"[*] Media for {base_name} already found locally ('{video_path}', '{sub_file}'). Skipping all downloads.", always=True)
        return video_path, sub_file, is_creator

    sub_file, is_creator_subtitle = download_subtitles(playlist_url, index, media_dir=media_dir)
    video_path = download_video(playlist_url, index, media_dir=media_dir)
    return video_path, sub_file, is_creator_subtitle
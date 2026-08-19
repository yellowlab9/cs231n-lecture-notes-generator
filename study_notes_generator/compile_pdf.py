import os
import sys
import glob
import argparse
import subprocess

def compile_single_note_to_pdf(md_path, pdf_path, engine="xelatex", fontsize=None):
    """
    Compiles a single Markdown note to a PDF via Pandoc and XeLaTeX.
    """
    lec_dir = os.path.dirname(md_path)
    pdf_dir = os.path.dirname(pdf_path)
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_filename = os.path.basename(pdf_path)

    # Check if target PDF is locked by an open PDF reader on Windows
    if os.path.exists(pdf_path):
        try:
            with open(pdf_path, "a+"):
                pass
        except PermissionError:
            print(f"\n[ERROR] Cannot overwrite '{pdf_filename}' because it is currently open in a PDF reader or browser.", file=sys.stderr)
            print(f"[TIP] Please close '{pdf_filename}' in your PDF viewer and run the command again.\n", file=sys.stderr)
            return 1

    lua_filter = os.path.abspath(os.path.join(os.path.dirname(__file__), "html_filter.lua"))

    print(f"Converting: {os.path.basename(md_path)} -> {os.path.relpath(pdf_path)}")
    cmd = [
        "pandoc",
        os.path.basename(md_path),
        "-o",
        pdf_path,
        f"--pdf-engine={engine}",
        f"--lua-filter={lua_filter}"
    ]

    if fontsize:
        fs_clean = str(fontsize).strip()
        if fs_clean.isdigit():
            fs_clean = f"{fs_clean}pt"
        cmd.extend(["-V", "documentclass=extarticle", "-V", f"fontsize={fs_clean}"])

    res = subprocess.run(cmd, cwd=lec_dir)
    if res.returncode == 0:
        print(f"Done! Generated {pdf_path} ({os.path.getsize(pdf_path)} bytes)")
        return 0
    else:
        print(f"Pandoc failed with exit code {res.returncode}", file=sys.stderr)
        return res.returncode

def compile_lecture_pdf(lecture, engine="xelatex", fontsize=None):
    """
    Finds the markdown study guide for the given lecture and compiles it to PDF.
    """
    lec_str = str(lecture).strip()
    if lec_str.startswith("lecture_"):
        lec_str = lec_str.replace("lecture_", "")
    if lec_str.isdigit():
        lec_num = f"{int(lec_str):02d}"
    else:
        lec_num = lec_str

    patterns = [
        os.path.join("**", "lectures", f"lecture_{lec_num}*.md"),
        os.path.join("lectures", f"lecture_{lec_num}*.md"),
    ]

    md_matches = []
    for pat in patterns:
        md_matches.extend(glob.glob(pat, recursive=True))

    if not md_matches:
        print(f"Error: No markdown study guide found matching lecture {lec_str} (or {lec_num}).", file=sys.stderr)
        print("Please generate the notes first (e.g. 'make notes LECTURE=" + lec_str + "')", file=sys.stderr)
        return 1

    md_path = os.path.abspath(md_matches[0])
    lec_dir = os.path.dirname(md_path)
    course_dir = os.path.dirname(lec_dir) if os.path.basename(lec_dir) == "lectures" else "."
    pdf_dir = os.path.join(course_dir, "lectures_pdf") if course_dir != "." else "lectures_pdf"
    base_name = os.path.splitext(os.path.basename(md_path))[0]
    pdf_path = os.path.join(pdf_dir, f"{base_name}.pdf")

    return compile_single_note_to_pdf(md_path, pdf_path, engine=engine, fontsize=fontsize)

def compile_all_lectures(force=False, engine="xelatex", fontsize=None):
    """
    Discovers all lecture notes and compiles any missing or outdated PDFs.
    """
    patterns = [
        os.path.join("**", "lectures", "lecture_*_notes_*.md"),
        os.path.join("lectures", "lecture_*_notes_*.md"),
    ]
    md_matches = []
    for pat in patterns:
        md_matches.extend(glob.glob(pat, recursive=True))

    # Deduplicate paths
    unique_mds = sorted(list({os.path.abspath(p) for p in md_matches}))

    if not unique_mds:
        print("No lecture markdown notes found in lectures/ directory.", file=sys.stderr)
        return 0

    compiled_count = 0
    skipped_count = 0
    errors = 0

    print(f"Found {len(unique_mds)} lecture note(s). Checking for new or updated files...")

    for md_path in unique_mds:
        lec_dir = os.path.dirname(md_path)
        course_dir = os.path.dirname(lec_dir) if os.path.basename(lec_dir) == "lectures" else "."
        pdf_dir = os.path.join(course_dir, "lectures_pdf") if course_dir != "." else "lectures_pdf"
        base_name = os.path.splitext(os.path.basename(md_path))[0]
        pdf_path = os.path.join(pdf_dir, f"{base_name}.pdf")

        is_missing = not os.path.exists(pdf_path)
        is_outdated = is_missing or (os.path.getmtime(md_path) > os.path.getmtime(pdf_path))

        if is_outdated or force:
            status_tag = "[NEW]" if is_missing else "[MODIFIED]"
            if force and not is_outdated:
                status_tag = "[FORCE]"
            print(f"\n{status_tag} {os.path.basename(md_path)} -> {os.path.basename(pdf_path)}")
            ret = compile_single_note_to_pdf(md_path, pdf_path, engine=engine, fontsize=fontsize)
            if ret == 0:
                compiled_count += 1
            else:
                errors += 1
        else:
            print(f"[UP TO DATE] Skipping {os.path.basename(md_path)}")
            skipped_count += 1

    print(f"\nSummary: {compiled_count} compiled, {skipped_count} up to date, {errors} errors.")
    return 1 if errors > 0 else 0

def main():
    parser = argparse.ArgumentParser(description="Compile lecture Markdown study guides to PDF.")
    parser.add_argument("--lecture", "-l", default=None, help="Lecture number or prefix (e.g. 1, 01, lecture_01).")
    parser.add_argument("--all", "-a", action="store_true", help="Compile all new or modified lecture notes to PDF.")
    parser.add_argument("--force", action="store_true", help="Force recompilation even if PDFs are already up to date.")
    parser.add_argument("--engine", "-e", default="xelatex", help="LaTeX engine to use (default: xelatex).")
    parser.add_argument("--fontsize", "--font_size", "-f", default=None, help="Font size override for PDF (e.g. 10pt, 12pt, 14pt).")
    args = parser.parse_args()

    if args.all or args.lecture is None:
        sys.exit(compile_all_lectures(force=args.force, engine=args.engine, fontsize=args.fontsize))
    else:
        sys.exit(compile_lecture_pdf(args.lecture, engine=args.engine, fontsize=args.fontsize))

if __name__ == "__main__":
    main()

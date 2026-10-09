"""
app.py - Flask web app for the Resume Screening System.
Run with: python app.py  (then open http://127.0.0.1:5000)
"""

import os

from flask import Flask, flash, redirect, render_template, request, url_for

from resume_parser import allowed_file, extract_text, unique_filename
from nlp_processor import screen_resumes, get_nlp

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "resume-screening-dev-key")

# In-memory store for the latest screening (single-user local app, nothing on disk).
# This keeps uploaded resume text out of cookies and off the disk.
LAST_RESULTS = []
LAST_JOB_KEYWORDS = []
LAST_WARNINGS = []
LAST_SUMMARY = {"total": 0, "processed": 0, "failed": 0, "top_score": 0.0}

MAX_FILES = 20
MAX_FILE_MB = 10


@app.route("/", methods=["GET"])
def index():
    spacy_ok = get_nlp() is not None
    # Read-only: the Overview cards reflect the latest screening stored in
    # LAST_SUMMARY. Nothing is mutated here, so revisits never double-count.
    return render_template("index.html", spacy_ok=spacy_ok,
                           summary=LAST_SUMMARY)


@app.route("/screen", methods=["POST"])
def screen():
    global LAST_RESULTS, LAST_JOB_KEYWORDS, LAST_WARNINGS, LAST_SUMMARY

    job_description = (request.form.get("job_description") or "").strip()
    uploaded = request.files.getlist("resumes")

    # Filter out empty file inputs (browser sends one empty entry if nothing chosen)
    uploaded = [f for f in uploaded if f and f.filename and f.filename.strip()]

    if not job_description:
        flash("Please paste a job description before screening.", "danger")
        return redirect(url_for("index"))

    if len(job_description) < 50:
        flash("Job description looks very short (under 50 characters). Results may be unreliable.", "warning")

    if not uploaded:
        flash("Please upload at least one PDF or DOCX resume.", "danger")
        return redirect(url_for("index"))

    if len(uploaded) > MAX_FILES:
        flash(f"Too many files. Maximum {MAX_FILES} resumes at a time. Only the first {MAX_FILES} will be used.", "warning")
        total_uploaded = len(uploaded)
        uploaded = uploaded[:MAX_FILES]
    else:
        total_uploaded = len(uploaded)

    resume_texts = {}
    warnings = []
    seen = set()
    failed = 0

    for f in uploaded:
        original = os.path.basename(f.filename.strip())
        if not allowed_file(original):
            warnings.append(f"Skipped '{original}': only PDF and DOCX files are allowed.")
            failed += 1
            continue
        # Pre-read size guard: skip obviously oversized uploads without
        # loading them fully into memory (content_length may be None).
        try:
            hint = f.content_length
        except Exception:
            hint = None
        if hint is not None and hint > MAX_FILE_MB * 1024 * 1024:
            warnings.append(f"Skipped '{original}': file exceeds {MAX_FILE_MB} MB.")
            failed += 1
            continue
        try:
            file_bytes = f.read()
        except Exception as e:
            warnings.append(f"Skipped '{original}': could not read upload ({e}).")
            failed += 1
            continue
        if not file_bytes or len(file_bytes) == 0:
            warnings.append(f"Skipped '{original}': file is empty.")
            failed += 1
            continue
        if len(file_bytes) > MAX_FILE_MB * 1024 * 1024:
            warnings.append(f"Skipped '{original}': file exceeds {MAX_FILE_MB} MB.")
            failed += 1
            continue

        fname = unique_filename(original, seen)
        seen.add(fname)
        text, error = extract_text(fname, file_bytes)
        if error or not text or not text.strip():
            warnings.append(f"Skipped '{fname}': {error or 'no readable text found.'}")
            failed += 1
            continue
        resume_texts[fname] = text

    if not resume_texts:
        for w in warnings:
            flash(w, "warning")
        flash("No resumes could be processed. Check file types and that PDFs contain selectable text (not scanned images).", "danger")
        return redirect(url_for("index"))

    try:
        ranked, job_keywords = screen_resumes(job_description, resume_texts)
    except Exception as e:
        flash(f"Screening failed due to an NLP error: {e}", "danger")
        return redirect(url_for("index"))

    LAST_RESULTS = ranked
    LAST_JOB_KEYWORDS = job_keywords
    LAST_WARNINGS = warnings
    top_score = ranked[0]["score"] if ranked else 0.0
    LAST_SUMMARY = {
        "total": total_uploaded,
        "processed": len(ranked),
        "failed": failed,
        "top_score": top_score,
    }

    return render_template(
        "results.html",
        results=ranked,
        job_keywords=job_keywords[:30],
        warnings=warnings,
        summary=LAST_SUMMARY,
    )


@app.route("/clear", methods=["GET"])
def clear():
    global LAST_RESULTS, LAST_JOB_KEYWORDS, LAST_WARNINGS, LAST_SUMMARY
    LAST_RESULTS = []
    LAST_JOB_KEYWORDS = []
    LAST_WARNINGS = []
    LAST_SUMMARY = {"total": 0, "processed": 0, "failed": 0, "top_score": 0.0}
    flash("Cleared. You can screen a new set of resumes.", "info")
    return redirect(url_for("index"))


if __name__ == "__main__":
    # Debug off by default so student laptops are not exposed; use 127.0.0.1 only.
    app.run(host="127.0.0.1", port=5000, debug=True)

"""
nlp_processor.py - NLP preprocessing + TF-IDF similarity ranking.
Uses spaCy for lemmatization when available, with a safe fallback
so the project still runs even if the spaCy model is not downloaded.
"""

import re
from collections import Counter

_nlp = None
_nlp_tried = False


def get_nlp():
    """Load spaCy English model once. Returns None if unavailable."""
    global _nlp, _nlp_tried
    if _nlp_tried:
        return _nlp
    _nlp_tried = True
    try:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Model not downloaded - fall back to rule-based processing.
            _nlp = None
    except ImportError:
        _nlp = None
    return _nlp


def _fallback_tokens(text):
    """Simple tokenizer used when spaCy model is unavailable."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    words = re.findall(r"[a-z]{3,}", text.lower())
    return [w for w in words if w not in ENGLISH_STOP_WORDS]


def clean_for_tfidf(raw_text, nlp=None, max_chars=500000):
    """Normalize text into a space-separated token string for TF-IDF.

    Steps: lowercase -> spaCy tokenize/lemmatize (or fallback) ->
    remove stop words, non-alphabetic tokens, short tokens.
    """
    if not raw_text or not raw_text.strip():
        return ""
    text = raw_text[:max_chars]
    if nlp is not None:
        try:
            doc = nlp(text.lower())
            tokens = [
                t.lemma_.lower().strip()
                for t in doc
                if t.is_alpha and not t.is_stop and len(t.lemma_) > 2
            ]
            # Keep only clean a-z tokens
            tokens = [re.sub(r"[^a-z]", "", t) for t in tokens]
            tokens = [t for t in tokens if len(t) > 2]
            return " ".join(tokens)
        except Exception:
            pass  # fall through to fallback
    return " ".join(_fallback_tokens(text))


def extract_keywords(raw_text, nlp=None, max_keywords=80):
    """Extract meaningful keywords with frequencies.

    Returns a list of (keyword, count) sorted by frequency (most common first).
    """
    if not raw_text or not raw_text.strip():
        return []
    text = raw_text[:500000]
    if nlp is not None:
        try:
            doc = nlp(text.lower())
            tokens = []
            for t in doc:
                if t.is_alpha and not t.is_stop and len(t.lemma_) > 2:
                    lemma = re.sub(r"[^a-z]", "", t.lemma_.lower().strip())
                    if len(lemma) > 2:
                        tokens.append(lemma)
            if tokens:
                return Counter(tokens).most_common(max_keywords)
        except Exception:
            pass
    tokens = _fallback_tokens(text)
    return Counter(tokens).most_common(max_keywords)


def screen_resumes(job_description, resume_texts):
    """Compare each resume against the job description.

    Args:
        job_description: raw JD string.
        resume_texts: dict {filename: extracted_text}.

    Returns:
        ranked_results: list of dicts sorted by score (high -> low), each with:
            rank, filename, score (0-100 float), matched_keywords (list),
            missing_keywords (list), preview, char_count, text (full text).
        job_keyword_list: list of JD keywords ordered by importance.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    nlp = get_nlp()

    job_clean = clean_for_tfidf(job_description, nlp)
    job_kw_counts = extract_keywords(job_description, nlp)
    job_kw_set = {k for k, _ in job_kw_counts}
    job_kw_order = [k for k, _ in job_kw_counts]  # already frequency-ordered

    filenames = list(resume_texts.keys())
    resume_cleans = [clean_for_tfidf(resume_texts[f], nlp) for f in filenames]

    # TF-IDF over JD + all resumes together so they share one vocabulary.
    # Empty documents are replaced with a placeholder to avoid empty-vocabulary errors.
    corpus = [job_clean if job_clean.strip() else "emptydocument"]
    for rc in resume_cleans:
        corpus.append(rc if rc.strip() else "emptydocument")

    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)
        job_vec = tfidf_matrix[0]
        resume_vecs = tfidf_matrix[1:]
        sims = cosine_similarity(job_vec, resume_vecs)[0]
    except ValueError:
        # Empty vocabulary (e.g. JD is only stop words) -> all scores 0.
        sims = [0.0] * len(filenames)

    results = []
    for i, fname in enumerate(filenames):
        raw = resume_texts[fname] or ""
        score = round(float(sims[i]) * 100, 2) if i < len(sims) else 0.0
        # Clamp to [0, 100] to guard against float noise.
        score = max(0.0, min(100.0, score))

        resume_kw_set = {k for k, _ in extract_keywords(raw, nlp)}
        matched = sorted(job_kw_set & resume_kw_set)
        # Missing keywords kept in JD frequency order (most important first), top 20.
        missing = [k for k in job_kw_order if k not in resume_kw_set][:20]

        text_stripped = raw.strip()
        preview = text_stripped[:600] + ("..." if len(text_stripped) > 600 else "")

        results.append({
            "filename": fname,
            "score": score,
            "matched_keywords": matched[:30],
            "matched_count": len(job_kw_set & resume_kw_set),
            "missing_keywords": missing,
            "preview": preview,
            "char_count": len(raw),
            "text": text_stripped,
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    for idx, r in enumerate(results, start=1):
        r["rank"] = idx

    return results, job_kw_order

# Resume Screening System Using NLP

A beginner-friendly college mini project. A Flask web app that screens multiple
resumes (PDF / DOCX) against a job description using **traditional NLP** and ranks
candidates by **textual similarity**.

> Match % = cosine similarity (TF-IDF vectors) × 100.
> It measures **text overlap**, not probability of hiring success.

---

## Features

- Upload multiple PDF + DOCX resumes (drag & drop, up to 20, 10 MB each)
- Paste job description in a large text area
- File-type validation; empty / unreadable / scanned PDFs handled gracefully
- Text extraction (pypdf + python-docx), in-memory only — uploads are never saved or executed
- NLP preprocessing: lowercase, tokenize, stop-word removal, lemmatization (spaCy `en_core_web_sm` when installed, otherwise a built-in fallback)
- TF-IDF vectorization + cosine similarity (scikit-learn)
- Ranked results table: rank, filename, match %, matching keywords, missing JD keywords
- Expandable candidate details (extracted text + keywords)
- Summary cards (total / processed / top score), progress-bar score indicators
- Clear / reset workflow, error messages, loading indicator

## NLP concepts used

| Concept | Where | Why |
|---|---|---|
| Tokenization | `nlp_processor.clean_for_tfidf` | Split text into words |
| Stop-word removal | spaCy `is_stop` / sklearn `ENGLISH_STOP_WORDS` | Drop low-signal words (the, and) |
| Lemmatization | spaCy `token.lemma_` | `developing` → `develop` so variants match |
| TF-IDF | `sklearn.feature_extraction.text.TfidfVectorizer` | Weight rare, informative terms higher |
| Cosine similarity | `sklearn.metrics.pairwise.cosine_similarity` | Angle between JD and resume vectors (0–1) |

## Project structure

```
resume-screening-system/
  app.py               # Flask routes (/, /screen, /clear)
  nlp_processor.py     # preprocessing + TF-IDF + ranking
  resume_parser.py     # PDF / DOCX text extraction
  requirements.txt
  README.md
  .gitignore
  templates/
    index.html         # upload + JD form
    results.html       # ranked table + details
  static/
    css/
      style.css
```

## Installation (Windows PowerShell, VS Code terminal)

```powershell
# 1. Go to the project folder
cd E:\NLP_Project

# 2. Create a virtual environment
py -m venv venv

# 3. Activate it
.\venv\Scripts\Activate.ps1
# If activation is blocked: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 4. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. (Recommended) spaCy English model for lemmatization.
#    The app still runs without it (uses a fallback tokenizer), but scores are better with it.
python -m spacy download en_core_web_sm
```

## Run the app

```powershell
cd E:\NLP_Project
.\venv\Scripts\Activate.ps1
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

## Usage

1. Drag & drop (or browse) 1–20 resumes (`.pdf` / `.docx`).
2. Paste the job description (aim for 100+ words).
3. Click **Screen Resumes**.
4. Review the ranked table, expand **View** for any candidate's extracted text.
5. Click **Screen Another Set** / **Clear** to start over.

## Sample job description (for testing)

```
We are hiring a Python Developer (Fresher / Junior) for our software team.

Required skills: Python, Flask, REST APIs, SQL, Git, HTML, CSS.
Good to have: pandas, scikit-learn, NLP, machine learning basics, data analysis.
Responsibilities: develop web applications, write clean documented code,
work with databases, debug and test software, collaborate with the team.

Education: Bachelor's degree in Computer Science, IT, or related field.
```

**How to make sample resumes:** create 2–4 DOCX files in Word/Google Docs —
one strong match (Python, Flask, SQL, REST API project), one partial match
(Java, HTML only), one weak match (sales/marketing). Export one as PDF too,
then upload them together and confirm the strong match ranks #1.

Expected behaviour: the Python/Flask resume scores highest; missing-keyword
column shows JD terms it lacks (e.g. `pandas`, `scikit-learn`).

## Limitations

- Pure keyword/text overlap — no semantic understanding (e.g. "JS" ≠ "JavaScript" unless both appear).
- Scanned/image PDFs with no text layer yield no text (needs OCR — out of scope).
- Very short JDs (<50 chars) give unreliable scores (app warns you).
- Results live only in server memory; restarting the app clears them.
- Single-user local design — no login, no database (by design).

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: flask` | venv not activated or `pip install -r requirements.txt` not run |
| spaCy warning on home page | Optional — run `python -m spacy download en_core_web_sm`, then restart |
| Resume skipped: no extractable text | PDF is scanned; export/print it as text PDF or use DOCX |
| Port 5000 busy | `python app.py` error → run on another port: set `port=5001` in `app.py` |
| Execution policy error in PowerShell | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `ModuleNotFoundError: pytest` | run `pip install -r requirements.txt` (includes pytest) |

## Running tests

Automated checks live in `tests/` (pytest, in-memory files only — no sample resumes needed):

```powershell
cd E:\NLP_Project
.\venv\Scripts\Activate.ps1
python -m pytest -q
```

49 tests cover file-type validation, PDF/DOCX extraction (including blank, corrupt and empty files), TF-IDF ranking and descending sort order, matched/missing keywords, single and batch uploads, graceful handling of empty JDs / invalid types / oversize files / the 20-file cap, HTML-escaping of filenames and resume text, unicode, and the reset workflow.

## Files created

`app.py`, `nlp_processor.py`, `resume_parser.py`, `requirements.txt`,
`README.md`, `.gitignore`, `templates/index.html`, `templates/results.html`,
`static/css/style.css`.

No remaining manual steps except installing dependencies (commands above).

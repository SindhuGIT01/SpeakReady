# SpeakReady — Project Notes

**Live demo:** https://speakready-sindhu.streamlit.app
**GitHub repo:** https://github.com/SindhuGIT01/SpeakReady

> This file explains the project task by task, in simple language, based on
> the actual code in this repo (not a plan of what *could* be built). Where
> I wasn't 100% sure of a detail, I've written `[NOTE: ...]` so you can
> double-check it instead of me guessing.

---

## 1. Overview

**What it is:** SpeakReady is a web app that runs a mock job interview with
you **out loud**. You pick a role (or type your own), it asks you questions
by voice, you answer by speaking into your microphone, and it:

- turns your speech into text,
- scores how *fluently* you spoke (pace, pauses, filler words) using a
  machine learning model trained on real voice data,
- scores *what you said* using an LLM (content, grammar, a rewritten
  answer),
- decides on the fly whether to ask a follow-up question,
- and at the end gives you a 7-day practice plan and a downloadable PDF
  report.

**Who it's for:** freshers and early-career candidates who want to practice
interview answers *out loud*, the way a real interview actually happens,
instead of just reading sample answers silently.

**Why it's interesting as a portfolio project:** it isn't just "call an LLM
API." It combines a trained ML model (not just a prompt), RAG (retrieval
over a question bank and over resumes), real speech processing, a stateful
multi-turn agent, a full web app, a test suite, and a live deployment with
real CI/CD — the kind of full pipeline a working product actually needs.

### Simple text diagram of the flow

```
 Resume PDF (optional)              Question bank (159 questions, JSON)
        │                                        │
        ▼                                        ▼
 Resume RAG (Chroma +               Question bank RAG (Chroma +
 MiniLM embeddings)                 MiniLM embeddings)
        │                                        │
        └──────────────────┬─────────────────────┘
                            ▼
                     Question Plan
                 (mix of bank + resume
                  questions, or LLM-
                  generated for a
                  custom role)
                            │
                            ▼
                   Interview Agent
            (asks question out loud via
                  TTS — gTTS)
                            │
                            ▼
                  Candidate speaks answer
                            │
              ┌─────────────┴─────────────┐
              ▼                            ▼
     Speech-to-Text                 Raw audio waveform
   (Groq Whisper API)                      │
              │                            ▼
              │                  Voice Confidence Meter
              │                  (pitch/volume/energy,
              │                   librosa)
              ▼
     Feature Extraction
  (WPM, pauses, fillers,
   vocabulary variety...)
              │
     ┌────────┴────────┐
     ▼                 ▼
ML Fluency Model   LLM Feedback Engine
(Random Forest,    (Groq LLM: content
 trained offline)   score, grammar,
     │              rewritten answer)
     └────────┬────────┘
              ▼
       Combined Score
              │
              ▼
     Interview Agent decides:
     ask a follow-up, or move on?
              │
              ▼
   SQLite (sessions + answers)
              │
      ┌───────┴───────┐
      ▼               ▼
  Report page     Progress page
 (PDF download,   (trends across
  per-question     all past
  chart)           sessions)
```

Optional add-ons that plug into the same flow without changing it:
**webcam snapshots** → body language heuristics (eye contact, posture), and
an **answer replay timeline** that marks filler words/pauses/fast speech
visually under the audio player.

---

## 2. Task-by-task

### Task 1 — Project setup: structure, config, Groq LLM client

**a) What we built**
The skeleton of the whole project: the folder layout (`src/`, `app_pages/`,
`data/`, `models/`, `tests/`), a central settings file, and the function
that creates the LLM client.

**b) Why we built it**
Every later task needs somewhere to read settings from (model names, file
paths, API keys) and a single way to get an LLM client — without this,
every module would hardcode its own values and the project would become
inconsistent fast.

**c) How it works, step by step**
1. `src/config.py` loads a `.env` file (via `python-dotenv`) and exposes
   every setting as a module-level constant — paths, model names, chunk
   sizes, scoring thresholds, everything.
2. `config.get_groq_api_key()` is the *only* sanctioned way other code
   reads the Groq key — it raises a clear `ConfigError` if it's missing,
   instead of a confusing crash somewhere deep in a library.
3. `src/llm.py`'s `get_llm()` builds a `ChatGroq` client using those
   settings (model name, temperature, retries).

**d) Key files and functions**
- [src/config.py](../src/config.py) — every setting lives here.
- [src/llm.py](../src/llm.py) — `get_llm()`.

**e) Libraries and tools used, and why**
- **python-dotenv** — loads `.env` without hand-rolling file parsing.
- **langchain-groq (`ChatGroq`)** — a standard LangChain chat-model
  interface over Groq's API, so later code (feedback, follow-ups, resume
  question generation) can all use the same `.with_structured_output(...)`
  pattern instead of hand-parsing raw API responses.

**f) Problems we hit and how we fixed them**
Early on, the Groq Python SDK's version pin needed fixing (commit
`9bb0498 Fix groq version pin and ignore placeholder API key`) — the
`langchain-groq` package requires `groq<1.0`, and `config.py` was updated so
a leftover placeholder value in `.env.example`
(`your_groq_api_key_here`) is treated as "not set" instead of being used as
a literal (broken) API key.

---

### Task 2 — Interview question bank with RAG

**a) What we built**
A bank of 159 interview questions (HR, Behavioral, Java, Python, SQL,
Machine Learning, AWS/Cloud, Web Development) stored as JSON, embedded into
a vector database so the app can *semantically* retrieve the most relevant
ones for a given role.

**b) Why we built it**
A fixed interview can't just be "show the same 10 questions every time."
Picking by *meaning* (e.g. a role like "Java Developer" should surface Java
questions, not SQL ones) needs semantic search, not simple keyword
matching or a lookup table.

**c) How it works, step by step**
1. Each question record has an `id`, `question` text, `category`,
   `difficulty`, and `what_good_answer_covers` (key points a strong answer
   should hit).
2. `ingest_questions()` embeds every question's `"category: question text"`
   string with a HuggingFace sentence-transformer model and stores it in a
   persistent **Chroma** vector collection, skipping any `id` already
   ingested (so re-running it is safe).
3. `get_questions(role, difficulty, n)` runs a similarity search against
   the role name, split into a technical bucket and an HR/Behavioral
   bucket, so every generated set always mixes in 1–2 HR/Behavioral
   questions alongside the role-relevant technical ones.

**d) Key files and functions**
- [src/question_bank.py](../src/question_bank.py) — `ingest_questions()`,
  `get_questions()`.
- [data/questions/question_bank.json](../data/questions/question_bank.json)
  — the 159 questions.

**e) Libraries and tools used, and why**
- **LangChain + Chroma** — Chroma is a free, local, file-based vector
  store (no separate server to run), which fits a portfolio project that
  needs to run "for free, no paid APIs" per the project's own rule.
- **sentence-transformers/all-MiniLM-L6-v2** (via `langchain-huggingface`)
  — a small, fast, free embedding model that runs on CPU, good enough for
  this scale of question bank.

**f) Problems we hit and how we fixed them**
`[NOTE: I didn't find a dedicated bug-fix commit for this task specifically
— it appears to have gone in fairly cleanly in commit ac0d7dc. If you
remember a specific issue here, add it.]`

---

### Task 3 — Resume parsing, resume RAG, and personalized questions

**a) What we built**
Upload a resume PDF → extract its text → pull out a structured profile
(skills, projects, education, certifications) with the LLM → generate
interview questions grounded in *that specific resume* (e.g. "Tell me about
your CustomerIQ project").

**b) Why we built it**
Generic questions don't test what a candidate actually did. Asking about
their real projects is both more realistic and harder to fake — it's what
a human interviewer who read your resume would do.

**c) How it works, step by step**
1. `load_resume()` extracts raw text from the PDF with `pypdf`; if there's
   no extractable text (e.g. a scanned image PDF), it raises a clear
   `ResumeParseError` instead of silently producing nothing.
2. `extract_profile()` sends that text to the LLM with
   `with_structured_output(ResumeProfile)` — a Pydantic model with `name`,
   `skills`, `projects`, `education`, `certifications` — and the system
   prompt explicitly says "only include what's actually in the text, never
   invent."
3. `generate_resume_questions()` asks the LLM for N questions grounded in
   that profile, where every question must cite exactly which resume
   section it came from (e.g. `"Projects: CustomerIQ"`). Any question
   citing a section that isn't actually in the profile (a hallucinated
   project) is **dropped** before it ever reaches the candidate.
4. The resume's text is also chunked and embedded into a **per-session**
   Chroma collection (`index_resume()`), so different users' resumes never
   mix together.

**d) Key files and functions**
- [src/resume.py](../src/resume.py) — `load_resume()`, `extract_profile()`,
  `generate_resume_questions()`, `index_resume()`.

**e) Libraries and tools used, and why**
- **pypdf** — free, pure-Python PDF text extraction, no external binary.
- **Pydantic + LangChain structured output** — forces the LLM's response
  into a typed schema instead of trusting it to format free text
  correctly.
- **RecursiveCharacterTextSplitter** (LangChain) — standard chunking for
  embedding longer text.

**f) Problems we hit and how we fixed them**
The main risk here was **hallucination** — an LLM inventing a project or
skill that isn't on the resume. The fix wasn't a bug fix after the fact,
it's built into the design: the `resume_section` field is checked against
the real profile after generation, and anything that doesn't match a real
section is filtered out before the candidate ever sees it.

---

### Task 4 — Speech-to-text (Groq Whisper) and text-to-speech (gTTS)

**a) What we built**
Two things that work together: turning the candidate's *spoken* answer
into text with word-level timestamps (speech-to-text), and turning the
*interview questions* into spoken audio (text-to-speech), so the interview
genuinely happens by voice in both directions.

**b) Why we built it**
The whole point of SpeakReady is practicing speaking, not typing. Without
real speech-to-text, there's no fluency to measure; without text-to-speech,
it isn't actually an interview "by voice."

**c) How it works, step by step**
1. `transcribe()` validates the uploaded audio first (format, file size,
   minimum duration) before ever calling the API, so bad uploads fail fast
   with a clear message.
2. It calls Groq's hosted Whisper API
   (`whisper-large-v3-turbo`) with `timestamp_granularities=["word",
   "segment"]`, which returns not just the text but the start/end time of
   every word, plus a confidence value (`avg_logprob`) per *segment*
   (chunk of speech) — never per individual word.
3. If the API returns a rate-limit error, `_transcribe_with_retry()` retries
   with exponential backoff instead of failing the whole interview on one
   transient error.
4. `speak()` (text-to-speech) uses `gTTS` to synthesize each question as an
   MP3, and **caches** the result on disk keyed by a hash of
   `(language, text)` — since the same questions repeat across users and
   sessions, this avoids re-generating (and re-waiting for) audio for text
   already spoken before.

**d) Key files and functions**
- [src/speech.py](../src/speech.py) — `transcribe()`.
- [src/tts.py](../src/tts.py) — `speak()`.

**e) Libraries and tools used, and why**
- **Groq Whisper API** — free-tier hosted Whisper, so transcription
  doesn't need a local GPU or a paid API.
- **gTTS (Google Text-to-Speech)** — simple, free, no API key needed.

**f) Problems we hit and how we fixed them**
Whisper's hosted API only exposes confidence **per segment**, never per
word — this isn't a bug to "fix," but it's an honest limitation that later
shaped Task 13's "words to double check" feature (see below): rather than
pretending to have per-word pronunciation scoring it doesn't have, the app
flags words inside low-confidence *segments* and says so explicitly.

---

### Task 5 — Speech fluency feature extraction

**a) What we built**
A function that turns a transcript (text + word timestamps) into a fixed
set of ten numeric features that describe *how* someone spoke — not what
they said.

**b) Why we built it**
The ML fluency model (Task 6) needs numbers to learn from, and those same
numbers need to be computed identically both when training the model and
when scoring a real answer live in the app — otherwise the model would be
scoring on different inputs than it was trained on.

**c) How it works, step by step**
1. `extract_features()` is the single function both the training notebook
   and the live app call, so the two can never drift apart.
2. It computes: speaking pace (words per minute), pause count and average
   pause length (gaps between words above a threshold), a *long*-pause
   count (a stricter threshold), filler word count and rate (per 100
   words), repeated-word count, vocabulary variety (type-token ratio —
   unique words ÷ total words), and average sentence length.
3. Filler-word detection (`um`, `uh`, `you know`, `basically`, etc.) has
   one deliberate special case: the word "like" is only counted as filler
   when it's *not* preceded by a word that suggests it's being used as an
   ordinary verb (e.g. "I **like** pizza" is not filler; a standalone
   "like," is).
4. `explain_features()` turns the raw numbers into plain-language coaching
   tips (e.g. "You spoke at 185 WPM — a bit fast").

**d) Key files and functions**
- [src/features.py](../src/features.py) — `extract_features()`,
  `find_filler_words()` (reused by Task 12's timeline), `pause_gaps()`
  (reused by Task 12), `explain_features()`.

**e) Libraries and tools used, and why**
- Pure Python + `re` (regex) — no heavy library needed for this; it's
  arithmetic and string matching over the word list Whisper already
  returned.

**f) Problems we hit and how we fixed them**
The "like" filler-word edge case above is the main subtlety here — a naive
filler-word counter would flag every "like," which is wrong often enough to
be misleading. The fix was a small allow-list of words that typically
precede "like" used as a real verb.

---

### Task 6 — ML fluency classification model

**a) What we built**
A real, trained machine-learning classifier (not a hand-written rule or an
LLM prompt) that predicts whether a spoken answer sounds like **Beginner**,
**Intermediate**, or **Fluent** speech, plus a continuous 0–100 score.

**b) Why we built it**
LLMs are good at judging *content*, but fluency (pace, pauses, filler
words) is a numeric, statistical pattern — exactly the kind of problem a
small classical ML model is well suited for, and it's also what makes this
project more than "just an LLM wrapper."

**c) How it works, step by step** (from `notebooks/fluency_model.ipynb`)
1. **Dataset:** [speechocean762](https://huggingface.co/datasets/mispeech/speechocean762),
   5,000 utterances (2,500 train / 2,500 test) of non-native speakers
   reading sentences aloud, each with an expert 0–10 fluency score.
2. **Labeling:** the 0–10 score is bucketed into 3 classes: Beginner (0–5),
   Intermediate (6–7), Fluent (8–10).
3. **Transcription:** every utterance is transcribed locally with
   `faster-whisper`'s `small` model (not the hosted Groq API — doing 5,000
   calls would burn through free-tier rate limits), with word timestamps.
4. **Features:** the exact same `extract_features()` from Task 5 is run on
   each transcript, so training features and live features can never
   diverge.
5. **Baseline:** a majority-class predictor is scored first, so later
   models have something real to beat — with classes this imbalanced
   (66% "Fluent"), accuracy alone can be misleading, so **macro F1** is
   tracked throughout.
6. **Training:** three candidates (Logistic Regression, Random Forest,
   XGBoost) are each tuned with 5-fold **stratified cross-validation on the
   train split only**, optimizing macro F1. XGBoost needs integer-encoded
   labels, so it's wrapped in a small `LabelDecodingClassifier` adapter
   (defined in `src/scorer.py` so it can be unpickled later) that exposes
   the same string-label interface as the other models.
7. **Evaluation:** every tuned model is evaluated exactly once on the
   held-out test split. **Random Forest won** by test macro F1.
8. **Saving:** the winning model and its metrics are saved to
   `models/fluency_model.joblib` and `models/metrics.json`.
9. **Inference:** `src/scorer.py`'s `predict_fluency()` loads that saved
   model, predicts class probabilities for a new answer's features, and
   turns those probabilities into a continuous 0–100 score by weighting
   each class's "anchor" score (Beginner≈25, Intermediate≈65, Fluent≈90)
   by its predicted probability — so the score isn't just "which bucket,"
   it reflects *how confidently*.

**Results** (from `models/metrics.json`):

| Model | Test accuracy | Test macro F1 |
|---|---|---|
| Baseline (majority class) | 68.0% | 0.270 |
| Logistic Regression | 70.0% | 0.598 |
| **Random Forest (winner)** | **72.7%** | **0.633** |
| XGBoost | 72.4% | 0.619 |

Per-class test F1 for the winner: Beginner 0.570, Intermediate 0.488,
Fluent 0.841 — strongest on clearly fluent speech, weakest at the
Beginner/Intermediate boundary (which is also where human raters tend to
disagree most).

**d) Key files and functions**
- [notebooks/fluency_model.ipynb](../notebooks/fluency_model.ipynb) —
  the full training pipeline.
- [src/scorer.py](../src/scorer.py) — `predict_fluency()`,
  `LabelDecodingClassifier`.
- [models/fluency_model.joblib](../models/fluency_model.joblib),
  [models/metrics.json](../models/metrics.json) — the saved model and its
  numbers.

**e) Libraries and tools used, and why**
- **scikit-learn** — Logistic Regression, Random Forest, cross-validation
  utilities, metrics.
- **XGBoost** — a stronger gradient-boosted tree baseline to compare
  against.
- **faster-whisper** — local, free transcription for generating training
  features at scale without hitting API rate limits.
- **joblib** — standard way to persist a scikit-learn-compatible model to
  disk.

**f) Problems we hit and how we fixed them**
- **Class imbalance** (66% "Fluent"): fixed by tracking macro F1 (not just
  accuracy) throughout, so a model can't look good purely by always
  guessing the majority class.
- **XGBoost needs integer labels, not strings:** fixed with the
  `LabelDecodingClassifier` wrapper so every model exposes the same
  interface regardless of its underlying label format.
- **Read-aloud vs. spontaneous speech:** the dataset is people *reading* a
  fixed sentence, not answering an interview question off the cuff —
  the README documents this honestly as a limitation rather than treating
  the model as more validated than it is.

---

### Task 7 — LLM feedback engine

**a) What we built**
A separate LLM pass that grades the **content** of an answer — not pace or
filler words (Task 5/6 already cover that) — producing a content score, a
list of key points covered/missed, real grammar corrections, a rewritten
"better" version of the answer, and one concrete tip.

**b) Why we built it**
Fluency alone doesn't tell you if the answer was actually *good*. A
candidate can speak smoothly and still miss the point of the question, or
speak haltingly but say something substantive — content and delivery are
different axes and need different judges (a trained model for delivery, an
LLM for content judgment).

**c) How it works, step by step**
1. `generate_feedback()` sends the question, the transcript, (optionally)
   the question bank's "what a good answer covers" list, and (optionally)
   resume context, to the LLM with `with_structured_output(...)`.
2. The system prompt is explicit that the LLM should score *content only*
   — fluency/delivery is explicitly out of scope for this call, since a
   separate system already measures it.
3. `key_points_covered`/`missed` are judged against the question bank's own
   list when one exists, so grading isn't just "whatever the LLM feels
   like."
4. `grammar_corrections` must be **real** mistakes actually present in the
   transcript — the prompt explicitly says to return an empty list rather
   than invent a correction to "fill" it.
5. `improved_answer` must stay honest: only using facts the candidate
   actually said, or that are in their resume context — never inventing
   experience.
6. The filler-word summary in the final `Feedback` object is **not** asked
   of the LLM at all — it's computed deterministically from Task 5's
   features and merged in afterward, so delivery feedback is never left to
   LLM guesswork.
7. `compute_overall_score()` combines the LLM's 0–10 content score and the
   ML model's 0–100 fluency score into one 0–100 number, weighted
   `CONTENT_SCORE_WEIGHT` (0.6) / `FLUENCY_SCORE_WEIGHT` (0.4) from config.
8. If the LLM's structured output fails to parse/validate, `_invoke_with_retry()`
   retries a configurable number of times before raising a clear
   `FeedbackError`, instead of crashing the interview on one bad response.

**d) Key files and functions**
- [src/feedback.py](../src/feedback.py) — `generate_feedback()`,
  `compute_overall_score()`.
- [src/prompts.py](../src/prompts.py) — `FEEDBACK_SYSTEM_PROMPT_V1`,
  `build_feedback_user_prompt()`.

**e) Libraries and tools used, and why**
- **langchain-groq + Pydantic structured output** — same pattern as Task 3,
  for a typed, validated response instead of parsing free text.
- **Groq (`openai/gpt-oss-120b`)** — the LLM model used for all feedback
  calls, set once in `src/config.py`.

**f) Problems we hit and how we fixed them**
LLM structured-output calls occasionally return invalid or unparseable
JSON (a known flaky behavior of LLM tool-calling). The fix was
`_invoke_with_retry()`: catch `ValidationError`/`OutputParserException`/
`ValueError`, retry up to `FEEDBACK_LLM_MAX_RETRIES` times, and only raise
after every attempt fails. This same pattern was reused later (see the
"Begin Interview error" story below) when the same flakiness showed up in
a newer code path that hadn't been given the same protection yet.

---

### Task 8 — Interview agent: follow-ups, text-to-speech, and session state

**a) What we built**
The stateful "brain" of one mock interview: `InterviewSession` — it builds
a question plan, asks each question (with TTS audio), scores each answer,
decides in real time whether a follow-up is worth asking, and tracks
everything needed for the end-of-session summary.

**b) Why we built it**
A real interview isn't a fixed script — a good interviewer reacts to what
you actually said. This task ties together every earlier piece (question
bank, resume RAG, speech-to-text, features, fluency model, feedback engine,
text-to-speech) into one coherent, testable flow, and adds the one thing
none of the earlier pieces could do alone: deciding to dig deeper.

**c) How it works, step by step**
1. `InterviewSession.start()` builds a `question_plan`: by default, roughly
   half question-bank questions and half resume-grounded questions
   (interleaved so they alternate), or — for a custom role not in the bank
   — a fully LLM-generated set (see the bonus feature below).
2. `next_question()` pops the next item off the plan's queue, synthesizes
   its audio with `src/tts.py`, and returns it with a running question
   number.
3. `submit_answer()` / `submit_transcript()` / `submit_text_answer()` run
   the shared `_process_transcript()` pipeline: extract features (Task 5),
   predict fluency (Task 6), generate feedback (Task 7), optionally analyze
   body language (Task 11) and voice confidence (Task 13), save everything
   to SQLite, and decide on a follow-up.
4. `_maybe_queue_follow_up()` asks the LLM a *separate*, focused question —
   "would a follow-up genuinely help here?" — via its own structured
   output call (`FollowUpDecision`). If yes, the follow-up question is
   pushed to the **front** of the queue so it's asked next, and a per-main-
   question follow-up limit (`MAX_FOLLOW_UPS_PER_QUESTION = 2`) stops the
   interview spiraling into endless follow-ups.
5. `end()` averages the session's scores, finds the strongest/weakest
   scoring area, and asks the LLM for a personalized 7-day practice plan
   grounded in the actual tips given during the session — not generic
   advice.

**d) Key files and functions**
- [src/interview_agent.py](../src/interview_agent.py) — `InterviewSession`
  (the whole class), `QuestionItem`, `AnswerRecord`, `SessionSummary`.
- [src/storage.py](../src/storage.py) — the SQLite schema
  (`sessions`/`answers` tables) this class writes to.

**e) Libraries and tools used, and why**
- **Python `dataclasses` + `collections.deque`** — the question queue and
  every record type are plain dataclasses; a `deque` lets a follow-up be
  pushed to the front cheaply.
- **sqlite3** (standard library) — zero-setup embedded database, fine for
  a single-user-at-a-time portfolio demo.

**f) Problems we hit and how we fixed them**
See **"The Begin Interview error"** story near the end of this file — the
follow-up/question-generation path is exactly where a missing retry
(the same class of bug Task 7 had already solved) resurfaced in a new
feature later.

---

### Task 9 — Streamlit app and progress tracker

**a) What we built**
The actual multi-page web app a candidate uses: Home, Setup, Interview,
Report, Progress — wired together with Streamlit's page navigation and
shared session state, plus a Progress page that reads SQLite history and
charts trends across *every* past session, not just the current one.

**b) Why we built it**
Everything up to this point was a library of working pieces; without a UI,
none of it is usable by an actual person. The Progress page specifically
answers "am I actually getting better?" — the kind of longitudinal feedback
a single mock interview can't give you.

**c) How it works, step by step**
1. [app.py](../app.py) sets page config, initializes shared session state,
   kicks off question-bank ingestion once per process (cached), and
   registers the five pages with `st.navigation`.
2. [app_pages/setup.py](../app_pages/setup.py) — pick a role (or type a
   custom one), difficulty, question count, optionally upload a resume;
   clicking "Begin Interview" builds an `InterviewSession` and switches to
   the Interview page.
3. [app_pages/interview.py](../app_pages/interview.py) — a small state
   machine (`idle` → `asking` → `reviewing` → repeat → `finished`) drives
   one question at a time: play the question's audio, record the answer
   with `st.audio_input`, show a fast local "quick signal" (filler/pause
   count, no LLM call) immediately, then the full feedback once the LLM
   responds.
4. [app_pages/report.py](../app_pages/report.py) — the session summary,
   a per-question score bar chart, the 7-day practice plan, and a PDF
   download button.
5. [app_pages/progress.py](../app_pages/progress.py) — reads every
   completed session from SQLite via `storage.list_sessions()`, and plots
   score trend, filler-word rate, and speaking pace over time with Plotly,
   plus a bar chart of average score per question category/resume area
   (color-scaled red-to-green) to show the weakest areas across the whole
   history.
6. [app_common.py](../app_common.py) holds shared, UI-only glue: cached
   resources (`get_llm_cached()`, `get_db_connection()`,
   `ensure_question_bank_ingested()`), session-state defaults, and the
   Task 12 timeline chart builder — kept separate from `src/` because it's
   Streamlit-specific, not business logic.

**d) Key files and functions**
- [app.py](../app.py), [app_common.py](../app_common.py),
  [app_pages/*.py](../app_pages).

**e) Libraries and tools used, and why**
- **Streamlit** — free, fast way to build a multi-page Python web app
  without writing frontend code by hand; `st.audio_input`/`st.camera_input`
  give browser mic/webcam access for free too.
- **Plotly / pandas** — interactive charts and the tabular data wrangling
  behind them.

**f) Problems we hit and how we fixed them**
`[NOTE: I didn't find a dedicated bug-fix commit scoped just to the app UI
itself — most app-level issues that came up later (the Begin Interview
error) are described in their own section below rather than here.]`

---

### Task 10 — Tests, README, CI, and deployment config

**a) What we built**
A 161-test pytest suite covering nearly every module, a GitHub Actions CI
workflow that runs lint + tests on every push, a detailed README, and the
initial deployment configuration.

**b) Why we built it**
A project that "works on my machine" isn't trustworthy. Automated tests
catch regressions before they reach users; CI proves the tests actually
pass in a clean environment (not just the developer's own machine); a good
README is what makes a portfolio project legible to someone who didn't
build it.

**c) How it works, step by step**
1. External APIs (Groq LLM, Groq Whisper) are **mocked** in almost every
   test (`unittest.mock.MagicMock`, fake LLMs that route
   `with_structured_output()` calls by schema), so the whole suite runs
   fast and without needing a real API key.
2. A handful of **live smoke tests** (in `test_llm.py`, `test_scorer.py`,
   `test_speech.py`) call the real Groq API, but are wrapped in
   `pytest.mark.skipif(not config.GROQ_API_KEY, ...)` so they automatically
   skip — rather than fail — when no key is configured (e.g. in CI).
3. [.github/workflows/ci.yml](../.github/workflows/ci.yml) runs two jobs on
   every push/PR to `main`: `ruff check .` (lint) and the full pytest
   suite, both on `ubuntu-latest`.
4. The README documents setup, the architecture diagram, model results,
   deployment steps, and — importantly — an honest **Limitations** section
   (read-aloud vs. spontaneous speech, transcription mismatch,
   English-only, heuristic-not-trained-model caveats for body
   language/voice confidence, the near-real-time nature of the filler
   alert, and more).

**d) Key files and functions**
- [tests/](../tests) — one `test_*.py` file per `src/` module, 161 tests
  total.
- [.github/workflows/ci.yml](../.github/workflows/ci.yml) — the CI
  pipeline.
- [README.md](../README.md) — setup, architecture, results, deployment,
  limitations.

**e) Libraries and tools used, and why**
- **pytest** — the standard Python test runner; `pytest.mark.skipif` for
  conditional live tests.
- **ruff** — a fast all-in-one Python linter, cheap to run on every push.
- **GitHub Actions** — free CI for public repos, directly integrated with
  GitHub.

**f) Problems we hit and how we fixed them**
This is where the CI story (detailed in its own section below) actually
started: the workflow file existed and *looked* correct from the start,
but — as later investigation found — it had been silently failing on
`pip install -r requirements.txt` since the very first commit that added
it, because of a Python-version mismatch with `numpy`. See **"CI
fixes"** below for the full story.

---

### Task 11 — Webcam body language coaching (eye contact, posture)

**a) What we built**
An **optional** toggle that takes a snapshot or two of the candidate while
they answer and scores eye contact and posture using MediaPipe's face and
pose landmark detection — pure geometry heuristics, not a trained model.

**b) Why we built it**
Interviews are about presence, not just words — but this had to be
strictly optional and never required, since not everyone has or wants to
use a webcam.

**c) How it works, step by step**
1. The Interview page shows a `st.toggle` for "Track eye contact & posture
   (optional)"; if it's on, `st.camera_input` lets the candidate capture
   snapshots while answering (Streamlit has no continuous webcam video
   recorder, so this works from discrete still frames, not a video
   stream).
2. Each snapshot is decoded with OpenCV (`cv2.imdecode`) into a frame and
   collected in session state.
3. `analyze_frames()` runs every frame through two MediaPipe **Tasks**
   models: Face Landmarker (for head pose — yaw/pitch angle, to judge
   "looking at the camera") and Pose Landmarker (for shoulder
   level/tilt and nose-to-shoulder distance, to judge posture).
4. Eye contact is the fraction of face-detected frames whose head pose was
   within a yaw/pitch threshold of facing forward. "Looking away" is
   counted once per *run* of consecutive away frames lasting over
   `LOOKING_AWAY_MIN_SECONDS`, not once per away frame.
5. Posture is a 0–100 average of two sub-scores per frame: how level the
   shoulders are (a tilted line suggests leaning) and how upright the head
   sits above the shoulders (a short nose-to-shoulder distance suggests
   hunching toward the camera).
6. The two MediaPipe model bundle files (`.task` files) are downloaded
   once on first use from Google's model CDN and cached under `models/` —
   they're deliberately **not** committed to git (`models/*.task` is
   gitignored).
7. Analysis is always **best-effort**: `_analyze_body_language()` in the
   interview agent catches any failure and returns `None` rather than ever
   taking down the interview over a webcam problem.

**d) Key files and functions**
- [src/body_language.py](../src/body_language.py) — `analyze_frames()`,
  `analyze_video()`.

**e) Libraries and tools used, and why**
- **MediaPipe** (Face Landmarker + Pose Landmarker, the "Tasks" API) —
  Google's free, pretrained, on-device landmark models; no training needed
  for this heuristic approach.
- **OpenCV (`cv2`)** — decoding snapshot images into frames MediaPipe can
  consume.

**f) Problems we hit and how we fixed them**
This is where the **libGL/libEGL story** (see below) actually originates:
`mediapipe` and `opencv-contrib-python` (which mediapipe depends on)
need graphics system libraries (`libGL`, later `libEGL`/`libGLES`) that a
minimal Linux container doesn't ship by default — this didn't surface
until *deployment* (Streamlit Cloud) and *CI* (GitHub Actions ubuntu
runners), both described in their own sections below.

---

### Task 12 — Answer replay timeline (filler/pause/pace markers)

**a) What we built**
A visual timeline under the answer's audio player that marks exactly where
each filler word, long pause, and fast-speech stretch happened — plus a
near-instant filler/pause count shown the moment transcription finishes,
before the slower LLM feedback call even starts.

**b) Why we built it**
A number like "3 filler words" is much less useful than *seeing where* they
happened while listening back to the actual recording — this turns
abstract feedback into something concrete to replay and learn from.

**c) How it works, step by step**
1. `build_timeline()` reuses Task 5's exact filler-detection logic
   (`find_filler_words()`) and pause-gap logic (`pause_gaps()`), so the
   timeline and the ML fluency score never disagree about what counts as a
   filler or a long pause.
2. It also detects **fast-speech** stretches: a sliding 5-word window whose
   local words-per-minute exceeds a threshold; overlapping flagged windows
   are merged into a single marker so one fast stretch produces one
   marker, not several overlapping ones.
3. All markers are sorted by start time into one flat list and handed to
   `app_common.render_timeline_chart()`, which draws them as a Plotly
   horizontal bar chart (amber = filler, blue = long pause, red = fast
   speech), with hover text per marker.
4. The "near-real-time" filler alert: right after transcription finishes
   (before the slower content-feedback LLM call), the Interview page shows
   a quick count computed locally from the transcript (no LLM call needed
   — it's pure arithmetic over the transcript), so the candidate gets
   *some* signal almost instantly.

**d) Key files and functions**
- [src/timeline.py](../src/timeline.py) — `build_timeline()`, `Marker`.
- [app_common.py](../app_common.py) — `render_timeline_chart()`.

**e) Libraries and tools used, and why**
- **Plotly** — interactive hoverable charts, already used elsewhere
  (Progress page), so no new charting library was introduced.

**f) Problems we hit and how we fixed them**
`[NOTE: no dedicated bug-fix commit found specific to this task — it's
worth calling out in an interview that this is deliberately *not* true
live/streaming feedback, since Streamlit reruns the whole page per
interaction and can't stream microphone audio word-by-word. The README's
Limitations section explains this honestly — see the "Future
improvements" section of this file too.]`

---

### Task 13 — Voice confidence meter and pronunciation signal

**a) What we built**
Two more *optional*, honestly-labeled heuristic signals computed from the
raw audio and the transcript: a **voice confidence meter** (pitch
variability, volume steadiness, speaking energy) and a **"words to double
check"** proxy for pronunciation.

**b) Why we built it**
Both signals round out the picture of *how* something was said, beyond
what the ML fluency model (Task 6, which only ever sees transcript/timing
features, never the waveform) can measure — while being explicit that
neither is a trained, validated model, unlike the fluency score.

**c) How it works, step by step — voice confidence**
1. `analyze_voice_confidence()` loads the raw answer audio with `librosa`.
2. **Pitch variability:** using `librosa.pyin` to estimate the fundamental
   frequency (f0) per frame, it computes the standard deviation of voiced
   frames in *semitones* relative to the answer's own median pitch (so
   it's not biased by someone's natural vocal register) — too flat
   (monotone) or too erratic both score lower, since both commonly read as
   nervous.
3. **Volume steadiness:** RMS energy over coarse ~0.5s windows (with
   near-silent windows excluded so pauses aren't scored as "shaky"), scored
   by how low its coefficient of variation is.
4. **Speaking energy:** mean RMS loudness normalized against a reference
   level.
5. The three sub-scores (each 0–100) are averaged into one
   `confidence_score`, with a short plain-language summary built from
   simple score thresholds.

**How it works — pronunciation proxy**
1. Groq's hosted Whisper API exposes ASR confidence **per segment**, never
   per word (see Task 4) — true per-word pronunciation scoring would need
   a local forced-alignment/phoneme-confidence model, which this project
   doesn't run.
2. `flag_low_confidence_words()` instead finds every word that falls
   inside a segment whose `avg_logprob` is below a configured threshold,
   and flags it as "worth a second listen" — explicitly documented as a
   proxy, not a verdict, since a flagged word could be mispronounced,
   mumbled, drowned out by noise, or just an unusual word the ASR wasn't
   expecting.

**d) Key files and functions**
- [src/confidence.py](../src/confidence.py) — `analyze_voice_confidence()`.
- [src/pronunciation.py](../src/pronunciation.py) —
  `flag_low_confidence_words()`.

**e) Libraries and tools used, and why**
- **librosa** — the standard Python audio-analysis library; already a
  project dependency (used in model training too), so no new dependency
  was needed for raw-waveform analysis.

**f) Problems we hit and how we fixed them**
`[NOTE: no dedicated bug-fix commit found specific to this task. The
interesting design decision here — worth mentioning in an interview — is
choosing to be explicit about what each signal *isn't* (not a trained
model, not true pronunciation scoring) rather than overstating accuracy.]`

---

### Bonus feature (added after Task 13) — Custom-role interview questions

**a) What we built**
On the Setup page, an **"Other (type your own role)"** option alongside
the fixed role dropdown. Picking it lets a candidate type any role (e.g.
"Business Analyst", "DevOps Engineer") and generates a fresh set of
interview questions for that exact role live, using the LLM, instead of
retrieving from the static 159-question bank (which only covers 8 fixed
categories).

**b) Why we built it**
The original question bank can't cover every possible role. Rather than
limiting candidates to a fixed list, this lets SpeakReady handle *any*
role a candidate wants to practice for.

**c) How it works, step by step**
1. `generate_custom_role_questions()` asks the LLM (via structured output,
   same pattern as Task 3/7) for N questions for the typed role, mixing in
   1–2 HR/Behavioral questions alongside role-specific ones, matching the
   requested difficulty.
2. The result is returned in **exactly the same record shape** as the
   static question bank (`id`, `question`, `category`, `difficulty`,
   `what_good_answer_covers`) — so `InterviewSession` and everything
   downstream (scoring, feedback, follow-ups) treats a custom-role question
   identically to a bank question. `QuestionItem.source` is simply tagged
   `"custom_role"` instead of `"bank"`, for traceability only.
3. `InterviewSession.start(..., use_custom_role=True)` branches to call
   this generator instead of the bank's vector search, purely based on
   that one boolean flag — no other code needed to change.

**d) Key files and functions**
- [src/question_bank.py](../src/question_bank.py) —
  `generate_custom_role_questions()`.
- [src/interview_agent.py](../src/interview_agent.py) — `start(...,
  use_custom_role=...)`.
- [app_pages/setup.py](../app_pages/setup.py) — the "Other" dropdown
  option and text input.

**e) Libraries and tools used, and why**
Same stack as Task 3/7 — LangChain structured output over Pydantic models.
No new library was needed.

**f) Problems we hit and how we fixed them**
See **"The Begin Interview error"** below — this is exactly where it
happened.

---

## 3. Deployment story

**What we built:** a live, public demo, deployed from this same GitHub
repo.

**The journey:** the README originally documented deploying to
**Hugging Face Spaces** — `packages.txt` and the opencv/mediapipe
dependency cleanup were written with that platform in mind. The actual
live demo ended up on **Streamlit Community Cloud** instead, which deploys
directly from a connected GitHub repo rather than a separate git push. The
README's Deployment section was rewritten to match Streamlit Cloud's real
flow: connect the repo at [share.streamlit.io](https://share.streamlit.io),
set the Python version and `GROQ_API_KEY` secret (as TOML) in "Advanced
settings" before deploying, and watch the build log.

**Key deployment-specific fixes, both still relevant:**

1. **`opencv` package conflict.** `mediapipe` hard-depends on
   `opencv-contrib-python`; the project's own `requirements.txt` also
   separately pinned `opencv-python`. Installing *both* corrupts the
   shared `cv2` Python module (symptoms like `cv2.cvtColor` or
   `cv2.VideoCapture` going missing) — reproduced locally while preparing
   this deployment. Fix: drop the project's own opencv pin and let
   mediapipe's own dependency provide `cv2`.
2. **`packages.txt` system libraries.** mediapipe's compiled core links
   against `libGL`/`libEGL`/`libGLES` even for plain CPU inference, and a
   minimal container doesn't ship them by default. `packages.txt` at the
   repo root (read by Streamlit Cloud's apt installer) now lists `libgl1`,
   `libglib2.0-0`, `libegl1`, `libgles2` — **one package name per line,
   no `#` comments**, because Streamlit Cloud's apt installer reads a
   comment's words as package names and fails the build (this was hit and
   fixed directly: an earlier version of `packages.txt` had explanatory
   comments and broke the build with errors like
   `E: Unable to locate package Hugging`).
3. **`numpy` needs Python ≥3.12.** `requirements.txt` pins
   `numpy~=2.5`, and numpy 2.5 dropped support for Python 3.11 (confirmed
   directly against PyPI's package metadata). Streamlit Cloud's Python
   version has to be picked explicitly in "Advanced settings" before
   deploying — it isn't read from a `runtime.txt`/`.python-version` file,
   and changing it for an *existing* app requires deleting and redeploying.

---

## 4. CI fixes (GitHub Actions)

The CI workflow (`.github/workflows/ci.yml`) had been **failing on every
single run since it was first added** — this wasn't something a later
change broke; it was never actually green. Fixing it took three rounds,
each round's fix revealing the next problem underneath:

1. **Round 1 — wrong Python version.** Same root cause as the deployment
   story above: CI pinned Python 3.11, but `numpy~=2.5` needs ≥3.12. Fixed
   by bumping `python-version: "3.12"` in both the lint and test jobs.
2. **Round 2 — missing `libGL`.** That fix let `pip install -r
   requirements.txt` succeed in CI for the *first time ever*, which
   immediately exposed a second, previously-hidden problem: `src/
   body_language.py` imports `cv2`/`mediapipe` at the top of the file, and
   GitHub's bare `ubuntu-latest` runner doesn't ship `libGL.so.1`.
   `test_body_language.py` imports `cv2` at module level too, so this broke
   test **collection**, not just individual test runs. Fixed by adding an
   `apt-get install -y libgl1 libglib2.0-0` step to the CI workflow,
   mirroring `packages.txt`.
3. **Round 3 — still missing `libEGL`.** That got further, but 9 tests in
   `test_body_language.py` still failed, now with
   `OSError: libEGL.so.1: cannot open shared object file`. GitHub's Actions
   log viewer and job summaries both require signing in for this repo, so
   the real error text had to be fetched by asking a human (who had
   browser access) to paste it from the Actions tab — a temporary
   diagnostic step that dumped pytest's output to the run's job summary was
   added and then removed once the real cause was known. Fixed by also
   installing `libegl1 libgles2` in the same apt-get step (and adding them
   to `packages.txt` too, for the same reason on Streamlit Cloud).

After all three rounds, both the lint and test jobs pass cleanly,
including the `test_body_language.py` suite that had genuinely never run
successfully in CI before.

---

## 5. The "Begin Interview" error (a debugging story)

**What happened:** on the live deployed app, picking "Other" → typing
"HTML" as a custom role → clicking "Begin Interview" sometimes showed
*"Something went wrong while preparing your interview. Please try
again."* — a generic message with no way to tell what actually failed.

**Step 1 — make the failure visible.** The `except Exception:` block in
`app_pages/setup.py` was swallowing every exception into one generic
message with nothing logged anywhere. The first fix added
`logger.exception(...)` so the real traceback would reach the app's server
logs — the error message shown to the user still stays friendly and
generic (never a raw traceback — that's a UX and security choice, not just
a style one).

**Step 2 — find and fix a real gap.** Reproducing it locally (the same
role, the same flow, repeatedly) never failed — but reading the code
closely revealed a real, demonstrable gap: `generate_custom_role_questions()`
(the bonus feature above) had **no retry** around its structured-output LLM
call, unlike `src/feedback.py`'s identical call, which already has one
(`_invoke_with_retry`, added back in Task 7 specifically because Groq's
structured output occasionally returns invalid/unparseable JSON). A single
bad response generating six questions at once — more surface area than
feedback's single object — is a very plausible way to hit exactly this
error with nothing logged. Fix: add the same retry pattern
(`CUSTOM_ROLE_LLM_MAX_RETRIES`, a new config value) and a dedicated
`QuestionGenerationError` raised only once retries are exhausted.

**Step 3 — temporary debugging, then clean up.** To confirm what was
actually happening live (and because a first redeploy still failed after
the retry fix), a *temporary* `st.exception(exc)` was added to show the raw
error directly on the page — useful for debugging, but never something to
ship, since showing raw tracebacks to end users is both a bad user
experience and a potential information leak. Once the real cause was
understood, that line was removed and replaced with the clean, friendly
message plus server-side logging described in Step 1.

**Lesson:** the common thread across this bug, the CI libGL/libEGL story,
and the deployment opencv conflict is the same: **a problem that only
shows up in a different environment than the one it was built in** —
whether that's "a different Python version," "a container without a
graphics library," or "an LLM response that's rare enough not to show up
in a handful of manual local tests." Logging the real exception, and
adding the same defensive pattern (retry) that an earlier, similar feature
already had, turned out to be the fix in more than one of these stories.

---

## 6. Glossary

- **RAG (Retrieval-Augmented Generation):** instead of an LLM making
  something up from scratch, first *retrieve* relevant real content (here,
  question-bank questions or resume text), then hand that to the LLM so
  its answer is grounded in real data.
- **Embeddings:** a way to turn text into a list of numbers (a vector)
  that captures its *meaning*, so similar meanings end up as similar
  vectors.
- **Vector store:** a database built to store embeddings and quickly find
  the ones most similar to a new query vector (here, Chroma).
- **Semantic search:** searching by *meaning* (via embeddings) instead of
  exact keyword matching.
- **STT (Speech-to-Text):** converting spoken audio into written text
  (here, Groq's hosted Whisper API).
- **TTS (Text-to-Speech):** converting written text into spoken audio
  (here, gTTS).
- **Fluency model:** the trained Random Forest classifier (Task 6) that
  predicts Beginner/Intermediate/Fluent from speech features.
- **Filler words:** words like "um," "uh," "like," "you know" that don't
  add meaning and often signal hesitation.
- **WPM (Words Per Minute):** a measure of speaking pace.
- **Structured output:** forcing an LLM's response into a predefined
  schema (here, a Pydantic model) instead of free-form text, so the
  response can be reliably parsed and validated.
- **Pydantic model:** a Python class that defines the exact shape (fields
  and types) data must have, and validates it automatically.
- **Hallucination:** when an LLM states something confidently that isn't
  actually true/present in its source data (e.g. inventing a project not
  on a resume) — this project actively guards against it in several
  places (Task 3, Task 7).
- **Macro F1:** a metric that averages performance equally across classes,
  so a model can't look good purely by nailing the majority class while
  ignoring minority ones.
- **Cross-validation:** repeatedly splitting the training data into
  different train/validation folds to tune a model without ever touching
  the real held-out test set.
- **CI (Continuous Integration):** automatically running checks (tests,
  lint) on every code push, so problems are caught before they reach
  users.
- **Heuristic:** a rule-of-thumb calculation (e.g. body language, voice
  confidence) based on geometry/signal thresholds — not a model trained
  and validated on labeled data, unlike the fluency model.
- **ASR (Automatic Speech Recognition):** another name for speech-to-text.
- **`avg_logprob`:** the average log-probability Whisper assigns a chunk
  of transcribed speech — closer to 0 means more confident.

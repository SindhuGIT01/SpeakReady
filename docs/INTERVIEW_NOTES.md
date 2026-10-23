# SpeakReady — Interview Notes

**Live demo:** https://speakready-sindhu.streamlit.app
**GitHub repo:** https://github.com/SindhuGIT01/SpeakReady

> Companion to [PROJECT_NOTES.md](PROJECT_NOTES.md). That file explains
> *what* was built, task by task. This file is about *talking* about it —
> pitches, likely questions, honest answers, and a cheat sheet to revise
> from right before an interview.

---

## 1. Pitches

### 30-second pitch

"SpeakReady is an AI voice interview coach I built end-to-end — you pick a
role, and it interviews you out loud. It transcribes your spoken answers
with Groq's hosted Whisper, scores your fluency with a Random Forest model
I trained myself on 5,000 labeled utterances, and grades the content of
your answer with an LLM — asking follow-up questions on the fly, just like
a real interviewer. It's not just an LLM wrapper: there's a real trained ML
model, RAG over a question bank and resumes, a 161-test suite, and it's
deployed live with CI/CD on GitHub Actions."

### 2-minute pitch

"SpeakReady is a free AI voice interview coach I built from scratch. The
idea came from noticing that most interview prep is either reading sample
answers silently or paying for a human mock interviewer — there wasn't a
free way to practice actually *speaking* answers out loud and getting real
feedback on both what you said and how you said it.

Here's the flow: you upload your resume and pick a role — or type a custom
one, which I added later — and it builds a question plan using RAG, mixing
questions from a 159-question bank with questions generated from your
actual resume, so you get both broad coverage and personalized questions
about your real projects.

Then it interviews you by voice. You speak your answer, Groq's hosted
Whisper transcribes it with word-level timestamps, and two separate systems
judge it: a Random Forest model I trained myself on the speechocean762
dataset scores your *fluency* — pace, pauses, filler words — and a
separate LLM call grades the *content* — did you actually answer the
question, grammar, a rewritten better version. Those combine into one
score. The interview agent then decides, using its own LLM call, whether a
follow-up question would genuinely help, the same way a human interviewer
reacts to what you just said rather than reading off a fixed script.

At the end you get a 7-day practice plan grounded in your actual weak
spots, a downloadable PDF report, and a progress page that tracks your
scores across every session over time.

On the engineering side: it's a Streamlit app with five pages, backed by
SQLite, with 161 pytest tests — external APIs are all mocked so the suite
runs without needing a real API key — and GitHub Actions CI. I also debugged
and fixed some real production issues after deploying it: a Python-version
mismatch and missing Linux graphics libraries that were silently failing
CI since day one, and a retry-logic gap that caused occasional errors on a
feature I'd added later. It's live right now on Streamlit Community Cloud."

---

## 2. Tech stack table (with the reason for each choice)

| Layer | Technology | Why I chose it |
|---|---|---|
| UI | Streamlit | Builds a real multi-page Python web app fast, with free built-in `st.audio_input`/`st.camera_input` for browser mic/webcam access — no separate frontend to write. |
| LLM | Groq API (`langchain-groq`, model `openai/gpt-oss-120b`) | Free-tier hosted inference with very low latency, so follow-up questions and feedback feel responsive instead of laggy. |
| Speech-to-text | Groq-hosted Whisper (`whisper-large-v3-turbo`) | Free-tier hosted Whisper with word-level timestamps — no GPU needed locally for inference. |
| Text-to-speech | gTTS | Free, no API key, simple enough for the (repeating) question bank text, with disk caching so the same question is never re-synthesized. |
| RAG | LangChain + Chroma + `sentence-transformers/all-MiniLM-L6-v2` | Chroma is a free, file-based vector store — no server to run or pay for; MiniLM is small, fast on CPU, and good enough at this scale. |
| ML | scikit-learn (Random Forest, winner) / XGBoost, features via librosa | Classical ML is the right tool for a numeric, statistical pattern like speech fluency — and it's what makes the project more than "just an LLM call." |
| Body language | MediaPipe (Face + Pose Landmarker) via OpenCV | Free, pretrained, on-device landmark detection — no training needed for a heuristic, geometry-based signal. |
| Voice confidence | librosa (pitch/RMS analysis) | Already a project dependency for model training; direct access to the raw waveform for signal-processing features an ML model never sees. |
| Storage | SQLite | Zero-setup embedded database — exactly right for a single-user-at-a-time portfolio demo, no server to provision. |
| Testing | pytest, 161 tests | Standard Python test runner; `pytest.mark.skipif` lets a few live-API smoke tests skip cleanly when no key is configured. |
| CI | GitHub Actions | Free for public repos, tightly integrated with GitHub, runs lint + tests on every push automatically. |
| Deployment | Streamlit Community Cloud | Free tier, deploys straight from the connected GitHub repo — no separate hosting to manage. |

---

## 3. 25 likely interview questions, with strong first-person answers

### Project overview

**1. Tell me about this project.**
"I built SpeakReady, an AI voice interview coach. You pick a role, it asks
you interview questions out loud, you answer by speaking, and it scores
your fluency with a Random Forest model I trained, grades your content with
an LLM, and asks follow-up questions the way a real interviewer would. It's
live, deployed, with a 161-test suite and CI."

**2. Why did you build this specific project?**
"I wanted a project that wasn't just 'call an LLM API and show the
response' — I wanted to combine a real trained ML model, retrieval,
speech processing, and a stateful multi-turn agent into one coherent
product, because that's closer to what a real engineering role actually
asks for than a single API wrapper."

**3. Walk me through what happens when a user clicks 'Begin Interview.'**
"Setup builds an `InterviewSession` and calls `start()`, which builds a
question plan — by default, roughly half from the question bank via RAG
and half generated from the candidate's resume, interleaved. If they typed
a custom role instead, the whole plan is generated live by the LLM in the
same record shape, so everything downstream treats it identically. Then it
switches to the Interview page, which pulls one question at a time with
`next_question()`, synthesizes its audio, and waits for a spoken answer."

### RAG, embeddings, and vector search

**4. What is RAG and why did you use it here?**
"Retrieval-Augmented Generation means you retrieve real, relevant content
first, then ground the LLM's output in it, instead of letting it generate
from scratch. I use it twice: retrieving the most relevant questions from
my 159-question bank for a given role, and retrieving relevant chunks of a
candidate's actual resume to generate personalized questions. It keeps
both grounded in real data instead of generic or hallucinated content."

**5. How does the question bank retrieval work?**
"Every question is embedded as `'{category}: {question text}'` into a
persistent Chroma collection using a HuggingFace MiniLM embedding model.
`get_questions()` does a similarity search against the role name, split
into a technical bucket and an HR/Behavioral bucket, so every set always
mixes in a couple of HR/Behavioral questions alongside the role-relevant
technical ones — not just the single closest matches overall."

**6. What are embeddings, and why Chroma specifically?**
"An embedding turns text into a vector of numbers that captures its
meaning, so semantically similar text ends up as nearby vectors — that's
what lets a search for 'Java Developer' surface Java questions even if the
word 'Java' isn't in the question text verbatim. I used Chroma because
it's free, runs locally as a file-based store with no server to run or
pay for, which matched the project's 'free, no paid APIs' constraint."

**7. Why a separate per-session Chroma collection for resumes instead of
one shared store?**
"So different users' resumes never mix. Each session gets its own
collection named with the session id, which also makes it trivial to keep
resume data from leaking across users in a shared deployment."

### Speech-to-text

**8. How does speech-to-text work in your app? What API?**
"I use Groq's hosted Whisper API, `whisper-large-v3-turbo`, requesting
word-level and segment-level timestamps. I validate the audio first —
format, file size, minimum duration — before ever calling the API, so bad
uploads fail fast with a clear message instead of a confusing API error."

**9. Why Groq's hosted Whisper instead of running Whisper locally?**
"Free-tier hosted inference means no GPU requirement for the deployed app,
and it's fast. The one place I *did* run Whisper locally was training the
fluency model — I used `faster-whisper`'s small model offline there,
because transcribing 5,000 training utterances through a hosted free-tier
API would burn through rate limits fast, and training doesn't need a
hosted model."

**10. What happens if the Whisper API rate-limits you?**
"`_transcribe_with_retry()` catches `RateLimitError` and retries with
exponential backoff up to a configured limit before raising a clear
`TranscriptionError` — so a transient rate limit doesn't just crash the
interview."

### ML fluency model (data, features, metrics)

**11. Walk me through how you trained the ML fluency model.**
"I used speechocean762, 5,000 utterances of non-native speakers reading
sentences aloud with expert 0–10 fluency scores, bucketed into three
classes — Beginner, Intermediate, Fluent. I transcribed every utterance
locally with faster-whisper, ran the exact same `extract_features()`
function the live app uses — ten features like words-per-minute, pause
counts, filler rate, vocabulary variety — so training and inference
features can never drift apart. I tuned three candidates — Logistic
Regression, Random Forest, XGBoost — with 5-fold stratified cross-
validation on the train split, optimizing macro F1, then evaluated each
exactly once on the held-out test split."

**12. Why Random Forest over Logistic Regression or XGBoost here?**
"Random Forest had the best test macro F1 — 0.633 versus 0.598 for
Logistic Regression and 0.619 for XGBoost — so it won on the metric I
actually cared about, not just raw accuracy."

**13. What metric did you optimize for, and why not just accuracy?**
"Macro F1, because the classes are heavily imbalanced — about 66% of the
dataset is 'Fluent.' A model can post a deceptively high accuracy by just
always predicting the majority class while being useless for the minority
classes. My baseline majority-class predictor got 68% accuracy but a macro
F1 of only 0.27 — that gap is exactly why I tracked F1 throughout instead
of trusting accuracy alone."

**14. How do you turn class probabilities into a continuous 0–100 score?**
"Each class has a representative 'anchor' score on the 0–100 scale —
Beginner≈25, Intermediate≈65, Fluent≈90 — and the final score is those
anchors weighted by the model's predicted probability for each class. So
the score reflects not just which bucket won, but how confidently."

### LLM prompt design

**15. How did you design your LLM prompts to avoid hallucination?**
"A few concrete guardrails: the feedback prompt explicitly tells the LLM
to only list grammar corrections that are real mistakes actually in the
transcript, and to return an empty list rather than invent one; the
rewritten 'improved answer' must only use facts the candidate actually
said or that are in their resume context, never invented experience. For
resume-grounded questions, I go a step further — every generated question
must cite exactly which resume section it came from, and after generation
I programmatically check that section is real; a question citing a
hallucinated project gets dropped before the candidate ever sees it."

**16. What is structured output, and why do you use it everywhere?**
"Instead of asking the LLM for free text and hoping it's formatted right,
I use LangChain's `with_structured_output()` with a Pydantic schema — the
LLM is constrained to return data matching that exact shape, and Pydantic
validates it. I use this pattern for every LLM call in the project:
resume profile extraction, resume question generation, content feedback,
follow-up decisions, the practice plan, and custom-role question
generation."

**17. How does the interview agent decide to ask a follow-up?**
"It's a separate, focused LLM call after each answer — a `FollowUpDecision`
structured output with `should_follow_up`, the follow-up question text, and
a one-sentence reason. The prompt tells it to only follow up when it would
genuinely help — vague answer, something worth digging into, a missed key
point — not just to ask one for the sake of it. There's also a hard limit
of 2 follow-ups per main question so it can't spiral."

### Streamlit app

**18. Why Streamlit for the frontend?**
"It let me build a real multi-page app — Home, Setup, Interview, Report,
Progress — in pure Python, with `st.audio_input` and `st.camera_input`
giving me browser mic and webcam access for free, which would otherwise
be a decent amount of frontend/JavaScript work."

**19. How is session state and progress stored?**
"Each browser session gets a SQLite connection; `sessions` and `answers`
tables store every session and every answer, including scores, features,
and the full feedback payload as JSON. The Progress page reads every
completed session back out and plots trends with Plotly — score over
time, filler rate, speaking pace, and weakest categories averaged across
all history, not just the current session."

### Deployment

**20. How did you deploy this app, and what platform issues did you hit?**
"It's live on Streamlit Community Cloud, deployed straight from GitHub. I
hit two real platform issues: `mediapipe` needs `libGL`/`libEGL` system
libraries a minimal container doesn't ship, fixed with a `packages.txt`
apt-get list — and that file can't have `#` comments on Streamlit Cloud,
which I only found out after a build actually failed reading my comment
text as package names. The other was a Python-version mismatch — `numpy`
2.5 needs Python 3.12+, which has to be picked explicitly in the deploy
settings."

### CI/CD with GitHub Actions

**21. Walk me through your CI/CD pipeline.**
"Two jobs on every push to main: a lint job running `ruff check .`, and a
test job that installs dependencies and runs the full pytest suite — both
on `ubuntu-latest`. Both have to pass before I consider a change done."

**22. What was the hardest CI bug you fixed?**
"CI had actually never passed, not once, since the workflow was added —
and the failure message alone didn't make that obvious. It took three
rounds: first, Python 3.11 couldn't satisfy `numpy~=2.5`'s requirement for
3.12+, so I bumped the Python version — that got dependency install to
succeed for the first time, which immediately exposed a second problem,
missing `libGL`, which broke test *collection* because `body_language.py`
imports `cv2`/`mediapipe` at module level. Fixing that got further but
still failed on `libEGL`. I couldn't see GitHub's raw log output — this
repo requires signing in to view it — so I added a temporary step that
dumped pytest's output to the run's public job summary, asked for the
exact error text, fixed it, and removed the diagnostic step."

### Testing

**23. How did you test code that depends on an LLM?**
"Almost everything is mocked — `MagicMock` objects standing in for the
LLM client, with `with_structured_output()` returning pre-built Pydantic
objects I construct in the test. For the interview agent, which makes
several different structured-output calls on the same model, I built a
small `_FakeLLM` that routes each call to its own configurable mock by
schema class. A handful of true live tests do call the real Groq API, but
they're marked `pytest.mark.skipif(not config.GROQ_API_KEY, ...)` so they
skip automatically without a key instead of failing."

**24. What's your test coverage strategy — how many tests, what's
mocked?**
"161 tests, roughly one file per `src/` module. Every external
dependency — Groq LLM, Groq Whisper, the fluency model file, MediaPipe's
landmarkers — is mocked by default so the suite runs fast and
deterministically without needing real credentials or downloaded model
files; only a few explicitly-marked live smoke tests touch the real
APIs."

### Debugging stories

**25. Tell me about a bug you found and fixed after deployment.**
"On the live app, a custom-role interview sometimes failed with a generic
'something went wrong' message and nothing logged anywhere. I added proper
server-side logging first so I could actually see failures next time, then
noticed the real gap on closer reading: the custom-role question generator
had no retry around its LLM structured-output call, even though an
*identical* call in the content-feedback module already had one, added
specifically because Groq's structured output occasionally returns
invalid JSON. I added the same retry pattern there. I also temporarily
showed the raw exception on the page itself to confirm the real failure
mode on the live deployment before removing that debug code and shipping
the clean fix."

---

## 4. 10 tough follow-up questions, honestly answered

**1. Your fluency model is trained on read-aloud non-native speech, not
spontaneous interview answers — isn't that a mismatch?**
"Yes, honestly — it's the single biggest limitation of the fluency model,
and I say so directly in the README. speechocean762 is people reading a
fixed sentence aloud; spontaneous speech has different fluency
signals — self-correction, thinking pauses, filler words while
formulating an answer — that often look more varied than reading a known
sentence. I'd treat the current score as a reasonable starting signal, not
a validated measure of spontaneous-speech fluency. Closing that gap would
need a labeled spontaneous-speech dataset, which I don't have access to
for free."

**2. The model was trained on faster-whisper transcripts but the app uses
Groq's Whisper — doesn't that create train/serve skew?**
"Yes, and that's also documented honestly. The training transcripts come
from faster-whisper's local `small` model; production uses Groq's hosted
`whisper-large-v3-turbo`. Any feature downstream of a transcription
difference between those two models carries that mismatch. I chose this
trade-off deliberately — transcribing 5,000 training utterances through a
hosted free-tier API wasn't practical — but it's a real, acknowledged
limitation, not something I'd pretend isn't there."

**3. How would this scale to thousands of concurrent users?**
"Honestly, not well as it stands — SQLite is single-file and fine for one
user at a time, which is what a portfolio demo needs, but it's not built
for concurrent writers at scale. Scaling it for real would mean moving to
a managed database (Postgres), moving the Chroma vector store to a hosted
instance instead of a local file, and probably putting the LLM/Whisper
calls behind a queue so a burst of traffic doesn't just hit Groq's rate
limits directly."

**4. What happens to a user's resume and voice data — is it private?**
"Resumes are written to a local `data/resumes/` folder (git-ignored, never
committed) and embedded into a Chroma collection scoped to that one
session only, so different users' resumes don't mix. Audio isn't
persisted anywhere long-term in the app itself beyond what the browser
session holds for replay. That said, I'd be upfront that this is a
portfolio project, not a product with a real privacy policy or data
retention guarantee — on Streamlit Cloud's free tier the whole filesystem
is ephemeral and resets when the app restarts or sleeps, which is
actually a point in favor of not accidentally retaining data long-term,
but it isn't a designed privacy feature."

**5. Your webcam/voice-confidence features are heuristics, not trained
models — why ship something 'unvalidated'?**
"Because I'd rather ship an honest heuristic that says what it is than
pretend it's more rigorous than it is. Both features are explicitly
labeled in the UI and the README as heuristic signals, not trained,
validated models — unlike the fluency score, which is. I think that
honesty is actually the more defensible engineering choice than either
not shipping the feature at all, or shipping it with an implied precision
it doesn't have."

**6. The filler-word alert claims 'near real-time' but isn't literally
live — isn't that misleading?**
"I was careful to call it 'near-real-time,' not live, specifically to be
accurate. Streamlit reruns the whole page per interaction — it can't
stream microphone audio word-by-word while someone's still talking. What
it actually does is transcribe once the candidate finishes, then compute
a quick filler/pause count locally from that transcript with no LLM call
(so it's fast), and show it immediately — before the slower full LLM
feedback call even starts. It's the fastest *honest* substitute, and the
README explains why it isn't truly live rather than letting the name
imply something it doesn't do."

**7. What's your biggest single point of failure in this architecture?**
"Groq, for both the LLM and Whisper. If it's down or rate-limited, almost
nothing in the interview flow works — transcription, feedback, follow-up
decisions, the practice plan. I do have retry logic for rate limits and
for flaky structured-output responses, but there's no fallback provider.
For a real product I'd want at least a degraded mode — maybe skip the
follow-up decision and content feedback but still let the fluency model
and transcript stand on their own — rather than failing the whole
session."

**8. If Groq's API goes down or rate-limits you, what happens to the
user?**
"For Whisper specifically, rate limits are retried automatically with
exponential backoff before surfacing a clear error. For the LLM calls, a
structured-output failure after retries raises a specific error type
(`FeedbackError`, `QuestionGenerationError`) that the Streamlit page
catches and turns into a friendly 'something went wrong, please try again'
message, with the real exception logged server-side — not a raw crash or
traceback shown to the user."

**9. How would you detect if your LLM feedback is just wrong/bad advice?**
"Right now, I don't have an automated way to catch that — it's the
honest gap. The structured-output schema and prompt guardrails (only real
grammar mistakes, only facts the candidate actually said) catch
*hallucination*, but not subjective quality of the coaching advice itself.
If I were taking this further, I'd want a small eval set of
question/answer pairs with human-judged 'was this feedback actually
good and accurate' labels, and periodically spot-check the LLM's output
against it — similar in spirit to how I validated the ML fluency model
against a real held-out test set, which the LLM feedback currently
doesn't have an equivalent of."

**10. What would you change if you rebuilt this from scratch?**
"Two things: I'd add the retry pattern (the one I eventually added to the
custom-role generator) to *every* structured-output LLM call from the
start, instead of discovering the gap one feature at a time. And I'd set
up CI correctly from day one and actually confirm it was green — it took
me a while to even notice CI had never once passed, because the workflow
*looked* right; I'd add a habit of checking the Actions tab after the very
first push of any new CI config, not assuming a clean-looking YAML file
means it's actually running."

---

## 5. Challenges I faced and how I solved them (Situation, Action, Result)

### Story 1 — CI had been silently broken since the very first commit

**Situation:** I was asked to fix "the failing CI check," assuming it was
something recent. When I checked GitHub Actions, every single run in the
repo's history had failed — going back to the very first commit that added
the workflow.

**Action:** I checked the failing job's steps via the GitHub API and found
it was dying at `pip install -r requirements.txt`, failing in about two
seconds — too fast to be a real network/build issue, which pointed at a
dependency resolution error. I checked PyPI's own package metadata
directly and confirmed `numpy` 2.5's `requires_python` is `>=3.12`, while
the workflow pinned Python 3.11.

**Result:** Bumping the workflow to Python 3.12 fixed the install step —
which had literally never succeeded before — and let the test job run for
the first time ever, immediately surfacing the next real bug underneath
(see Story 2).

### Story 2 — Fixing one CI bug revealed a second, hidden one

**Situation:** With the Python version fixed, dependencies installed
successfully, but the test job still failed — now inside the test run
itself, not the install step.

**Action:** I reasoned from the code: `src/body_language.py` imports
`cv2` and `mediapipe` at the top of the file, and I already knew (from
preparing the Streamlit Cloud deployment) that mediapipe needs system
graphics libraries a minimal container doesn't ship. I added an
`apt-get install -y libgl1 libglib2.0-0` step to CI, matching what I'd
already set up in `packages.txt` for the deployed app.

**Result:** That got further but still failed, now with a *different*
missing library (`libEGL.so.1`) — which led directly into Story 3, because
I had no way to see the actual error text.

### Story 3 — I couldn't see the CI logs, so I built a way to see them

**Situation:** GitHub's Actions log viewer required signing in for this
repo, and so did the job summary page — I had no credentials to use and
wasn't going to try to work around that. I had a failing test job and no
visibility into *why*, beyond "exit code 1."

**Action:** Rather than guessing blindly, I added a temporary step to the
workflow that ran pytest, captured its output, and wrote it into the run's
`$GITHUB_STEP_SUMMARY` — which I'd hoped would be visible without signing
in. It turned out that was *also* gated, so I asked for the output to be
copied and pasted back to me directly from the Actions tab.

**Result:** The pasted output showed the real error:
`OSError: libEGL.so.1: cannot open shared object file`, affecting 9 tests
in `test_body_language.py`. I added `libegl1` and `libgles2` to both the
CI apt-get step and `packages.txt`, removed the temporary diagnostic step,
and CI went fully green for the first time in the project's history.

### Story 4 — A retry-logic gap caused silent failures in a newer feature

**Situation:** After adding a custom-role interview feature (type any job
role, get LLM-generated questions for it), the live deployed app sometimes
showed "Something went wrong while preparing your interview" with no
indication of why.

**Action:** I first made the failure loud instead of silent — adding
`logger.exception()` so the real traceback would reach server logs instead
of vanishing. Then, reading the new code against the rest of the codebase,
I noticed `generate_custom_role_questions()` called the LLM's structured
output directly with no retry, while an *almost identical* call in
`src/feedback.py` already had one — added specifically because Groq's
structured output occasionally returns malformed JSON. The new feature
simply hadn't inherited that protection.

**Result:** I added the same retry pattern (a new
`CUSTOM_ROLE_LLM_MAX_RETRIES` config value and a dedicated
`QuestionGenerationError`), with a matching test for both the retry-then-
succeed case and the retries-exhausted case.

### Story 5 — `packages.txt` comments broke the Streamlit Cloud build

**Situation:** While preparing the Streamlit Cloud deployment, I wrote a
`packages.txt` file with explanatory `#` comments above the actual package
names (standard practice in most config files). The deploy failed with
errors like `E: Unable to locate package Hugging` and `E: Unable to locate
package #`.

**Action:** The errors made it obvious almost immediately: Streamlit
Cloud's apt installer was reading the words inside my comments as if they
were package names to install. I stripped every comment out of
`packages.txt`, leaving only the bare package names, one per line, and
moved the explanation into the README instead.

**Result:** The build installed cleanly. It was a small fix, but a good
reminder that a config file format that looks standard in one tool (shell
scripts, most Linux config files) can be interpreted completely
differently by another tool reading the same file.

---

## 6. Future improvements

> `[NOTE for you: the task brief I was given asked me to list "the planned
> webcam body-language coach, live filler-word alerts and the voice
> confidence meter" here as future work — but reading the actual code and
> README, all three are already fully built and shipped (Tasks 11, 12, 13
> in PROJECT_NOTES.md, with their own tests). I didn't want to describe
> finished features as "planned," so below is the project's real,
> documented backlog instead — pulled directly from README.md's own
> "Future improvements" section. Please double-check this swap makes
> sense to you.]`

- **Close the read-aloud → spontaneous-speech gap** — add non-native
  accent variety and real spontaneous-speech data to the fluency training
  set, since speechocean762 is read-aloud speech, not interview answers.
- **Persistent storage** — point the Chroma index and SQLite history at
  external/managed storage, so a Streamlit Cloud restart doesn't wipe
  session history.
- **A real continuous webcam recorder** — replace the current
  manual-snapshot approach (Streamlit has no built-in video recorder) with
  something like a custom `streamlit-webrtc` component, for a less coarse
  eye-contact/posture signal.
- **True word-by-word live filler alerts** — a streaming-ASR frontend
  (e.g. a WebSocket client) instead of today's honest near-real-time,
  post-answer timeline.
- **Multi-language support** — question bank, prompts, and TTS language.
- **Per-question difficulty adaptation** — adjust difficulty mid-interview
  based on how the candidate is actually doing, not just the difficulty
  picked at setup.
- **A local forced-alignment/phoneme-confidence model** — for genuine
  per-word pronunciation scoring, replacing today's honest
  segment-confidence proxy ("words to double check").

If asked live in an interview "what would you add next," picking **one**
of these and explaining it in a sentence or two (e.g. the streaming-ASR
one, since it directly follows from the "near-real-time, not live" caveat)
tends to land better than listing all of them.

---

## 7. One-page quick revision cheat sheet

**What it is:** AI voice mock-interview coach — speak answers, get scored
on fluency (ML) + content (LLM), follow-ups, practice plan, progress
tracking. Live at https://speakready-sindhu.streamlit.app

**Core pipeline:** Resume + question bank → RAG → question asked by voice
(gTTS) → candidate speaks → Groq Whisper STT → features extracted →
ML fluency model (Random Forest) scores delivery + LLM scores content →
combined score → agent decides on a follow-up → saved to SQLite →
report/progress.

**Key numbers to have ready:**
- 159 questions in the static question bank, 8 categories.
- 5,000 training utterances (speechocean762), 2,500 train / 2,500 test.
- 3 fluency classes: Beginner (0–5), Intermediate (6–7), Fluent (8–10).
- Winner: **Random Forest**, test accuracy **72.7%**, test macro F1
  **0.633** (vs. 68.0% / 0.270 for the majority-class baseline).
- Overall score = 0.6 × content (LLM, 0–10 scaled ×10) + 0.4 × fluency
  (ML, 0–100).
- **161 pytest tests**, almost everything mocked; a few live tests
  auto-skip without an API key.
- Max 2 follow-up questions per main question.

**Tech stack one-liners:**
- Streamlit (UI) · Groq LLM `openai/gpt-oss-120b` (feedback, follow-ups,
  practice plan, custom roles) · Groq Whisper `whisper-large-v3-turbo`
  (STT) · gTTS (TTS) · LangChain + Chroma + MiniLM embeddings (RAG) ·
  scikit-learn Random Forest (fluency ML) · MediaPipe (body language) ·
  librosa (voice confidence) · SQLite (storage) · pytest + GitHub Actions
  (tests/CI) · Streamlit Community Cloud (deployment).

**The three debugging stories to have ready:**
1. CI failed on every run since day one — Python 3.11 vs. `numpy`'s
   `>=3.12` requirement.
2. Fixing that revealed missing `libGL`/`libEGL` for mediapipe on bare
   Ubuntu CI runners (same fix needed in `packages.txt` for deployment).
3. A missing retry on a newer LLM call (custom-role question generation)
   caused silent live-app failures — fixed by copying the retry pattern
   already used in the feedback engine.

**The honesty theme (good for "tell me about a design decision"):**
Every heuristic/limitation is labeled as what it is, not oversold —
body language and voice confidence are explicitly "heuristic, not a
trained model"; the filler alert is "near-real-time, not live"; the
pronunciation signal is "a proxy, not a verdict"; the README has a
dedicated Limitations section. This is a deliberate, repeatable design
choice, not an afterthought — worth naming directly if asked about
engineering judgment.

**If you only remember one sentence:** "I combined a trained ML model, RAG,
real speech processing, and a stateful LLM agent into one working, tested,
deployed product — and when things broke in production, I debugged them
with evidence (API data, local reproduction, careful log access) instead
of guessing."

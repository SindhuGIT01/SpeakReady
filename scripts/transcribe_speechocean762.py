"""One-off script: transcribe the speechocean762 dataset with faster-whisper.

Runs locally (no Groq calls) so the whole 5000-utterance dataset can be
transcribed without hitting Groq's free-tier rate limits. Results are cached
to ``data/whisper_cache/{split}.jsonl`` and appended incrementally, so
interrupting and re-running skips utterances already transcribed.

Parallelized across processes (each with a single-threaded model instance)
because a single multi-threaded ``small`` model instance is CPU-bound and
slow per utterance on a laptop CPU; N single-threaded workers in parallel
give close to an N-x speedup for this many independent, short clips.

Usage: python scripts/transcribe_speechocean762.py
"""

from __future__ import annotations

import io
import json
import multiprocessing as mp
import time
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset

WHISPER_SIZE = "small"
NUM_WORKERS = 6
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "whisper_cache"

_model = None


def _init_worker() -> None:
    """Load one faster-whisper model per worker process (module-global)."""
    global _model
    from faster_whisper import WhisperModel

    _model = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8", cpu_threads=1)


def _transcribe_one(task: tuple[str, int, bytes]) -> dict:
    """Transcribe a single cached audio clip with word-level timestamps.

    Args:
        task: (split name, row index, raw audio file bytes).

    Returns:
        A JSON-serializable dict with the row index, hypothesis text,
        per-word (word, start, end) timestamps, and audio duration.
    """
    split, idx, audio_bytes = task
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    duration = len(data) / sr

    segments, _ = _model.transcribe(
        data,
        language="en",
        word_timestamps=True,
        beam_size=1,
        condition_on_previous_text=False,
    )
    segments = list(segments)
    words = [
        [w.word.strip(), round(w.start, 3), round(w.end, 3)]
        for seg in segments
        for w in (seg.words or [])
    ]
    text = " ".join(seg.text.strip() for seg in segments)
    return {"idx": idx, "text": text, "words": words, "duration": duration}


def _already_done(cache_path: Path) -> set[int]:
    """Row indices already transcribed and cached for a split.

    Args:
        cache_path: Path to the split's JSONL cache file.

    Returns:
        The set of row indices found in the cache file, if it exists.
    """
    if not cache_path.exists():
        return set()
    done = set()
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["idx"])
    return done


def transcribe_split(split: str) -> None:
    """Transcribe every utterance in one dataset split, caching to disk.

    Args:
        split: Either ``"train"`` or ``"test"``.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{split}.jsonl"
    done = _already_done(cache_path)

    ds = load_dataset("mispeech/speechocean762", split=split)
    ds = ds.cast_column("audio", Audio(decode=False))

    tasks = [
        (split, i, ds[i]["audio"]["bytes"]) for i in range(len(ds)) if i not in done
    ]
    total = len(ds)
    print(f"[{split}] {len(done)}/{total} already cached, {len(tasks)} remaining")
    if not tasks:
        return

    start = time.time()
    completed = len(done)
    with cache_path.open("a", encoding="utf-8") as f, mp.Pool(
        NUM_WORKERS, initializer=_init_worker
    ) as pool:
        for result in pool.imap_unordered(_transcribe_one, tasks, chunksize=1):
            f.write(json.dumps(result) + "\n")
            f.flush()
            completed += 1
            if completed % 50 == 0 or completed == total:
                elapsed = time.time() - start
                rate = (completed - len(done)) / elapsed if elapsed > 0 else 0
                eta_min = (total - completed) / rate / 60 if rate > 0 else float("nan")
                print(
                    f"[{split}] {completed}/{total} "
                    f"({rate:.2f} samples/sec, ETA {eta_min:.1f} min)"
                )


def main() -> None:
    """Transcribe both dataset splits."""
    for split in ("train", "test"):
        transcribe_split(split)
    print("Done.")


if __name__ == "__main__":
    main()

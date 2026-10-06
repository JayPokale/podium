"""Build the Podium contrastive dataset.

For each public-domain speech:
  1. force-align the full transcript and cut 50-80 s excerpts at sentence boundaries -> "ideal" baselines;
  2. bad mirrors of the *same audio*: a gradient L0 (untouched) .. L4 (egregious), plus single-flaw stress
     tests for every flaw type x severity;
  3. different-speaker readings of the same text with open Piper TTS voices, clean and with flaws,
     to test that the comparison is speaker-agnostic.
Every file gets a JSON label with flaw type, severity, affected words and start/end in that file.

Split: JFK speeches calibrate thresholds; Reagan (another speaker, another speech) is held out for testing.
"""
import json
import re
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from podium import align, audio, flaws  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "dataset"
SPEECHES = {  # name: (split, speaker)
    "jfk_inaugural": ("calibration", "John F. Kennedy"),
    "jfk_moon": ("calibration", "John F. Kennedy"),
    "reagan_challenger": ("test", "Ronald Reagan"),
}
# Applause-free stretches, picked from a first Whisper pass (seconds in the source file). Each window is
# transcribed on its own (long-form transcription hallucinates over applause), force-aligned, and cut
# to whole sentences.
WINDOWS = {
    "jfk_inaugural": [(465.5, 546.0), (786.5, 841.0)],
    "jfk_moon": [(47.0, 118.5), (576.5, 644.5)],
    "reagan_challenger": [(0.0, 54.2), (63.0, 147.5), (183.5, 251.0)],
}
VOICES = {"calibration": ["en_US-lessac-medium", "en_GB-alan-medium"], "test": ["en_US-ryan-medium"]}
GRADIENT = {  # level: list of (type, severity)
    "L0": [],
    "L1": [("rushed", 1), ("monotone", 1)],
    "L2": [("rushed", 2), ("long_pause", 1), ("mumbled", 2)],
    "L3": [("rushed", 2), ("monotone", 2), ("long_pause", 2), ("stutter", 2), ("no_pause", 2)],
    "L4": [("rushed", 3), ("dragged", 3), ("monotone", 3), ("mumbled", 3), ("loud", 3),
           ("long_pause", 3), ("stutter", 3)],
}


def tokens_with_breaks(text):
    """Normalised words plus a flag marking words that end a sentence."""
    out = []
    for tok in text.split():
        ws = audio.words_of(tok)
        for k, w in enumerate(ws):
            out.append((w, k == len(ws) - 1 and bool(re.search(r"[.!?][\"')]*$", tok))))
    return out


def excerpts(aligned, ends, lo=50, hi=80, max_n=3):
    """Non-overlapping windows that start after and end on a sentence boundary, with no applause gap."""
    starts = [0] + [i + 1 for i, e in enumerate(ends[:-1]) if e]
    found, i_after = [], 0
    for s in starts:
        if s < i_after:
            continue
        best = None
        for e in range(s, len(aligned)):
            dur = aligned[e]["end"] - aligned[s]["start"]
            if dur > hi:
                break
            gaps = [aligned[k]["start"] - aligned[k - 1]["end"] for k in range(s + 1, e + 1)]
            if gaps and max(gaps) > 2.5:
                break  # applause or an edit: not a clean stretch of speech
            if ends[e] and dur >= lo:
                best = e  # keep extending: the longest clean excerpt within limits
        if best is not None:
            found.append((s, best))
            i_after = best + 1
        if len(found) >= max_n:
            break
    return found


def tts(text, voice, path):
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        subprocess.run([sys.executable, "-m", "piper", "-m", str(DATA / "voices" / f"{voice}.onnx"),
                        "-f", tmp.name], input=text.encode(), check=True, capture_output=True)
        y = audio.load(tmp.name)
    audio.save(path, y)
    return y


def write(rec_dir, name, y, meta):
    audio.save(rec_dir / f"{name}.wav", y)
    (rec_dir / f"{name}.json").write_text(json.dumps(meta, indent=1))


def transcribe_windows():
    """Whisper (open weights) on each window; cached in data/asr/windows.json."""
    cache = DATA / "asr" / "windows.json"
    done = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [(sp, k, w) for sp, ws in WINDOWS.items() for k, w in enumerate(ws) if f"{sp}_{k + 1}" not in done]
    if todo:
        import torch
        from transformers import pipeline as hf_pipeline
        # float32: on this GTX 1660 Ti, float16 Whisper silently returns empty text
        asr = hf_pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo",
                          dtype=torch.float32, device=0 if torch.cuda.is_available() else -1)
        for sp, k, (a, b) in todo:
            y = audio.load(DATA / "raw" / f"{sp}.ogg", start=a, end=b)
            done[f"{sp}_{k + 1}"] = asr(y.copy(), chunk_length_s=30, generate_kwargs={"language": "en"})["text"].strip()
            print("transcribed", sp, k + 1)
        cache.write_text(json.dumps(done, indent=1))
        del asr
        torch.cuda.empty_cache()
    return done


def main():
    index = []
    texts = transcribe_windows()
    for speech, (split, speaker) in SPEECHES.items():
        y_full = audio.load(DATA / "raw" / f"{speech}.ogg")
        for n, (wa, wb) in enumerate(WINDOWS[speech]):
            ex = f"{speech}_{n + 1}"
            toks = tokens_with_breaks(texts[ex])
            y_win = y_full[int(wa * audio.SR): int(wb * audio.SR)]
            win = align.align(y_win, " ".join(w for w, _ in toks))
            ends = [e for _, e in toks]
            cut = excerpts(win, ends, lo=40, hi=85, max_n=1)
            if not cut:
                print("no clean excerpt in", ex)
                continue
            s, e = cut[0]
            d = OUT / split / ex
            d.mkdir(parents=True, exist_ok=True)
            t0, t1 = max(0.0, win[s]["start"] - 0.3), win[e]["end"] + 0.4
            y = y_win[int(t0 * audio.SR): int(t1 * audio.SR)]
            transcript = " ".join(w for w, _ in toks[s:e + 1])
            al = align.align(y, transcript)
            (d / "transcript.txt").write_text(transcript + "\n")
            base_meta = {"excerpt": ex, "speech": speech, "speaker": speaker, "split": split,
                         "source": f"data/raw/{speech}.ogg", "source_start": round(wa + t0, 3),
                         "source_end": round(wa + t1, 3), "kind": "baseline", "flaws": [], "words": al}
            write(d, "baseline", y, base_meta)
            index.append({"excerpt": ex, "file": f"{split}/{ex}/baseline.wav", "kind": "baseline"})

            def mirror(name, y_src, al_src, plan, seed, kind, voice=None):
                rng = np.random.default_rng(seed)
                spans = flaws.pick_spans(al_src, plan, rng)
                y2, labels, keys = flaws.inject(y_src, al_src, spans, seed=seed)
                words = [{**w, "start": round(flaws.remap(w["start"], keys), 3),
                          "end": round(flaws.remap(w["end"], keys), 3)} for w in al_src]
                write(d, name, y2, {"excerpt": ex, "split": split, "kind": kind, "voice": voice,
                                    "flaws": labels, "words": words, "reference": "baseline.wav"})
                index.append({"excerpt": ex, "file": f"{split}/{ex}/{name}.wav", "kind": kind,
                              "voice": voice, "n_flaws": len(labels)})

            seed = zlib.crc32(ex.encode()) % 10_000
            for lvl, plan in GRADIENT.items():
                mirror(f"mirror_{lvl}", y, al, plan, seed + int(lvl[1]), "gradient")
            for kind in flaws.TYPES:
                for sev in (1, 2, 3):
                    mirror(f"stress_{kind}_{sev}", y, al, [(kind, sev)], seed + 100 + sev, "stress")
            for voice in VOICES[split]:
                yv = tts(transcript, voice, d / f"tts_{voice}.wav")
                alv = align.align(yv, transcript)
                for lvl in ("L0", "L2", "L4"):
                    mirror(f"tts_{voice}_{lvl}", yv, alv, GRADIENT[lvl], seed + 7 + int(lvl[1]), "tts", voice)
                (d / f"tts_{voice}.wav").unlink()
            print(ex, f"{t1 - t0:.1f}s", len(al), "words", f"min align score {min(w['score'] for w in al):.2f}")
    (OUT / "index.json").write_text(json.dumps(index, indent=1))
    print(len(index), "files")


if __name__ == "__main__":
    main()

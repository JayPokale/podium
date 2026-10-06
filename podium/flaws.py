"""Build the "bad mirror" of a good speech: the same audio and transcript with delivery flaws injected.

Because we inject the flaws ourselves, every flawed file comes with exact labels: flaw type, severity
(1 mild, 2 clear, 3 egregious), the words affected and the start/end time in the *new* recording.
That turns a subjective task into one we can score: did the analyzer find the flaw, and where?

Flaw types:
  rushed      words delivered faster (time-compressed, pauses squeezed out)
  dragged     words delivered slower
  monotone    pitch movement flattened toward the phrase mean (WORLD vocoder)
  mumbled     quieter and muffled (gain down + low-pass)
  loud        sudden volume spike with soft clipping
  long_pause  an awkward silence inserted mid-phrase
  no_pause    a natural breath pause between phrases removed
  stutter     the start of a word repeated ("w- w- we choose")
"""
import numpy as np
import pyworld
from scipy.signal import butter, sosfiltfilt
import librosa

from .audio import SR

TYPES = ["rushed", "dragged", "monotone", "mumbled", "loud", "long_pause", "no_pause", "stutter"]
SEVERITY = {
    "rushed": {1: 1.3, 2: 1.6, 3: 2.0},          # speed factor
    "dragged": {1: 0.8, 2: 0.65, 3: 0.5},
    "monotone": {1: 0.5, 2: 0.75, 3: 0.95},      # fraction of pitch movement removed
    "mumbled": {1: (-6, 3000), 2: (-11, 1800), 3: (-16, 1100)},  # (gain dB, low-pass Hz)
    "loud": {1: 4, 2: 7, 3: 10},                 # gain dB
    "long_pause": {1: 0.7, 2: 1.3, 3: 2.2},      # seconds of inserted silence
    "no_pause": {1: 0.5, 2: 0.2, 3: 0.0},        # fraction of the natural pause kept
    "stutter": {1: 1, 2: 2, 3: 3},               # extra repetitions of the word onset
}
WORDS_PER_FLAW = {"rushed": (5, 9), "dragged": (4, 7), "monotone": (7, 12), "mumbled": (4, 8),
                  "loud": (3, 6), "long_pause": (1, 1), "no_pause": (1, 1), "stutter": (1, 1)}


def _xfade_join(parts):
    """Concatenate with 5 ms fades at each edit point so edits don't click. Lengths are preserved
    exactly, so label times computed from segment lengths stay sample-accurate."""
    n = int(0.005 * SR)
    out = []
    for k, p in enumerate(parts):
        p = p.astype(np.float32).copy()
        if len(p) > 2 * n:
            if k:
                p[:n] *= np.linspace(0, 1, n, dtype=np.float32)
            if k < len(parts) - 1:
                p[-n:] *= np.linspace(1, 0, n, dtype=np.float32)
        out.append(p)
    return np.concatenate(out)


def _room_tone(y, seconds, rng):
    """Silence that sounds like the room: noise at the recording's own noise floor."""
    hop = int(0.05 * SR)
    rms = np.array([np.sqrt(np.mean(y[i:i + hop] ** 2)) for i in range(0, len(y) - hop, hop)])
    floor = np.percentile(rms, 5)
    return (rng.standard_normal(int(seconds * SR)) * floor).astype(np.float32)


def _monotone(seg, alpha):
    x = seg.astype(np.float64)
    f0, t = pyworld.harvest(x, SR, f0_floor=70, f0_ceil=400, frame_period=5.0)
    sp = pyworld.cheaptrick(x, f0, t, SR)
    ap = pyworld.d4c(x, f0, t, SR)
    v = f0 > 0
    if v.sum() < 5:
        return seg
    st = 12 * np.log2(f0[v])
    st = st.mean() + (st - st.mean()) * (1 - alpha)
    f0[v] = 2 ** (st / 12)
    out = pyworld.synthesize(f0, sp, ap, SR, frame_period=5.0)[: len(seg)]
    out = np.pad(out, (0, max(0, len(seg) - len(out))))
    # match loudness of the original segment
    return (out * (np.sqrt(np.mean(seg ** 2)) / (np.sqrt(np.mean(out ** 2)) + 1e-9))).astype(np.float32)


def _apply(kind, sev, seg, rng, y):
    p = SEVERITY[kind][sev]
    if kind in ("rushed", "dragged"):
        return librosa.effects.time_stretch(seg, rate=p)
    if kind == "monotone":
        return _monotone(seg, p)
    if kind == "mumbled":
        gain, cutoff = p
        sos = butter(6, cutoff, btype="low", fs=SR, output="sos")
        return (sosfiltfilt(sos, seg) * 10 ** (gain / 20)).astype(np.float32)
    if kind == "loud":
        g = 10 ** (p / 20)
        return (np.tanh(seg * g * 1.2) / 1.2).astype(np.float32)
    raise ValueError(kind)


def pick_spans(aligned, plan, rng, min_gap=3):
    """Choose non-overlapping word spans for each (type, severity) in plan."""
    taken = np.zeros(len(aligned), bool)
    spans = []
    for kind, sev in plan:
        for _ in range(200):
            lo, hi = WORDS_PER_FLAW[kind]
            n = int(rng.integers(lo, hi + 1))
            i = int(rng.integers(1, len(aligned) - n - 1))
            j = i + n - 1
            if taken[max(0, i - min_gap): j + min_gap + 1].any():
                continue
            gap = aligned[i]["start"] - aligned[i - 1]["end"]
            if kind == "no_pause" and gap < 0.35:
                continue  # needs a real breath pause to remove
            if kind == "long_pause" and gap > 0.12:
                continue  # inserted mid-phrase, where nobody pauses
            if kind == "stutter" and (aligned[i]["end"] - aligned[i]["start"] < 0.18):
                continue
            taken[i: j + 1] = True
            spans.append({"type": kind, "severity": sev, "i": i, "j": j})
            break
    return sorted(spans, key=lambda s: s["i"])


def inject(y, aligned, spans, seed=0):
    """Apply spans to y. Returns (new audio, labels with times in the new audio, time-map keypoints)."""
    rng = np.random.default_rng(seed)
    parts, keys, labels = [], [(0.0, 0.0)], []
    cursor = 0  # sample index in the original
    out_len = 0

    def emit(seg):
        nonlocal out_len
        parts.append(seg)
        out_len += len(seg)

    for s in spans:
        kind, sev, i, j = s["type"], s["severity"], s["i"], s["j"]
        w0, w1 = aligned[i], aligned[j]
        if kind in ("long_pause", "no_pause"):
            prev_end = int(aligned[i - 1]["end"] * SR)
            nxt = int(w0["start"] * SR)
            emit(y[cursor:prev_end])
            if kind == "long_pause":
                gap = y[prev_end:nxt]
                sil = _room_tone(y, SEVERITY[kind][sev], rng)
                a = out_len / SR
                emit(gap[: len(gap) // 2]); emit(sil); emit(gap[len(gap) // 2:])
                b = out_len / SR
                keys += [(prev_end / SR, a - len(gap) // 2 / SR), (nxt / SR, b)]
                labels.append({**s, "start": round(a, 3), "end": round(b, 3)})
            else:
                gap = y[prev_end:nxt]
                keep = int(len(gap) * SEVERITY[kind][sev]) or int(0.03 * SR)
                a = out_len / SR
                emit(np.concatenate([gap[: keep // 2], gap[len(gap) - (keep - keep // 2):]]))
                b = out_len / SR
                keys += [(prev_end / SR, a), (nxt / SR, b)]
                # the flaw is the missing breath: label the joint between the two phrases
                labels.append({**s, "start": round(max(0, a - 0.25), 3), "end": round(b + 0.25, 3)})
            cursor = nxt
            continue
        a_in, b_in = int(w0["start"] * SR), int(w1["end"] * SR)
        emit(y[cursor:a_in])
        a = out_len / SR
        if kind == "stutter":
            onset = y[a_in: a_in + int(min(0.16, (w0["end"] - w0["start"]) * 0.45) * SR)]
            gap = _room_tone(y, 0.07, rng)
            for _ in range(SEVERITY[kind][sev]):
                emit(onset); emit(gap)
            emit(y[a_in:b_in])
        else:
            emit(_apply(kind, sev, y[a_in:b_in], rng, y))
        b = out_len / SR
        keys += [(a_in / SR, a), (b_in / SR, b)]
        labels.append({**s, "start": round(a, 3), "end": round(b, 3)})
        cursor = b_in
    emit(y[cursor:])
    keys.append((len(y) / SR, out_len / SR))
    return _xfade_join(parts), labels, keys


def remap(t, keys):
    """Map a time in the original recording to the flawed one (piecewise linear between edits)."""
    ks = sorted(keys)
    xs, ys = [k[0] for k in ks], [k[1] for k in ks]
    return float(np.interp(t, xs, ys))

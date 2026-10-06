"""Compare a participant's delivery with a reference delivery of the same text and explain the flaws.

1. Both recordings are force-aligned to the same transcript, so word i matches word i.
2. For every word we compute deltas on speaker-normalised features (pace, pauses, pitch movement,
   loudness, clarity, word length).
3. Global style vs local flaws: a different speaker is allowed to be a bit slower or flatter
   everywhere. We take the median delta over the whole speech as the speaker's overall style (reported
   separately) and look for *local* departures from it, as robust z-scores (median / MAD).
4. Words whose z-score passes a threshold are merged into flaw regions with start/end times, and each
   region gets a plain-language explanation built from the measured numbers.
"""
import json
import math
from pathlib import Path

import numpy as np

# Floors for the robust scale, so a near-identical recording doesn't turn tiny jitter into huge z-scores.
SCALE_FLOOR = {"pace": 0.10, "pause": 0.10, "pitch": 1.0, "volume": 1.5, "clarity": 1.5, "length": 0.15}
THRESH = {"rushed": 3.0, "dragged": 3.0, "monotone": 3.0, "mumbled": 3.0, "loud": 3.0,
          "long_pause": 3.5, "no_pause": 3.0, "stutter": 3.5}
# Thresholds tuned on the calibration split by scripts/evaluate.py, then frozen in results/thresholds.json.
_TUNED = Path(__file__).resolve().parents[1] / "results" / "thresholds.json"
if _TUNED.exists():
    THRESH.update(json.loads(_TUNED.read_text()))
LABEL = {"rushed": "Rushed pacing", "dragged": "Dragging pace", "monotone": "Monotone delivery",
         "mumbled": "Mumbled / too quiet", "loud": "Volume spike", "long_pause": "Awkward pause",
         "no_pause": "Missing breath pause", "stutter": "Stumble / repetition"}
DIMENSION = {"rushed": "pacing", "dragged": "pacing", "long_pause": "pauses", "no_pause": "pauses",
             "monotone": "pitch", "mumbled": "volume", "loud": "volume", "stutter": "fluency"}


def _nan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def _robust(v, floor):
    v = np.asarray(v, float)
    ok = ~np.isnan(v)
    med = float(np.median(v[ok])) if ok.any() else 0.0
    mad = float(np.median(np.abs(v[ok] - med))) * 1.4826 if ok.any() else 0.0
    return med, max(mad, floor)


def deltas(base, part):
    """Per-word raw deltas between participant and reference rows (same length, same words)."""
    d = {k: [] for k in ("pace", "pause", "pitch", "volume", "clarity", "length")}
    for b, p in zip(base, part):
        d["pace"].append(math.log2(p["rate_local"] / b["rate_local"]))
        d["pause"].append(p["pause_before"] - b["pause_before"])
        d["pitch"].append(np.nan if _nan(p["f0_spread"]) or _nan(b["f0_spread"]) else p["f0_spread"] - b["f0_spread"])
        d["volume"].append(np.nan if _nan(p["int"]) or _nan(b["int"]) else p["int"] - b["int"])
        d["clarity"].append(np.nan if _nan(p["hf"]) or _nan(b["hf"]) else p["hf"] - b["hf"])
        d["length"].append(math.log2(p["dur"] / b["dur"]))
    return {k: np.array(v, float) for k, v in d.items()}


def _rz(v, floor):
    med, scale = _robust(v, floor)
    return (np.asarray(v, float) - med) / scale


def word_scores(base, part):
    """Per-word flaw scores. Each one needs two kinds of evidence:

    z_ref   the word departs from the reference, beyond the speaker's overall style offset;
    z_self  the word also stands out within the speaker's *own* delivery.

    The score is the smaller of the two (a soft AND). A different voice that is consistently slower or
    flatter than the reference moves z_ref but not z_self, so it is reported as style, not as a flaw.
    """
    d = deltas(base, part)
    style, z = {}, {}
    for k, v in d.items():
        med, scale = _robust(v, SCALE_FLOOR[k])
        style[k] = med
        z[k] = (v - med) / scale
    n = len(base)
    win = lambda a, i, r=2: a[max(0, i - r):i + r + 1]
    ln = z["length"]
    # Pace = median word-length change over a 5-word window. The median ignores one stretched word,
    # so a single stumble doesn't read as a slow phrase, while a time-compressed phrase moves every word.
    # Articulation rate (syllables per second of actual speech, pauses excluded) over a 5-word window.
    # Summing over the window averages out the 20 ms alignment jitter of individual short words.
    def art(rows):
        syl = np.array([r["syl"] for r in rows], float); dur = np.array([r["dur"] for r in rows], float)
        return np.array([win(syl, i).sum() / max(win(dur, i).sum(), 0.05) for i in range(n)])
    art_b, art_p = art(base), art(part)
    pace_ref = _rz(np.log2(art_b / art_p), 0.08)  # positive = slower than the reference, beyond style
    neigh = np.array([np.nanmedian(np.r_[ln[max(0, i - 3):i], ln[i + 1:i + 4]]) if n > 1 else 0 for i in range(n)])
    extra = np.array([p["dur"] - b["dur"] for b, p in zip(base, part)])
    bp = np.array([b["pause_before"] for b in base])
    pp = np.array([p["pause_before"] for p in part])

    # Self-referenced evidence, from the participant's recording alone.
    spw = np.array([p["dur"] / p["syl"] for p in part])  # seconds per syllable, per word
    own_len = _rz(np.log2(spw), 0.15)
    # own pace: syllables per second over a 5-word window, compared with the speaker's typical pace
    pace_self = -_rz(np.log2(art_p), 0.06)  # positive = slower than the speaker's own usual pace
    pitch_self = _rz([p["f0_spread"] for p in part], 1.0)
    vol_self = _rz([p["int"] for p in part], 1.5)
    clar_self = _rz([p["hf"] for p in part], 1.5)
    inner = bp < 0.15  # places where the reference does not pause: mid-phrase
    med_in, sc_in = _robust(pp[inner], 0.08) if inner.any() else (0.0, 0.08)
    long_self = (pp - med_in) / sc_in
    edge = bp > 0.3   # phrase boundaries where the reference breathes
    med_e, sc_e = _robust(pp[edge], 0.1) if edge.any() else (0.0, 0.1)
    short_self = (med_e - pp) / sc_e
    gv = np.array([p["gap_voiced"] - b["gap_voiced"] for b, p in zip(base, part)])
    stretch_self = own_len - np.array([np.nanmedian(np.r_[own_len[max(0, i - 3):i], own_len[i + 1:i + 4]]) for i in range(n)])

    both = lambda r, s_: np.fmin(r, s_)
    s = {
        "rushed": both(-pace_ref, -pace_self),
        "dragged": both(pace_ref, pace_self),
        "monotone": both(-z["pitch"], -pitch_self),
        "mumbled": both(np.fmax(-z["volume"], 0) * 0.6 + np.fmax(-z["clarity"], 0) * 0.6,
                        np.fmax(-vol_self, 0) * 0.6 + np.fmax(-clar_self, 0) * 0.6),
        "loud": both(z["volume"], vol_self),
        # a long pause is a *silent* gap; a gap full of voiced sound is a restart, scored as a stumble
        "long_pause": np.where(gv < 0.15, both(z["pause"], long_self), 0.0),
        "no_pause": np.where(edge, both(-z["pause"], short_self), 0.0),
        "stutter": np.fmax(np.where(extra > 0.15, both(ln - np.fmax(neigh, 0), stretch_self), 0.0),
                           np.where(gv > 0.12, gv / 0.06, 0.0)),
    }
    # Words the aligner could not place confidently in the reference (numbers read aloud, names) give
    # unreliable timings, so they can't raise flags on their own.
    trusted = np.array([b["score"] >= 0.1 for b in base])
    return {k: np.where(trusted, np.nan_to_num(v, nan=0.0), 0.0) for k, v in s.items()}, style, d


def regions(base, part, scores, d, max_gap=1):
    """Merge flagged words into regions; resolve overlaps by keeping the strongest type."""
    n = len(part)
    out = []
    for kind, sc in scores.items():
        flag = sc > THRESH[kind]
        i = 0
        while i < n:
            if not flag[i]:
                i += 1
                continue
            j, gap = i, 0
            while j + 1 < n and (flag[j + 1] or (gap < max_gap and j + 2 < n and flag[j + 2])):
                gap = 0 if flag[j + 1] else gap + 1
                j += 1
            peak = float(sc[i:j + 1].max())
            if kind in ("long_pause", "no_pause"):
                start = part[i - 1]["end"] if i else part[i]["start"] - part[i]["pause_before"]
                end = part[j]["start"]
                if kind == "no_pause":
                    start, end = max(0.0, start - 0.25), end + 0.25
            elif kind == "stutter" and i and part[i].get("gap_voiced", 0) > 0.12:
                start, end = part[i - 1]["end"], part[j]["end"]  # include the restart before the word
            else:
                start, end = part[i]["start"], part[j]["end"]
            out.append({"type": kind, "i": i, "j": j, "start": round(start, 3), "end": round(end, 3),
                        "z": round(peak, 2)})
            i = j + 1
    # Overlap resolution: rushed/dragged spans also squeeze or stretch pauses and word lengths, so a
    # pacing region swallows pause/stutter flags it fully contains; otherwise the stronger z wins.
    out.sort(key=lambda r: -r["z"])
    kept = []
    for r in out:
        clash = [k for k in kept if r["start"] < k["end"] and k["start"] < r["end"]]
        if not clash:
            kept.append(r)
    kept.sort(key=lambda r: r["start"])
    for r in kept:
        r.update(explain(r, base, part, d))
    return kept


def _words(part, i, j, limit=6):
    w = [x["word"] for x in part[i:j + 1]]
    return " ".join(w if len(w) <= limit else w[:limit] + ["…"])


def explain(r, base, part, d):
    """Turn the numbers behind a region into a sentence a speaker can act on."""
    i, j, k = r["i"], r["j"], r["type"]
    B, P = base[i:j + 1], part[i:j + 1]
    quote = _words(part, i, j)
    if k in ("rushed", "dragged"):
        rb = sum(x["syl"] for x in B) / max(B[-1]["end"] - B[0]["start"], 0.05)
        rp = sum(x["syl"] for x in P) / max(P[-1]["end"] - P[0]["start"], 0.05)
        pct = (rp / rb - 1) * 100
        why = (f"{rp:.1f} syllables/s here vs {rb:.1f} in the reference ({pct:+.0f}%)."
               + (" Listeners lose the cadence and key words blur together." if k == "rushed"
                  else " The phrase loses momentum and sounds hesitant."))
        tip = "Slow down and let the phrase breathe." if k == "rushed" else "Tighten the phrase; keep the energy moving."
    elif k == "monotone":
        sb = np.nanmean([x["f0_spread"] for x in B]); sp = np.nanmean([x["f0_spread"] for x in P])
        why = (f"Pitch moves over {sp:.1f} semitones here vs {sb:.1f} in the reference "
               f"({(sp / sb - 1) * 100 if sb else 0:+.0f}%). Without pitch movement, emphasis disappears.")
        tip = "Lift the key word of the phrase and let the pitch fall at the end."
    elif k == "mumbled":
        dv = np.nanmean(d["volume"][i:j + 1]); dc = np.nanmean(d["clarity"][i:j + 1])
        why = (f"{abs(dv):.1f} dB quieter than the reference here (after matching your overall level), with "
               f"{abs(dc):.1f} dB less high-frequency consonant energy. Consonants are what make words intelligible.")
        tip = "Project to the back of the room and finish your consonants."
    elif k == "loud":
        dv = np.nanmean(d["volume"][i:j + 1])
        why = f"{dv:.1f} dB louder than the reference here (after matching your overall level): a sudden jump that sounds like shouting."
        tip = "Build intensity gradually instead of jumping."
    elif k == "long_pause":
        pb, pp = B[0]["pause_before"], P[0]["pause_before"]
        why = (f"A {pp:.2f} s silence before \"{P[0]['word']}\" where the reference pauses {pb:.2f} s. "
               "Mid-phrase, it reads as a lost thread rather than a dramatic pause.")
        tip = "Pause at the end of an idea, not in the middle of one."
    elif k == "no_pause":
        pb, pp = B[0]["pause_before"], P[0]["pause_before"]
        why = (f"Only {pp:.2f} s between phrases where the reference takes {pb:.2f} s. "
               "Skipping the breath runs two ideas together.")
        tip = "Take the breath: it gives the last line time to land."
    else:  # stutter
        lb, lp = B[0]["dur"], P[0]["dur"]
        gap = P[0].get("gap_voiced", 0) - B[0].get("gap_voiced", 0)
        if gap > 0.12:
            why = (f"{gap:.2f} s of voiced sound just before \"{P[0]['word']}\" where the reference has none: "
                   "the word was started, cut off and restarted.")
        else:
            why = (f"\"{P[0]['word']}\" takes {lp:.2f} s vs {lb:.2f} s in the reference ({lp / lb:.1f}×) while the "
                   "words around it keep their normal length: a drawn-out restart.")
        tip = "Slow the word's onset; plant your first syllable before you start."
    return {"label": LABEL[k], "dimension": DIMENSION[k], "quote": quote, "why": why, "tip": tip}


def rubric(part, regs, style):
    """0-10 per dimension. Each dimension loses points for the share of speaking time in its flaw regions,
    weighted by how strong the flaw is. Deterministic, so the same recording always gets the same score."""
    total = max(part[-1]["end"] - part[0]["start"], 1.0)
    dims = ["pacing", "pauses", "pitch", "volume", "fluency"]
    penalty = {k: 0.0 for k in dims}
    for r in regs:
        dur = max(r["end"] - r["start"], 0.4)
        penalty[r["dimension"]] += (dur / total) * min(r["z"], 10) * 2.0
    scores = {k: round(10 * math.exp(-penalty[k]), 1) for k in dims}
    # Overall style differences cost a little too, e.g. reading 30% flatter than the reference throughout.
    scores["pitch"] = round(scores["pitch"] * (1 - min(max(-style["pitch"], 0) / 12, 0.4)), 1)
    scores["pacing"] = round(scores["pacing"] * (1 - min(abs(style["pace"]) / 2, 0.3)), 1)
    scores["overall"] = round(float(np.mean([scores[k] for k in dims])), 1)
    return scores

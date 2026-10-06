"""End to end: two recordings of the same transcript in, flaw regions + rubric + plot data out."""
import numpy as np

from . import align, compare, features


def describe(y, transcript):
    """Align and extract features for one recording."""
    aligned = align.align(y, transcript)
    fr = features.frames(y)
    return {"aligned": aligned, "frames": fr, "words": features.words(fr, aligned)}


def warp(ref, part):
    """Map reference time onto participant time through matching word boundaries, so the reference
    contour can be drawn on the participant's timeline."""
    xs = [0.0]; ys = [0.0]
    for b, p in zip(ref["words"], part["words"]):
        for tb, tp in ((b["start"], p["start"]), (b["end"], p["end"])):
            if tb > xs[-1] and tp > ys[-1]:
                xs.append(tb); ys.append(tp)
    return np.array(xs), np.array(ys)


def analyze(ref, part):
    scores, style, d = compare.word_scores(ref["words"], part["words"])
    regs = compare.regions(ref["words"], part["words"], scores, d)
    return {"regions": regs, "style": {k: round(float(v), 3) for k, v in style.items()},
            "rubric": compare.rubric(part["words"], regs, style),
            "word_scores": {k: [round(float(x), 2) for x in v] for k, v in scores.items()}}


def plot_payload(ref, part, result, step=2):
    """Downsampled series for the dashboard: participant contours and the reference warped onto them."""
    fr, rf = part["frames"], ref["frames"]
    xs, ys = warp(ref, part)
    t_ref_on_part = np.interp(rf["t"], xs, ys)

    def clean(a):
        return [None if (x is None or np.isnan(x)) else round(float(x), 2) for x in a[::step]]

    return {
        "t": clean(fr["t"]), "f0": clean(fr["f0_st"]), "int": clean(fr["int_db"]),
        "ref_t": clean(t_ref_on_part), "ref_f0": clean(rf["f0_st"]), "ref_int": clean(rf["int_db"]),
        "words": [{k: w[k] for k in ("word", "start", "end", "rate_local", "pause_before")} for w in part["words"]],
        "ref_words": [{"start": float(np.interp(w["start"], xs, ys)), "end": float(np.interp(w["end"], xs, ys)),
                       "rate_local": w["rate_local"]}
                      for w in ref["words"]],
        **result,
    }

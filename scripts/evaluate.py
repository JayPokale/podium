"""Score the analyzer against the dataset's ground-truth labels.

A detected region matches a labelled flaw when the types agree and their temporal IoU >= 0.3
(for pauses, which are short, any overlap counts). We report per-type precision / recall / F1, mean IoU and
onset/offset error of matched regions, false alarms per minute on untouched recordings, how the rubric
falls along the L0..L4 gradient, and whether measured strength rises with injected severity.

Thresholds are tuned on the calibration split only (JFK) and then frozen for the held-out test split
(Reagan, a different speaker and speech, plus an unseen TTS voice).

  python scripts/evaluate.py            # features cache + calibrate + test, writes results/
"""
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from podium import audio, compare, pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "dataset"
RES = ROOT / "results"
CACHE = ROOT / "data" / "cache"


def described(path, transcript):
    """Alignment + features for one file, cached (the expensive part)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    c = CACHE / (str(path.relative_to(DS)).replace("/", "__") + ".pkl")
    if c.exists():
        return pickle.load(open(c, "rb"))
    d = pipeline.describe(audio.load(path), transcript)
    pickle.dump(d, open(c, "wb"))
    return d


def iou(a, b):
    inter = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    union = max(a["end"], b["end"]) - min(a["start"], b["start"])
    return inter / union if union > 0 else 0.0


def match(pred, gold):
    """Greedy one-to-one matching by IoU within the same type."""
    pairs, used = [], set()
    for g in gold:
        best, bi = 0.0, None
        for k, p in enumerate(pred):
            if k in used or p["type"] != g["type"]:
                continue
            v = iou(p, g)
            ok = v >= 0.3 or (g["type"] in ("long_pause", "no_pause") and v > 0)
            if ok and v > best:
                best, bi = v, k
        if bi is not None:
            used.add(bi)
            pairs.append((g, pred[bi], best))
    return pairs


def load_split(split):
    items = []
    for ex in sorted((DS / split).iterdir()):
        tr = (ex / "transcript.txt").read_text().strip()
        ref = described(ex / "baseline.wav", tr)
        for meta_path in sorted(ex.glob("*.json")):
            if meta_path.stem == "baseline":
                continue
            meta = json.loads(meta_path.read_text())
            items.append((ex.name, meta_path.stem, meta, ref, described(ex / f"{meta_path.stem}.wav", tr)))
    return items


def run(items, thresh):
    old = dict(compare.THRESH)
    compare.THRESH.update(thresh)
    out = []
    for ex, name, meta, ref, part in items:
        res = pipeline.analyze(ref, part)
        out.append((ex, name, meta, part, res))
    compare.THRESH.clear(); compare.THRESH.update(old)
    return out


def metrics(results):
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    ious, on_err, off_err = defaultdict(list), defaultdict(list), defaultdict(list)
    sev_pairs = defaultdict(list)
    clean_fa, clean_min = 0, 0.0
    for ex, name, meta, part, res in results:
        gold, pred = meta["flaws"], res["regions"]
        pairs = match(pred, gold)
        matched_p = {id(p) for _, p, _ in pairs}
        for g, p, v in pairs:
            tp[g["type"]] += 1
            ious[g["type"]].append(v)
            on_err[g["type"]].append(abs(p["start"] - g["start"]))
            off_err[g["type"]].append(abs(p["end"] - g["end"]))
            sev_pairs[g["type"]].append((g["severity"], p["z"]))
        for g in gold:
            if not any(g is gg for gg, _, _ in pairs):
                fn[g["type"]] += 1
        for p in pred:
            if id(p) not in matched_p:
                fp[p["type"]] += 1
        if not gold:
            clean_fa += len(pred)
            clean_min += (part["words"][-1]["end"] - part["words"][0]["start"]) / 60
    table = {}
    for t in compare.THRESH:
        P = tp[t] / (tp[t] + fp[t]) if tp[t] + fp[t] else float("nan")
        R = tp[t] / (tp[t] + fn[t]) if tp[t] + fn[t] else float("nan")
        F = 2 * P * R / (P + R) if P + R and not np.isnan(P + R) else 0.0
        sp = spearmanr(*zip(*sev_pairs[t])).statistic if len({s for s, _ in sev_pairs[t]}) > 1 else float("nan")
        table[t] = {"tp": tp[t], "fp": fp[t], "fn": fn[t], "precision": P, "recall": R, "f1": F,
                    "mean_iou": float(np.mean(ious[t])) if ious[t] else float("nan"),
                    "onset_err_s": float(np.median(on_err[t])) if on_err[t] else float("nan"),
                    "offset_err_s": float(np.median(off_err[t])) if off_err[t] else float("nan"),
                    "severity_spearman": sp}
    T = {k: sum(d[k] for d in (tp, fp, fn) for _ in [0]) for k in ()}
    allp = sum(tp.values()) / max(sum(tp.values()) + sum(fp.values()), 1)
    allr = sum(tp.values()) / max(sum(tp.values()) + sum(fn.values()), 1)
    table["ALL"] = {"tp": sum(tp.values()), "fp": sum(fp.values()), "fn": sum(fn.values()),
                    "precision": allp, "recall": allr, "f1": 2 * allp * allr / max(allp + allr, 1e-9),
                    "mean_iou": float(np.mean(sum(ious.values(), []))) if any(ious.values()) else float("nan"),
                    "false_alarms_per_min_clean": clean_fa / max(clean_min, 1e-9)}
    return table


def confusion(results):
    """For each labelled flaw, which detected type overlaps it most (or 'missed')."""
    types = list(compare.THRESH)
    m = {g: {p: 0 for p in types + ["missed"]} for g in types}
    for ex, name, meta, part, res in results:
        for g in meta["flaws"]:
            best, bt = 0.0, "missed"
            for p in res["regions"]:
                v = iou(p, g)
                if v > best:
                    best, bt = v, p["type"]
            m[g["type"]][bt] += 1
    return m


def fmt_conf(m):
    types = list(m)
    cols = types + ["missed"]
    short = {"long_pause": "l_pause", "no_pause": "no_pause"}
    head = "| injected \\ detected | " + " | ".join(short.get(c, c) for c in cols) + " |"
    lines = [head, "|" + "---|" * (len(cols) + 1)]
    for g in types:
        lines.append(f"| {g} | " + " | ".join(str(m[g][c]) for c in cols) + " |")
    return "\n".join(lines)


def gradient(results):
    """Rubric overall score along L0..L4 for each excerpt and source (mirror / each TTS voice)."""
    rows = defaultdict(dict)
    for ex, name, meta, part, res in results:
        if meta["kind"] in ("gradient", "tts"):
            lvl = int(name.rsplit("_L", 1)[1])
            src = "mirror" if meta["kind"] == "gradient" else meta["voice"]
            rows[(ex, src)][lvl] = res["rubric"]["overall"]
    rhos = []
    for key, d in rows.items():
        lv = sorted(d)
        if len(lv) > 2:
            rhos.append(spearmanr(lv, [d[l] for l in lv]).statistic)
    return {f"{k[0]}|{k[1]}": v for k, v in rows.items()}, float(np.nanmean(rhos)) if rhos else float("nan")


def calibrate(items):
    """Per-type threshold maximising F1 on the calibration split, one type at a time."""
    best = dict(compare.THRESH)
    for t in compare.THRESH:
        scores = []
        for th in (2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0):
            m = metrics(run(items, {**best, t: th}))[t]
            scores.append((m["f1"], -abs(th - 3.0), th))
        best[t] = max(scores)[2]
        print(f"  {t:11s} -> {best[t]}  (F1 {max(scores)[0]:.2f})")
    return best


def fmt(table):
    lines = ["| type | P | R | F1 | IoU | onset err (s) | offset err (s) | severity ρ |", "|---|---|---|---|---|---|---|---|"]
    for t, m in table.items():
        if t == "ALL":
            continue
        f = lambda x: "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"
        lines.append(f"| {t} | {f(m['precision'])} | {f(m['recall'])} | {f(m['f1'])} | {f(m['mean_iou'])} | "
                     f"{f(m['onset_err_s'])} | {f(m['offset_err_s'])} | {f(m['severity_spearman'])} |")
    a = table["ALL"]
    lines.append(f"| **all** | **{a['precision']:.2f}** | **{a['recall']:.2f}** | **{a['f1']:.2f}** | {a['mean_iou']:.2f} | | | |")
    lines.append(f"\nFalse alarms on untouched recordings: {a['false_alarms_per_min_clean']:.2f} per minute.")
    return "\n".join(lines)


def main():
    RES.mkdir(exist_ok=True)
    cal, test = load_split("calibration"), load_split("test")
    print("calibrating on", len(cal), "files")
    th = calibrate(cal)
    out = {"thresholds": th}
    for name, items in (("calibration", cal), ("test", test)):
        r = run(items, th)
        table = metrics(r)
        grad, rho = gradient(r)
        conf = confusion(r)
        out[name] = {"metrics": table, "gradient": grad, "gradient_spearman": rho, "confusion": conf}
        # split mirror vs different-speaker (TTS) on test
        if name == "test":
            out["test_by_source"] = {
                "same speaker (mirror + stress)": metrics([x for x in r if x[2]["kind"] != "tts"]),
                "different speaker (TTS)": metrics([x for x in r if x[2]["kind"] == "tts"]),
            }
        print(f"\n## {name}\n" + fmt(table) + f"\nRubric vs gradient level, mean Spearman ρ: {rho:.2f}\n\n" + fmt_conf(conf))
    if "test_by_source" in out:
        for k, t in out["test_by_source"].items():
            a = t["ALL"]
            print(f"test, {k}: P {a['precision']:.2f} R {a['recall']:.2f} F1 {a['f1']:.2f} "
                  f"false alarms/min on clean {a['false_alarms_per_min_clean']:.2f}")
    json.dump(out, open(RES / "evaluation.json", "w"), indent=1, default=float)
    json.dump(th, open(RES / "thresholds.json", "w"), indent=1)
    (RES / "evaluation.md").write_text("\n\n".join(
        [f"## {n}\n\n{fmt(out[n]['metrics'])}\n\nRubric vs gradient level (L0→L4), mean Spearman ρ: "
         f"{out[n]['gradient_spearman']:.2f}\n\nConfusion (best-overlapping detection per injected flaw):\n\n{fmt_conf(out[n]['confusion'])}"
         for n in ("calibration", "test")]
        + ["## Test by source\n\n" + "\n\n".join(f"### {k}\n\n{fmt(t)}" for k, t in out["test_by_source"].items())]
        + ["## Frozen thresholds\n\n" + json.dumps(th)]))


if __name__ == "__main__":
    main()

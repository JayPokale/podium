"""Acoustic features, frame level and word level, normalised per speaker.

Frame level (10 ms hop):
  f0_st      pitch in semitones relative to the speaker's median F0 (NaN when unvoiced)
  int_db     intensity in dB relative to the speaker's median voiced intensity
  hnr        harmonics-to-noise ratio (dB), a voice-clarity measure
  centroid   spectral centroid (Hz), which drops when speech is muffled
  hf_db      energy above 2.5 kHz relative to total (dB), consonant crispness

Word level (one row per aligned word): duration, syllables per second, pause before the word, mean
pitch, pitch spread over the surrounding phrase, mean intensity, clarity and the aligner's confidence.

Relative units (semitones, dB re median) make the comparison speaker-agnostic: a deep voice and a high
voice reading with the same expressiveness produce the same contour.
"""
import librosa
import numpy as np
import parselmouth

from .audio import SR, syllables

HOP = 0.01


def frames(y, sr=SR, f0_floor=70, f0_ceiling=400):
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
    n = int(len(y) / sr / HOP)
    t = np.arange(n) * HOP + HOP / 2

    def on_grid(xs, vals, fill=np.nan):
        return np.interp(t, xs, vals, left=fill, right=fill) if len(xs) else np.full(n, fill)

    pitch = snd.to_pitch_ac(time_step=HOP, pitch_floor=f0_floor, pitch_ceiling=f0_ceiling)
    f0_raw = pitch.selected_array["frequency"].astype(float)
    f0_raw[f0_raw <= 0] = np.nan
    # nearest frame, not interpolation, so voiced/unvoiced edges stay sharp
    idx = np.clip(np.searchsorted(pitch.xs(), t), 0, len(f0_raw) - 1)
    f0 = f0_raw[idx] if len(f0_raw) else np.full(n, np.nan)
    inten = snd.to_intensity(minimum_pitch=f0_floor, time_step=HOP)
    db = on_grid(inten.xs(), inten.values[0], fill=float(np.min(inten.values[0])))
    hnr_obj = snd.to_harmonicity_cc(time_step=HOP, minimum_pitch=f0_floor)
    hv = hnr_obj.values[0].astype(float)
    hv[hv < -50] = np.nan  # Praat marks silence as -200 dB
    hnr = on_grid(hnr_obj.xs(), hv)

    hop = int(HOP * sr)
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    S = S[:, :n] if S.shape[1] >= n else np.pad(S, ((0, 0), (0, n - S.shape[1])))
    tot = S.sum(0) + 1e-12
    centroid = (freqs[:, None] * S).sum(0) / tot
    hf_db = 10 * np.log10(S[freqs > 2500].sum(0) / tot + 1e-12)

    voiced = ~np.isnan(f0)
    med_f0 = np.nanmedian(f0) if voiced.any() else 120.0
    loud = db > np.nanpercentile(db, 30)
    med_db = np.nanmedian(db[voiced & loud]) if (voiced & loud).any() else np.nanmedian(db)
    return {
        "t": t,
        "f0_hz": f0,
        "f0_st": 12 * np.log2(f0 / med_f0),
        "int_db": db - med_db,
        "hnr": hnr,
        "centroid": centroid,
        "hf_db": hf_db,
        "speaker": {"median_f0_hz": float(med_f0), "median_db": float(med_db)},
    }


def _span(fr, a, b):
    i, j = int(a / HOP), max(int(a / HOP) + 1, int(b / HOP))
    return slice(i, min(j, len(fr["t"])))


def _nanmean(x):
    x = x[~np.isnan(x)]
    return float(x.mean()) if len(x) else float("nan")


def words(fr, aligned, phrase=5):
    """Per-word feature rows. `phrase` words either side define the window for pitch spread."""
    rows = []
    for i, w in enumerate(aligned):
        s = _span(fr, w["start"], w["end"])
        prev_end = aligned[i - 1]["end"] if i else w["start"]
        dur = max(w["end"] - w["start"], 0.02)
        speech = fr["int_db"][s] > -25  # ignore near-silent frames inside the word
        rows.append({
            "i": i, "word": w["word"], "start": w["start"], "end": w["end"], "score": w["score"],
            "dur": dur,
            "syl": syllables(w["word"]),
            "rate": syllables(w["word"]) / dur,
            "pause_before": max(0.0, w["start"] - prev_end),
            # seconds of voiced sound inside the gap before the word: ~0 for a real pause, high when the
            # gap holds a restart ("w- w- we") that the aligner left outside the word
            "gap_voiced": float(np.sum(~np.isnan(fr["f0_st"][_span(fr, prev_end, w["start"])])) * HOP)
            if w["start"] - prev_end > 0.05 else 0.0,
            "f0": _nanmean(fr["f0_st"][s]),
            "int": _nanmean(np.where(speech, fr["int_db"][s], np.nan)),
            "hnr": _nanmean(fr["hnr"][s]),
            "centroid": _nanmean(np.where(speech, fr["centroid"][s], np.nan)),
            "hf": _nanmean(np.where(speech, fr["hf_db"][s], np.nan)),
        })
    # Pitch spread of the phrase around each word: the 10th-90th percentile range of the contour.
    for i, r in enumerate(rows):
        a = rows[max(0, i - phrase)]["start"]
        b = rows[min(len(rows) - 1, i + phrase)]["end"]
        c = fr["f0_st"][_span(fr, a, b)]
        c = c[~np.isnan(c)]
        r["f0_spread"] = float(np.percentile(c, 90) - np.percentile(c, 10)) if len(c) > 10 else float("nan")
        # Local speech rate over a 5-word window: syllables over time from first onset to last offset.
        lo, hi = max(0, i - 2), min(len(rows) - 1, i + 2)
        r["rate_local"] = sum(x["syl"] for x in rows[lo:hi + 1]) / max(rows[hi]["end"] - rows[lo]["start"], 0.05)
    return rows

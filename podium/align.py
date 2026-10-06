"""Forced alignment of a known transcript to audio (torchaudio MMS_FA, a wav2vec2 CTC aligner).

Every word gets a start and end time and a confidence. Because the transcript is fixed, word i of a
participant recording corresponds to word i of the baseline, which is what lets us compare deliveries
of the same text word by word.
"""
import os
from functools import lru_cache

import numpy as np
import torch
import torchaudio

from .audio import SR, words_of

DEVICE = os.environ.get("PODIUM_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def _bundle():
    b = torchaudio.pipelines.MMS_FA
    return b, b.get_model(with_star=False).to(DEVICE).eval(), b.get_tokenizer(), b.get_aligner()


def align(y, transcript):
    """Return a list of {"word", "start", "end", "score"} (seconds) for the transcript's words."""
    bundle, model, tokenizer, aligner = _bundle()
    words = words_of(transcript)
    wav = torch.from_numpy(np.ascontiguousarray(y)).unsqueeze(0)
    with torch.inference_mode():
        try:
            emission, _ = model(wav.to(DEVICE))
        except torch.OutOfMemoryError:  # GPU busy (or a long recording): fall back to CPU for this call
            torch.cuda.empty_cache()
            emission, _ = model.to("cpu")(wav)
            model.to(DEVICE)
        spans = aligner(emission[0].cpu(), tokenizer(words))
    ratio = wav.size(1) / emission.size(1) / SR
    out = []
    for w, sp in zip(words, spans):
        out.append({
            "word": w,
            "start": round(sp[0].start * ratio, 3),
            "end": round(sp[-1].end * ratio, 3),
            "score": round(float(np.mean([s.score for s in sp])), 3),
        })
    return out

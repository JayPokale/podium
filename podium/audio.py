"""Audio loading and text normalisation shared by every stage."""
import re

import librosa
import numpy as np
import soundfile as sf

SR = 16000


def load(path, sr=SR, start=None, end=None):
    """Mono float32 at `sr`, optionally cut to [start, end) seconds."""
    y, _ = librosa.load(path, sr=sr, mono=True, offset=start or 0.0,
                        duration=None if end is None else end - (start or 0.0))
    return y.astype(np.float32)


def save(path, y, sr=SR):
    sf.write(path, np.clip(y, -1, 1), sr, subtype="PCM_16")


def words_of(text):
    """Transcript -> list of words as the aligner sees them (lower case, letters and apostrophes)."""
    text = text.lower().replace("’", "'")
    text = re.sub(r"(\d+)", lambda m: _num(m.group(1)), text)
    return [w for w in (re.sub(r"[^a-z']", "", t).strip("'") for t in text.split()) if w]


_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen " \
        "sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def _num(s):
    n = int(s)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + _num(str(n % 100)))
    if 1900 <= n < 2100 and n % 100:  # years read as "nineteen sixty one"
        return _num(str(n // 100)) + " " + _num(str(n % 100))
    return " ".join(_ONES[int(d)] for d in s)


def syllables(word):
    """Vowel-group syllable estimate; good enough for a rate measure compared against the same words."""
    groups = re.findall(r"[aeiouy]+", word)
    n = len(groups)
    if word.endswith("e") and n > 1 and not word.endswith(("le", "ee")):
        n -= 1
    return max(1, n)

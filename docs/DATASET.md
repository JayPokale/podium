# The Podium contrastive speech dataset

**207 recordings, 6 "ideal" baselines, every flaw labelled to the millisecond.**

There is no public dataset that pairs a good delivery with a spectrum of bad deliveries of the *same text*.
Podium builds one, and because it injects the flaws itself, the labels are exact rather than a human's guess
of where a "rushed" phrase starts.

## 1. The "good" baselines

Three speeches, all in the public domain (works of the US federal government, from Wikimedia Commons):

| Excerpt | Speech | Speaker | Source time (s) | Words | Split |
|---|---|---|---|---|---|
| `jfk_inaugural_2` | Inaugural Address, 1961 ("Finally, whether you are citizens…") | John F. Kennedy | 791.3–833.9 | 74 | calibration |
| `jfk_moon_1` | Rice University "We choose to go to the Moon", 1962 | John F. Kennedy | 47.6–99.6 | 105 | calibration |
| `jfk_moon_2` | same ("In the last 24 hours…") | John F. Kennedy | 576.7–643.3 | 143 | calibration |
| `reagan_challenger_1` | Address on the Challenger disaster, 1986 | Ronald Reagan | 0.3–48.7 | 139 | **test** |
| `reagan_challenger_2` | same ("The families of the seven…") | Ronald Reagan | 63.0–147.9 | 229 | **test** |
| `reagan_challenger_3` | same (the Sir Francis Drake passage to the end) | Ronald Reagan | 183.5–250.8 | 172 | **test** |

How the excerpts were cut:

1. A first Whisper pass over each full speech located applause and crowd noise. Long-form transcription
   hallucinates during applause (we saw "Bye-bye" repeated for 20 seconds), so only applause-free windows
   were kept.
2. Each window was transcribed on its own with **Whisper large-v3-turbo** (open weights).
3. The transcript was **force-aligned** with torchaudio's **MMS_FA** (wav2vec2 CTC) aligner, giving every word a
   start, end and confidence.
4. The excerpt was cut to the longest run of whole sentences (40–85 s) with no gap over 2.5 s.

Words the aligner places with low confidence (numbers read aloud such as "three hundred ninety", names) are
kept but marked; the analyzer does not let them raise a flaw on their own.

## 2. The "bad" spectrum

Eight delivery flaws, each at three severities, injected into chosen word spans:

| Flaw | How it is made | Severity 1 / 2 / 3 |
|---|---|---|
| `rushed` | time-compression of a 5–9 word span (phase vocoder) | ×1.3 / ×1.6 / ×2.0 speed |
| `dragged` | time-stretch of a 4–7 word span | ×0.8 / ×0.65 / ×0.5 speed |
| `monotone` | pitch movement flattened toward the phrase mean with the WORLD vocoder, loudness matched | 50% / 75% / 95% of movement removed |
| `mumbled` | gain down + 6th-order low-pass | −6 dB @ 3 kHz / −11 dB @ 1.8 kHz / −16 dB @ 1.1 kHz |
| `loud` | gain up with soft clipping (tanh) | +4 / +7 / +10 dB |
| `long_pause` | room-tone silence inserted mid-phrase (where the speaker did not pause) | 0.7 / 1.3 / 2.2 s |
| `no_pause` | a natural breath pause (> 0.35 s) between phrases squeezed out | 50% / 20% / 0% kept |
| `stutter` | the onset of a word repeated with short gaps ("w- w- we") | 1 / 2 / 3 repeats |

Edits are joined with 5 ms fades and never change segment lengths, so label times are sample-accurate.
Inserted silence uses noise at the recording's own noise floor, so a pause doesn't sound like a digital cut.

For every baseline the dataset contains:

- **Gradient L0–L4** (5 files): from untouched (L0) through "almost perfect" (L1: mild rushing and a slightly
  flat phrase) to egregious (L4: seven severe flaws). This is the "botched to almost perfect" spectrum.
- **Stress tests** (24 files): one flaw per file, every type × severity, for clean per-type measurement.
- **Different speaker** (3 or 6 files): the same transcript read by open **Piper** TTS voices
  (`en_US-lessac`, `en_GB-alan` for calibration; `en_US-ryan` held out for testing), at L0, L2 and L4.
  These test that the comparison is speaker-agnostic: the voice, pitch level and natural rhythm differ from
  JFK or Reagan everywhere, and only the injected flaws are labelled.

## 3. Labels

Each `*.wav` has a `*.json` next to it:

```json
{
  "excerpt": "reagan_challenger_1", "split": "test", "kind": "gradient", "voice": null,
  "reference": "baseline.wav",
  "flaws": [
    {"type": "rushed", "severity": 2, "i": 12, "j": 18, "start": 3.762, "end": 5.288}
  ],
  "words": [{"word": "ladies", "start": 0.62, "end": 0.9, "score": 0.767}, "..."]
}
```

`i`/`j` are word indices in `transcript.txt`; `start`/`end` are seconds in *this* file; `words` gives every
word's time in this file (mapped through the edits), which also lets you check an aligner.

## 4. Splits

Thresholds are tuned only on **calibration** (JFK, two Piper voices). **Test** is a different speaker and
speech (Reagan) and an unseen voice, so reported test numbers measure generalisation, not memorisation.

Rebuild everything from the raw public-domain audio with `python scripts/build_dataset.py`
(Whisper and the seeds are deterministic; the random spans use `zlib.crc32` of the excerpt name).

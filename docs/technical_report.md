# Podium: Contrastive Speech Analytics with Temporal Flaw Grounding

*Technical report, Multimodal AI Hackathon 2026, Track C.* Jay Pokale. Code: github.com/JayPokale/podium

## Abstract

Judging spoken delivery is subjective because there is no shared reference: two judges hear the same rushed
phrase and disagree on where it started or how bad it was. Podium makes delivery measurable by comparing a
participant with a reference delivery *of the same text*. Both recordings are force-aligned to the transcript,
so every word has a counterpart. Speaker-normalised acoustic features are then compared word by word, and flaws
are reported as time ranges with a causal explanation and a deterministic 0-10 rubric. To evaluate this, we
built a contrastive dataset of 207 recordings: six public-domain "ideal" baselines (JFK, Reagan) mirrored by
programmatically injected flaws (8 types × 3 severities) with sample-accurate labels, plus different-speaker
readings by open TTS voices. On a held-out speaker and speech, Podium localises flaws with mean IoU 0.81 and an
overall F1 of 0.51. Pauses, mumbling and stutters are located to within 0.06 s, and the rubric decreases
monotonically with flaw severity (Spearman ρ = −1.00).

## 1. Dataset construction

**Baselines.** Three US public-domain speeches from Wikimedia Commons: Kennedy's 1961 inaugural address and
1962 Rice University address, and Reagan's 1986 Challenger address. A first pass with Whisper large-v3-turbo
located applause; long-form transcription hallucinated over it (a 20-second "Bye-bye" loop), so we kept only
applause-free windows. Each window was transcribed separately, force-aligned with torchaudio's MMS_FA
(wav2vec2 CTC) aligner, and cut to the longest run of whole sentences of 40-85 s with no gap over 2.5 s.
This gives 6 excerpts (42-85 s, 74-229 words).

**Bad mirror.** Because we inject flaws into the baseline audio, labels are exact. For a span of words
[i, j], the injector applies:

| Flaw | Operation | Severity 1 / 2 / 3 |
|---|---|---|
| rushed / dragged | phase-vocoder time-scale | ×1.3/1.6/2.0 · ×0.8/0.65/0.5 |
| monotone | WORLD analysis, F0 contour compressed toward its mean in semitones, resynthesis, RMS matched | 50/75/95% movement removed |
| mumbled | gain + 6th-order Butterworth low-pass | −6 dB@3 kHz / −11@1.8k / −16@1.1k |
| loud | gain with tanh soft clipping | +4 / +7 / +10 dB |
| long_pause | room tone at the recording's 5th-percentile RMS, inserted where the speaker did not pause (<0.12 s) | 0.7 / 1.3 / 2.2 s |
| no_pause | a natural pause (>0.35 s) shortened | 50 / 20 / 0% kept |
| stutter | the word onset (≤160 ms) repeated with 70 ms gaps | 1 / 2 / 3 repeats |

Segments are joined with 5 ms fades that preserve length, so label times stay sample-accurate. Every word's
time is mapped through the edits with a piecewise-linear time map. Per baseline we generate a gradient
L0-L4 (from untouched through "almost perfect" to seven severe flaws) and 24 single-flaw stress tests.
For speaker agnosticism, open Piper TTS voices read the same transcript (clean, L2, L4).

**Split.** Calibration: the 3 JFK excerpts and 2 voices (105 files). Test: the 3 Reagan excerpts and an unseen
voice (102 files). Thresholds are tuned on calibration only.

## 2. Alignment and feature extraction

Audio is resampled to 16 kHz mono. MMS_FA yields word start/end times and a confidence score $c_w$.
Frame features use a 10 ms hop:

- **Pitch**: Praat autocorrelation F0 (70-400 Hz), as semitones relative to the speaker's median:
  $f_{st}(t) = 12\log_2(F_0(t)/\tilde F_0)$.
- **Intensity**: Praat intensity in dB, minus the speaker's median over voiced, non-quiet frames.
- **Clarity**: harmonics-to-noise ratio, spectral centroid, and high-frequency energy
  $HF(t) = 10\log_{10}(\sum_{f>2.5\,kHz}|X|^2 / \sum_f |X|^2)$, which drops when consonants are swallowed.

Per word *w* we compute duration, syllables (vowel groups), the pause before it, mean $f_{st}$, mean
intensity and HF energy over speech frames, the 10th-90th percentile pitch spread over a ±5-word phrase
window, the **articulation rate** over a 5-word window $A_i = \sum syl / \sum dur$ (pauses excluded), and
**voiced-gap time**: seconds of voiced frames inside the gap before the word. That last one is ~0 for a real
pause and large when a restart ("w- w- we") sits outside the aligned word.

Relative units (semitones, dB re median, rates) make the comparison speaker-agnostic: a deep and a high voice
with equally expressive intonation produce the same contour.

## 3. Temporal grounding

**Deltas and global style.** For each word, deltas between participant and reference: pace
$\log_2(A^{ref}_i/A^{p}_i)$, pause difference, pitch-spread difference, intensity and HF differences, and
log duration ratio. The median of each delta over the whole recording is the speaker's **overall style**
(reported separately, e.g. "33% shorter words"). Local evidence is a robust z-score around it:
$z^{ref}_i = (\delta_i - \mathrm{med}\,\delta) / \max(1.4826\cdot\mathrm{MAD}(\delta), \epsilon)$, with floors
$\epsilon$ (0.08 log₂ for pace, 1 st for pitch, 1.5 dB for loudness) so a near-identical recording doesn't turn
jitter into huge scores.

**Self evidence.** The same statistic computed within the participant's own recording, $z^{self}_i$: is this
phrase faster, flatter, quieter or more paused than *this speaker* usually is? The flaw score is a soft AND,

$$ s^{k}_i = \min(z^{ref,k}_i,\; z^{self,k}_i), $$

so a consistent difference from the reference (a different voice, a slower reader) moves $z^{ref}$ but not
$z^{self}$ and is not flagged. On the different-speaker test set, this cut false alarms on clean recordings
from 41 to 3 per minute.

**Type-specific rules.** A *long pause* must be silent (voiced gap < 0.15 s); a gap full of voiced sound is a
*stutter*. A *missing breath* is only scored where the reference pauses > 0.3 s. A *stutter* is either a
voiced gap above 0.12 s, or a single word stretched > 0.15 s while its neighbours are not. Words whose
reference alignment is unreliable ($c_w$ < 0.1: numbers read aloud, names) cannot raise flags.

**Regions.** Words with $s^k_i > \theta_k$ are merged into regions (allowing a one-word gap). Pause regions span
the gap between words; others span first-word onset to last-word offset. Overlapping regions of different types
keep the stronger z. Thresholds $\theta_k$ maximise per-type F1 on calibration over {2, 2.5, …, 6}.

## 4. Causal explanation and rubric

Each region is turned into a sentence from its own numbers. For example, *"14.3 syllables/s here vs 7.7 in the
reference (+85%). Listeners lose the cadence"*, or *"Pitch moves over 1.1 semitones here vs 6.9 in the reference
(−84%)"*, followed by a tip. The rubric gives each dimension (pacing, pauses, pitch, volume, fluency)
$10\cdot e^{-P}$, where $P = \sum_{r} 2\,\frac{\max(d_r, 0.4)}{T}\min(z_r, 10)$ over that dimension's regions
(duration $d_r$, recording length $T$), with a small deduction for overall style. It is deterministic: the
same recording always gets the same score.

## 5. Evaluation

A detection matches a label if the types agree and temporal IoU ≥ 0.3 (any overlap for pauses). Test split:

| Flaw | P | R | F1 | IoU | onset err (s) | severity ρ |
|---|---|---|---|---|---|---|
| rushed | 0.32 | 0.26 | 0.29 | 0.68 | 0.20 | 0.72 |
| dragged | 0.32 | 0.40 | 0.35 | 0.63 | 0.45 | 0.62 |
| monotone | 0.69 | 0.43 | 0.53 | 0.67 | 0.49 | 0.43 |
| mumbled | 0.83 | 0.71 | 0.77 | 0.90 | 0.02 | 0.42 |
| loud | 0.47 | 0.53 | 0.50 | 0.60 | 0.07 | 0.38 |
| long_pause | 0.74 | 0.71 | 0.72 | 0.99 | 0.00 | 0.95 |
| no_pause | 0.50 | 0.25 | 0.33 | 0.95 | 0.02 | 0.00 |
| stutter | 0.57 | 0.44 | 0.50 | 0.84 | 0.06 | 0.76 |
| **all** | **0.55** | **0.48** | **0.51** | **0.81** | | |

By source: same speaker P 0.70 / R 0.54 / F1 0.61 with **0 false alarms per minute** on untouched audio;
different speaker (TTS) P 0.18 / R 0.23 / F1 0.21 with 3.1 false alarms per minute. Across all L0-L4 gradients
the rubric is perfectly monotone (mean Spearman ρ = −1.00 on test, −0.92 on calibration). Calibration F1 is 0.62,
close to test, so the frozen thresholds generalise to a new speaker.

**Error analysis.** By best-overlapping detection per injected flaw, 75% of test flaws (114 of 153) are localised by *some*
detector. Most errors are type confusions: 5 of 15 dragged phrases are reported as awkward pauses, because slowing
a phrase also stretches its internal pauses. 9 of 12 removed breaths are missed: once the pause is gone, the
aligner often splits the surrounding words differently. Monotone recall is limited by our 1-semitone pitch-spread
floor: the speeches are already fairly flat in places, so a 50% flattening is sometimes below natural variation.

## 6. Limitations and next steps

- **Synthetic flaws are cleaner than human ones.** Real rushing changes articulation, not just time scale.
  The next step is self-recorded flawed readings (the dashboard records in the browser) with human labels.
- **Different speakers remain hard.** A reader's rhythm differs from the reference everywhere. Learning
  per-feature noise models from paired readings, instead of fixed floors, should raise the 0.21 F1.
- **One reference is one style.** Multiple references per text would let "ideal" be a range, not a point.
- Pacing would benefit from a syllable nucleus detector instead of vowel-group counting.

## 7. Reproducibility

`python scripts/build_dataset.py` rebuilds the dataset from the raw public-domain audio (seeds derived from
`crc32` of excerpt names). `python scripts/evaluate.py` recalibrates and writes `results/`. The dashboard
(`uvicorn app.server:app`) runs fully locally; alignment uses a GPU if available and falls back to CPU.
Analysing a 50-s recording takes about 10 s on a laptop GTX 1660 Ti.

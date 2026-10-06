## calibration

| type | P | R | F1 | IoU | onset err (s) | offset err (s) | severity ρ |
|---|---|---|---|---|---|---|---|
| rushed | 0.40 | 0.24 | 0.30 | 0.62 | 0.36 | 0.10 | 0.09 |
| dragged | 0.45 | 0.28 | 0.34 | 0.44 | 1.20 | 1.14 | -0.89 |
| monotone | 0.55 | 0.25 | 0.34 | 0.55 | 0.85 | 1.42 | 0.78 |
| mumbled | 0.96 | 0.93 | 0.94 | 0.91 | 0.02 | 0.02 | 0.37 |
| loud | 0.54 | 0.39 | 0.45 | 0.88 | 0.02 | 0.01 | 0.20 |
| long_pause | 0.91 | 0.97 | 0.94 | 0.98 | 0.00 | 0.00 | 0.93 |
| no_pause | 0.69 | 0.92 | 0.79 | 0.96 | 0.02 | 0.01 | 0.81 |
| stutter | 0.54 | 0.67 | 0.60 | 0.89 | 0.06 | 0.01 | 0.63 |
| **all** | **0.68** | **0.57** | **0.62** | 0.87 | | | |

False alarms on untouched recordings: 0.00 per minute.

Rubric vs gradient level (L0→L4), mean Spearman ρ: -0.92

Confusion (best-overlapping detection per injected flaw):

| injected \ detected | rushed | dragged | monotone | mumbled | loud | l_pause | no_pause | stutter | missed |
|---|---|---|---|---|---|---|---|---|---|
| rushed | 16 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 13 |
| dragged | 0 | 5 | 0 | 0 | 0 | 3 | 0 | 5 | 5 |
| monotone | 0 | 0 | 11 | 0 | 0 | 0 | 0 | 2 | 11 |
| mumbled | 0 | 0 | 0 | 26 | 0 | 0 | 0 | 0 | 1 |
| loud | 0 | 0 | 0 | 0 | 12 | 0 | 0 | 0 | 6 |
| long_pause | 0 | 1 | 0 | 0 | 0 | 29 | 0 | 0 | 0 |
| no_pause | 0 | 0 | 0 | 0 | 0 | 0 | 11 | 0 | 1 |
| stutter | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 14 | 5 |

## test

| type | P | R | F1 | IoU | onset err (s) | offset err (s) | severity ρ |
|---|---|---|---|---|---|---|---|
| rushed | 0.32 | 0.26 | 0.29 | 0.68 | 0.20 | 0.04 | 0.72 |
| dragged | 0.32 | 0.40 | 0.35 | 0.63 | 0.45 | 0.55 | 0.62 |
| monotone | 0.69 | 0.43 | 0.53 | 0.67 | 0.49 | 0.50 | 0.43 |
| mumbled | 0.83 | 0.71 | 0.77 | 0.90 | 0.02 | 0.00 | 0.42 |
| loud | 0.47 | 0.53 | 0.50 | 0.60 | 0.07 | 0.10 | 0.38 |
| long_pause | 0.74 | 0.71 | 0.72 | 0.99 | 0.00 | 0.00 | 0.95 |
| no_pause | 0.50 | 0.25 | 0.33 | 0.95 | 0.02 | 0.01 | 0.00 |
| stutter | 0.57 | 0.44 | 0.50 | 0.84 | 0.06 | 0.01 | 0.76 |
| **all** | **0.55** | **0.48** | **0.51** | 0.81 | | | |

False alarms on untouched recordings: 0.87 per minute.

Rubric vs gradient level (L0→L4), mean Spearman ρ: -1.00

Confusion (best-overlapping detection per injected flaw):

| injected \ detected | rushed | dragged | monotone | mumbled | loud | l_pause | no_pause | stutter | missed |
|---|---|---|---|---|---|---|---|---|---|
| rushed | 13 | 4 | 0 | 0 | 0 | 0 | 3 | 1 | 6 |
| dragged | 0 | 8 | 0 | 0 | 0 | 5 | 0 | 1 | 1 |
| monotone | 0 | 1 | 12 | 0 | 0 | 0 | 0 | 1 | 7 |
| mumbled | 0 | 2 | 0 | 16 | 0 | 0 | 0 | 1 | 2 |
| loud | 0 | 0 | 0 | 0 | 11 | 0 | 0 | 1 | 3 |
| long_pause | 1 | 5 | 0 | 0 | 0 | 17 | 0 | 0 | 1 |
| no_pause | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 9 |
| stutter | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 10 |

## Test by source

### same speaker (mirror + stress)

| type | P | R | F1 | IoU | onset err (s) | offset err (s) | severity ρ |
|---|---|---|---|---|---|---|---|
| rushed | 0.47 | 0.33 | 0.39 | 0.68 | 0.20 | 0.04 | 0.72 |
| dragged | 0.67 | 0.50 | 0.57 | 0.63 | 0.45 | 0.55 | 0.62 |
| monotone | 0.80 | 0.44 | 0.57 | 0.70 | 0.37 | 0.52 | 0.44 |
| mumbled | 0.86 | 0.80 | 0.83 | 0.93 | 0.02 | 0.00 | 0.51 |
| loud | 0.70 | 0.58 | 0.64 | 0.60 | 0.12 | 0.00 | 0.47 |
| long_pause | 0.73 | 0.89 | 0.80 | 0.99 | 0.00 | 0.00 | 0.95 |
| no_pause | 0.50 | 0.25 | 0.33 | 0.95 | 0.02 | 0.01 | 0.00 |
| stutter | 0.88 | 0.47 | 0.61 | 0.83 | 0.06 | 0.00 | 0.79 |
| **all** | **0.70** | **0.54** | **0.61** | 0.82 | | | |

False alarms on untouched recordings: 0.00 per minute.

### different speaker (TTS)

| type | P | R | F1 | IoU | onset err (s) | offset err (s) | severity ρ |
|---|---|---|---|---|---|---|---|
| rushed | 0.00 | 0.00 | 0.00 | – | – | – | – |
| dragged | 0.00 | 0.00 | 0.00 | – | – | – | – |
| monotone | 0.33 | 0.33 | 0.33 | 0.41 | 0.82 | 0.14 | – |
| mumbled | 0.75 | 0.50 | 0.60 | 0.78 | 0.04 | 0.02 | 0.00 |
| loud | 0.14 | 0.33 | 0.20 | 0.60 | 0.01 | 0.21 | – |
| long_pause | 1.00 | 0.17 | 0.29 | 0.94 | 0.10 | 0.04 | – |
| no_pause | – | – | 0.00 | – | – | – | – |
| stutter | 0.17 | 0.33 | 0.22 | 0.97 | 0.02 | 0.01 | – |
| **all** | **0.18** | **0.23** | **0.21** | 0.75 | | | |

False alarms on untouched recordings: 3.07 per minute.

## Frozen thresholds

{"rushed": 2.0, "dragged": 2.5, "monotone": 2.0, "mumbled": 2.5, "loud": 2.0, "long_pause": 5.0, "no_pause": 2.0, "stutter": 3.0}
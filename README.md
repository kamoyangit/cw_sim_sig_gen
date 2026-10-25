# cwsim — labelled CW through an IONOS/Watterson HF channel

**English** | [日本語](README_JP.md)

Offline generator that turns known text into **Morse audio passed through the same
fading channel the IONOS simulator implements**, together with sample-exact ground
truth. Built to produce training and evaluation data for CW (Morse) decoders, where
the bottleneck is not compute but labelled data.

It is an extraction of the DSP in `src/HFSim_BFD_2_05/HFSim_BFD_2_05.ino`
(IONOS HF/VHF Channel Simulator, © 2020 Amateur Radio Safety Foundation Inc., MIT).
The mathematics is preserved deliberately; the Teensy real-time plumbing is not.

## Current status (read this first)

This file is a record of the work, including several conclusions that were later
withdrawn. **The current, correct numbers are these** — anything earlier that disagrees is
historical:

| | median/mean | IQR | flutter |
|---|---|---|---|
| Rayleigh theory | 0.939 | 6.83 dB | — |
| real 40 m capture `7.020153 MHz` | 0.970 ± 0.026 | 6.58 ± 0.46 dB | 0.98 Hz |
| synthetic MPP | 0.922 | 7.08 dB | 1.23 Hz |

The real channel is Rayleigh and the Watterson model reproduces it. Two earlier claims were
artefacts of a 10 ms measurement window and are retracted: "the real channel is steadier
than the model" and "no parameter set satisfies the three statistics at once". The Rayleigh
IQR is 6.83 dB, not 8.7 dB.

Sections marked **superseded** below are kept only so the mistakes are not repeated.
`HowToUse.md` contains the current usage guide without the history.

## Build and run

```sh
make                      # builds build/gen_cw
make test                 # 49 tests (26 channel + 23 calibration)
```

```sh
./build/gen_cw --text "CQ CQ DE JA1ABC K" \
  --channel MPP --paths 2 --snr 0 --wpm 20 \
  --out msg.wav --labels msg.json

./build/gen_cw --text-file corpus.txt --repeat 4 \
  --channel MPD --delays 0,15,35 --doppler 1.0 \
  --fade-depth 12 --fade 0.5 --offset 40 \
  --snr -3 --seed 42 --out hard.wav --labels hard.json
```

`--out` is a 32-bit float WAV. Peaks can exceed 1.0 because fading adds gain on top
of the mean-normalised level; keep it float or normalise downstream, do not clip.

`--list-delay-presets` prints the available delay tables.

For a whole evaluation grid there is `generate_dataset.sh`, which sweeps 5 channels x 3
speeds x 5 S:N values (75 files, about 19 minutes and 2.2 GB) and writes a `README.md` into
the output directory describing the grid, the naming and the bandwidth of the S:N figure.
Seeds are derived from the parameter triple, so changing the grid does not invalidate
files already generated.

## Output

`msg.json` carries, per file:

| field | meaning |
|---|---|
| `text` | the exact message |
| `channel` | mode, paths, Doppler spread, delays, fade, tick rate, mean power gain, seed |
| `snr` | requested / measured audio-band S:N, and the equivalent S:N in 250 Hz and 500 Hz filters |
| `cw` | wpm, Farnsworth element speed, tone, risetime, jitter, drift |
| `characters[]` | per character: `start`, `end` (sample indices into the WAV) and `min_gain_db` |
| `symbols[]` | per element (`dit`, `dah`, `element_gap`, `char_gap`, `word_gap`): `start`, `end` |

`min_gain_db` is the label that makes failure analysis possible: a character lost to a
fade shows a very negative value, one lost to noise does not.

It is defined as the **weakest 10 ms key-down RMS inside that character, relative to the
long-term key-down RMS of the whole render**, and it is measured on the signal *before*
noise is added — so it is a property of the channel, not of the decoder's outcome. It is
computed on the summed multipath signal rather than on one tap: two independent Rayleigh
paths combine to something far shallower than either, and a per-path figure would badly
overstate the fade. Because it reports the worst instant, a single character can straddle
a null and a peak and still show a strongly negative value.

## CW-oriented delay presets

IONOS ships CCIR F.520 delays of 0.5–4 ms, which are calibrated for data modes. At
20 wpm a dit is 60 ms, so a 4 ms echo is ~7% of a symbol — effectively inaudible. HF CW
"hollow"/echo sound is a **10–100+ ms** phenomenon, so this tool also provides
`cw-short`, `cw-mid`, `cw-long` and `cw-roundworld` presets. A PC has no audio
delay-line limit, so `--delays` accepts anything.

Everything else about the fading (complex Gaussian taps, Doppler shaping) is
speed-agnostic, which is why the CCIR parameters are the wrong axis for CW rather than
the model being inadequate.

## S:N reference — read this before quoting numbers

`--snr` is the ratio of **key-down signal power** to **total noise power inside
`--bw`**, both measured over the whole render. Noise is continuous; the signal is not.

This matters because receivers do not listen in 3 kHz. The same recording at
`--snr 0` is about **+10.8 dB** in a 250 Hz filter and **+7.8 dB** in 500 Hz. The JSON
reports all three so a result can never be quoted without its bandwidth.

## Deviations from the IONOS firmware

Each of these is deliberate. The tests pin the behaviour that matters.

1. **Power normalisation instead of the IONOS gain chain.** IONOS mixes signal and
   noise with gains that sum to 1 (`g_s = R/(1+R)`, `g_n = 1/(1+R)`), which makes the
   displayed S:N exact only when the input audio sits at the design level — this is why
   the manual pins input level to a window. Offline there is no input level to depend
   on, so the default normalises the channel to unit long-term power gain and sets the
   noise from the measured signal power. `--hardware-staging` reproduces the original
   absolute chain (including the 4.28 dB `AdjustS_N` calibration constant and the
   fixed 0.235/0.313 noise rms for 3/6 kHz) for hardware-parity comparisons.
2. **Tap gain interpolation.** IONOS updates `AudioMixer4` gains once per audio block,
   so taps are zero-order held at a 64×Doppler rate. That staircases the gain and
   sprinkles the spectrum with step artefacts. Default here is linear interpolation
   (closer to the Watterson ideal); `--interp zoh` reproduces the firmware.
3. **FIR warm-up.** The firmware fades in over ~2 s as the 128-tap delay line fills.
   Offline this is discarded (256 ticks of pre-roll) so that opening characters are not
   silently labelled as deep fades.
4. **Fractional delay in the frequency domain.** The firmware uses a millisecond
   `AudioEffectDelay`; this implementation is sample-exact.
5. **Noise.** Generated as Gaussian and band-limited by an ideal mask, then scaled to
   the target power. The firmware low-passes a uniform white source with a 119-tap FIR.
   Spectrum shape inside the band is flat either way; the high-order statistics differ.
6. **Flat fading** is applied continuously in dB; `--hardware-staging` quantises it to
   whole dB as `Fade()` does (`int()` on the envelope).

### A calibration finding worth knowing

The IONOS tap generator rejects samples outside the annulus `0.1 <= |z|^2 < 1`, which
truncates the Box-Muller radius. The resulting complex weight has a per-component
variance **1.28 dB below** an ideal complex Gaussian (`E[|z|^2] = 1.4883` instead of 2).
The 2-path / 4-path `fltSTD` values (4.0 and 4.0/√2) still make the two cases exactly
equal, which is what that constant exists to guarantee — but the absolute level is not
what a textbook Watterson model would give. The tests assert the ratio between 2 and 4
paths, and the reported `mean_power_gain` reflects the real value.

## What the tests actually verify

`make test` runs 26 checks on the generator, including:

- **S:N accuracy** across −5…+30 dB, measured independently from the rendered audio
  (noise from the inter-element gaps, signal from key-down minus noise): within 0.5 dB.
- **Filter-bandwidth conversion** against `10·log10(BW_audio / BW_filter)`.
- **Doppler scale invariance** — because the 128-tap Gauss FIR runs at 64× the spread,
  the *normalised* tap spectrum must be identical for every CCIR channel. This is the
  strongest single check that the fading generator matches the firmware design.
- **Rayleigh envelope** via scale-free quantiles (`median/mean`, `q10/median`,
  `q90/median`), which is what a complex Gaussian channel must produce.
- **2-path vs 4-path mean power equality** through the raw gain staging.
- **Flat fade depth** against the requested dB.
- **Keying timing**: dit and dah lengths from wpm, and unit-by-unit agreement of the
  whole message span.
- **`min_gain_db` against the rendered audio** — an independent recomputation of the
  weakest key-down 10 ms window per character, so the label cannot silently drift.
- **Label/audio alignment**, determinism per seed, and seed sensitivity.

## Verifying realism against real recordings

`calibrate.py` measures a recording's channel statistics, so a real capture can be
compared with a synthesised one. It needs `numpy` and `scipy`.

```sh
python3 calibrate.py analyze real.wav          # what channel is this?
python3 calibrate.py compare real.wav synth.wav
python3 calibrate.py roundtrip                 # closed-loop check of the estimators
```

`compare` prints the shared metrics side by side and runs two-sample KS tests on the
key-down envelope distribution and on the instantaneous-frequency flutter.

### I/Q input

Yes — and it is the better input. Recordings are auto-detected: a stereo WAV whose two
channels are not near-identical, a complex `.npy`, or raw interleaved `.cf32` / `.cs16`
is treated as I/Q; anything else goes down the audio path. Force it either way with
`--iq` / `--mono`. Raw formats need `--fs`.

```sh
python3 calibrate.py analyze sdr_capture.wav              # offset found automatically
python3 calibrate.py analyze sdr.cf32 --fs 48000 --bw-hz 500
python3 calibrate.py compare sdr_capture.wav synth.wav --bw-hz 500
```

The I/Q path is not a convenience wrapper. With a complex baseband the fading gain is
*observable* rather than inferred: during key-down, `|z|` **is** `|g(t)|` and `arg(z)`
**is** the channel phase, with no Hilbert transform in the way. Two consequences:

- The fade depth is measured against a 500 Hz passband instead of the 3 kHz audio band,
  so less noise sits under the fade floor. In the closed-loop tests the I/Q path lands
  nearer the true depth than the audio path does.
- `rayleigh_median_over_mean` becomes meaningful as a distribution check (0.939 for a
  pure Rayleigh channel; lower when a flat fade is also present).

Bandwidth is the one trap: `snr_db` is measured inside whatever bandwidth the input has,
so an audio file (3 kHz) and an I/Q file (500 Hz) differ by about 7.8 dB purely from the
passband. `compare` prints a note when the input types differ, and fade rate and duty
remain directly comparable.

**Size.** Channelisation is polyphase, so a multi-MHz capture is fine time-wise (measured:
2.4 MHz × 3 s, 7.2 Msamples in 2.7 s, offset recovered exactly). Memory is the limit,
because the capture is loaded whole: expect roughly 64 bytes per input sample at peak.
**Decimate to ≤192 kHz before analysing** — a 500 Hz CW channel needs nothing more — and
the tool warns above 50 Msamples.

### What has and has not been established

**Established (closed loop).** Every estimator is checked against a recording whose true
parameters it never sees. Current accuracy from `make roundtrip`:

| quantity | requested | measured | error |
|---|---|---|---|
| S:N, no fading | 0 / 10 / 25 / 30 dB | +0.46 / +0.04 / −0.06 / −0.08 | < 0.5 dB |
| fading rate | 0.2 / 0.5 / 1.0 Hz | 0.20 / 0.50 / 1.01 | < 0.02 Hz |
| S:N of a faded file | predicted from the fade-average penalty | within 0.1–1.3 dB | — |

The second row matters most: recovering the fade rate to 0.02 Hz means the envelope
model and its measurement agree, not just the generator with itself.

**First real-data result (n = 1).** A 182 s MacSDR I/Q capture of 7.025 MHz (40 m CW),
channelised at 800 Hz around 7.0202 MHz, against a synthetic recording fitted to it:

```sh
gen_cw --wpm 18 --channel MPG --snr 17.8 --bw 3000 --fade-depth 4 --fade 0.05 --seed 4
```

| metric | real | synthetic | delta |
|---|---|---|---|
| S:N in the 800 Hz channel | 18.760 dB | 18.880 dB | **+0.12 dB** |
| fading rate | 0.054 Hz | 0.054 Hz | **0.000 Hz** |
| phase-flutter (IQR) | 1.582 Hz | 1.528 Hz | **−0.054 Hz** |
| key-down envelope std | 4.42 dB | 4.76 dB | +0.34 dB |
| envelope IQR | — | — | see the bandwidth table below |
| phase-flutter distribution | — | — | KS D = 0.016 |
| envelope distribution | — | — | KS D = 0.118 |
| duty | 0.595 | 0.509 | −0.086 |

S:N, fading rate and the flutter distribution match closely. Duty differs because the
synthetic text is not the recorded text.

**Which envelope statistic is trustworthy.** A bandwidth sweep of the one clean carrier
(7.0202 MHz) shows that some of the obvious metrics are artefacts:

| channel bw | dB std | IQR | p10–p90 | median/mean |
|---|---|---|---|---|
| 100 Hz | 9.39 | 10.5 | 26.2 | 0.948 |
| 200 Hz | 6.84 | 7.3 | 15.9 | 0.978 |
| 500 Hz | 4.98 | 6.4 | 12.4 | 0.964 |
| 800 Hz | 4.38 | 6.1 | 11.4 | 0.957 |

A pure Rayleigh envelope has **p10–p90 = 13.4 dB, IQR = 6.83 dB, median/mean = 0.939**.
The peak envelope (p99) is bandwidth-independent, but the fade nulls are limited by the
noise floor, so **p10–p90 and dB std move with the analysis bandwidth and cannot be used to
fit a channel**. The median/mean ratio is stable to ±0.02 across the whole sweep, and the
IQR is stable to about 1 dB. Use those two.

This also retracts an earlier conclusion in this file: a span of 11.5 dB in one capture was
read as evidence of receiver AGC or a Rician component. Across several carriers on the same
band the span ranges from 9 to 26 dB purely as a function of bandwidth and signal quality,
so that inference was not supported. Measured with the stable statistics, the clean
carrier's median/mean is **0.96–0.98**, i.e. **the real channel is Rayleigh, as the
Watterson model assumes** — no AGC or Rician term is needed to explain it.

**Not established: generality.** One recording of one signal on one night is not a
distribution. The honest claim remains *"a Watterson/CCIR channel with the firmware's
exact tap generator, matched to a real 40 m CW capture on envelope statistics"* — not
*"statistically identical to the bands"*.

One documented caveat applies to any real capture: `--snr` is defined at a **fade peak**,
while any recording averages over the fade sweep. `fade_average_penalty_db()` computes the
difference (about −4 dB for a 12 dB fade) so the two can be compared properly.

### Capture protocol, to close the gap

1. **Prefer a known text.** ARRL/W1AW code-practice transmissions are real RF with
   published copy, so the same file serves for both channel calibration and decoder
   scoring. WebSDR or KiwiSDR recordings around 7.000–7.030 MHz work for ordinary QSOs,
   but then the text is unknown — that is fine for channel statistics, not for CER.
2. **Record the metadata**: UTC date/time, frequency, receiver, filter width, AGC setting,
   antenna. A 500 Hz filter and AGC-on produce a different file from 2.7 kHz and AGC-off.
3. **60 s minimum**, WAV, 44.1 kHz, mono, uncompressed. Do not enable noise reduction,
   notch filters or audio compression — the point is to measure what the channel did.
4. **Avoid clipping.** A clipped recording cannot be measured.
5. Run `analyze`, then generate a candidate with the closest settings, `compare`, adjust
   `--channel/--doppler/--fade-depth/--fade/--snr`, repeat. The `delta` column is the
   objective function.

Two useful public sources for real off-air audio with labels, if downloading is possible:
ARRL W1AW practice archives (real RF, exact transcripts) and the Morse Learning Machine
Challenge v2 training set (hand-labelled noisy CW). Neither is redistributable here, which
is why they are not vendored into this repository.

### Finding a usable carrier (`scan`)

A crowded band has to be searched before anything can be measured, and the strongest peak
is usually the wrong choice.

```sh
python3 calibrate.py scan capture.wav --center-mhz 7.025 --bw-hz 200
```

`scan` measures every carrier and scores it 0-110 as a calibration reference: dit:dah
ratio near 3.00, duty in 0.40-0.70, enough elements, adequate S:N, an envelope
median/mean near the Rayleigh 0.939, and stability of that ratio when the channel width
doubles. It reports the reasons for every deduction rather than just a number.

Two traps it is built to catch, both of which cost real time here:

- **A strong peak is not a clean signal.** On the 40 m captures the strongest carrier
  often has a neighbour 40-90 Hz away, which no filter bandwidth can separate.
  `in_channel_peak_ratio_db` measures that directly: a competing peak within 15 dB of the
  wanted carrier marks the measurement `UNUSABLE`, because narrowing further would damage
  the keying before it removed the interference.
- **Keying sidebands are not separate carriers.** They sit 30-40 Hz from the carrier and
  around -25 dB, so they appear in a full-band peak list. This is why the purity test is
  measured inside the channel instead of from the carrier list.

On the two 40 m captures here, `scan` independently ranked `7.020153 MHz` as the only
`clean` carrier (score 110) in the 182 s file and `7.028153 MHz` (score 83) in the 73 s
file — the same signals that were found by hand.

Applied to the three 31 m captures (9.000 MHz, 11-16 s each) it rejects every carrier,
and correctly: 20-56 elements is far too few to estimate a fading distribution, and
`competing peak -8 dB` / `-0 dB` inside the channel means what was picked up is not a
single signal at all. **The only carrier in this collection that survives as a
single-Rayleigh-channel reference is 7.020153 MHz.** Everything here therefore rests on a
sample size of one, and no amount of internal consistency changes that.

### How much can one capture support?

> **Superseded.** The IQR column below was measured with the 10 ms window and is too
> large; the corrected values are in *The estimator was the problem* below.


Splitting the 182 s MacSDR capture into six independent 30 s segments and measuring each
shows the measurement is repeatable and the channel is stationary over three minutes:

| metric | mean | sd | min | max |
|---|---|---|---|---|
| S:N | 24.42 dB | 0.60 | 23.62 | 25.42 |
| median/mean | 0.992 | 0.035 | 0.948 | 1.035 |
| envelope IQR | 7.28 dB | 0.41 | 6.57 | 7.77 |
| phase flutter (IQR) | 0.982 Hz | 0.080 | 0.888 | 1.132 |
| dit:dah | 2.719 | 0.082 | 2.63 | 2.87 |
| duty | 0.683 | 0.012 | 0.668 | 0.698 |

So one *capture* yields six usable estimates, and they agree. That is not six channels —
it does not license a general claim — but it does show the estimator is not producing
noise, and the numbers below are stable properties of this signal.

Two results here changed the generator:

- **`dit:dah` is 2.72, not the ITU 3.00**, and it is stable to 0.08. Real keying is not
  exact, so `--dah-ratio` now exists (default 3.0) to reproduce a measured habit. Beware
  that the estimator itself reads about 0.1-0.3 low depending on fading, because the mask
  edges shift with the envelope.
- **`--fade-depth` alone cannot reproduce a large IQR**; that is what pointed at the HDSDR
  record chain rather than the propagation.

A calibrated candidate (`--wpm 13.8 --dah-ratio 2.72 --channel MPG --snr 17 --fade-depth 2
--fade 0.05`) reproduces the S:N to 0.1 dB and the flutter to 0.1 Hz, but its **segment-to-
segment spread is 2-3x larger than the real capture's** (median/mean sd 0.112 vs 0.035,
IQR sd 1.49 vs 0.41). The real channel reads as *steadier* than the model predicts — but
**that comparison was made with a wrongly chosen synthetic case and is retracted**; with
MPP and the correct window the spreads match (sd 0.033 vs 0.035, see *The estimator was the
problem*).

### Rician K: tried, and it does not close the gap

> **Superseded.** This section chases an IQR that the estimator had inflated. `--rician-k`
> is still a real option, but it was not needed. See *The estimator was the problem*.


The measured capture has median/mean 0.992 and IQR 7.28 dB, against Rayleigh's 0.939 and
6.83 dB. The median/mean is above Rayleigh's, which is what a steady path produces, so
`--rician-k` was added (K = steady-path power / scattered power; total mean power stays at
unity so S:N keeps its meaning). Sweeping it on 180 s signals:

| K | median/mean | IQR | flutter (IQR) |
|---|---|---|---|
| 0 (Rayleigh) | 0.958 | 10.5 | 0.84 |
| 4 | **0.995** | 9.66 | 0.74 |
| 8 | 1.031 | 8.65 | 0.66 |
| 16 | 1.109 | 6.70 | 0.53 |
| **real capture** | **0.992** | **7.28** | **0.98** |

`K = 4` reproduces the median/mean almost exactly (0.995 vs 0.992), and the
segment-to-segment spread with it (sd 0.033 vs the real 0.035). But the **same K leaves the
IQR 2.4 dB too wide**, and reaching IQR 7.28 needs K around 12, where median/mean has
already overshot to 1.05. The flutter goes the wrong way entirely: it falls as K rises,
while the real signal has *more* flutter than pure Rayleigh.

So a single K cannot match all three. `--rician-k` is a real and useful degree of freedom —
it is the right knob for the median/mean — but **the measured channel is not a Rician
channel**, and adding it did not explain the residual.

**This also retracts the previous claim that the real channel is "steadier than the model
predicts".** That came from comparing against a badly chosen synthetic case (MPG with
`--fade-depth 2`). With MPP and the correct K, the segment-to-segment spread matches the
real capture (sd 0.033 vs 0.035). There is no extra stability to explain.

What still does not fit is the combination *narrow median/mean + narrow IQR + wide phase
flutter*. None of Watterson's parameters (Doppler, delay, fading depth) nor Rician K
produces that triple. It needs a model with more structure than a single scattered path
plus one steady path — for example two scattered paths of unequal strength at different
Doppler shifts, which is testable with the existing generator but has not been tried.

### The estimator was the problem, and it is now fixed

The fading statistics were being measured in **10 ms** windows. That is short enough for the
within-element ripple to be reported as fading, and it inflated every envelope figure. The
window is now half a dit, clamped to 8-40 ms.

Verified against a ground-truth mask on a known Rayleigh channel:

| mask | window | median/mean | IQR |
|---|---|---|---|
| oracle | 5 ms | 0.983 | 14.02 |
| oracle | 10 ms | 0.978 | 12.04 |
| oracle | 20 ms | 0.957 | 9.30 |
| **oracle** | **40 ms** | **0.922** | **6.89** |
| detected | 10 ms | 0.964 | 17.11 |
| detected | 40 ms | 0.964 | 9.20 |
| *Rayleigh theory* | — | *0.939* | *6.83* |

With the window fixed, the real capture and a plain synthetic Rayleigh channel now agree
with each other **and** with theory (six 30 s segments each):

| | median/mean | IQR | flutter |
|---|---|---|---|
| Rayleigh theory | 0.939 | 6.83 dB | — |
| **real 7.020153 MHz** | **0.970** | **6.58 dB** | 0.98 Hz |
| **synthetic MPP** | **0.922** | **7.08 dB** | 1.23 Hz |

**So the real channel is Rayleigh, the Watterson model reproduces it, and the earlier
"three statistics that no parameter set could satisfy" was an artefact of the estimator.**
That conclusion, and the searches built on it, are withdrawn:

- Rician K is not needed to explain the envelope. `--rician-k` remains as a genuine
  modelling option, not as a correction for a measurement error.
- Nakagami-m would have been added to chase the same artefact. It was checked numerically
  first and does not fit the numbers anyway (m = 6 gives median/mean 0.9926 but only 2.45 dB
  of IQR), so it was not implemented.
- Per-path Doppler is a real capability and stays, but it was not needed to fit the data.

**Two errors of mine are recorded here so they are not repeated:** the Rayleigh IQR is
**6.83 dB**, not 8.7 dB (13.39 dB is the p10-p90 span), and the estimator's window is a
parameter that changes the answer by several dB.

**The key-down mask: tried, and left alone.** It agrees with a ground-truth mask only to
IoU 0.665, and that inflates the measured IQR by 2-3 dB against an oracle mask (9.20 vs
6.89 at 40 ms). Attempts to improve it:

| change | IoU | oracle-style envelope stats | effect on real captures |
|---|---|---|---|
| baseline (percentile hysteresis) | 0.665 | rayl 0.964, IQR 9.20 | works |
| zero-phase `filtfilt` (removes the ~15 ms FIR delay) | 0.676 | IQR 10.40 | worse |
| detection window 10-30 ms | 0.664-0.667 | — | no change |
| Otsu threshold | 0.630 | IQR 8.6, but FN 10% | worse |
| peak-relative thresholds | 0.625 | **rayl 0.919, IQR 6.34** | **breaks: IQR 6.6 -> 18.1, ratio 2.7 -> 4.2** |

Two things this taught:

- **IoU is the wrong objective function.** The peak-relative variant has *lower* IoU yet
  reproduces the oracle's fading statistics almost exactly. Optimising IoU would have
  picked the wrong configuration.
- **A fading peak defeats peak-relative thresholds on real data.** When the carrier is
  deep in a fade, "12 dB below the peak" falls inside the noise, the mask fragments, and
  the element lengths collapse into noise. The percentile version tolerates a moving peak,
  which is why it survives even though it is measurably worse on clean synthetic input.

Getting past IoU 0.665 needs a method that uses the known Morse structure (element
durations constrained to k x the unit length) rather than thresholding an envelope
independently. That has not been attempted.

`dit:dah` is likewise read about 0.3 low, for the same reason.

### Per-path Doppler: independent flutter, but the IQR still will not move

> **Superseded.** The IQR gap that motivated this was a measurement artefact. `--path-doppler`
> is still a real capability. See *The estimator was the problem*.


`--path-doppler` gives each scattered path its own Doppler spread, so one path can fade
slowly (narrow envelope) while another flutters fast. That is a degree of freedom the plain
Watterson model does not have. Measured on 60 s signals against the real target
(median/mean 0.992, IQR 7.28, flutter 0.98 Hz):

| path Doppler | median/mean | IQR | flutter |
|---|---|---|---|
| 0.5, 0.5 | 0.988 | 10.59 | 0.43 |
| 0.5, 4 | 0.931 | 10.75 | 2.06 |
| 0.5, 10 | 0.934 | 10.39 | 4.87 |
| 0.5, 25 | 0.953 | 9.72 | 11.18 |
| 1, 15 | 0.946 | 9.72 | 7.46 |
| **real capture** | **0.992** | **7.28** | **0.98** |

**It works for what it was added for**: phase flutter is now independently tunable — 0.43 to
11.2 Hz by changing the second path alone — which no single Doppler value could do. An
intermediate setting lands on the measured 0.98 Hz.

**It does not fix the IQR.** Whatever the path split, the envelope IQR stays at 9.7-10.8 dB
against the measured 7.28 dB. So the search so far has ruled out, with measurements rather
than argument:

| tried | effect on IQR | verdict |
|---|---|---|
| flat fading depth (0-15 dB) | 11 max | too weak |
| Doppler spread (1-25 Hz) | unchanged | no effect |
| delay spread (2-140 ms) | unchanged | no effect |
| Rician K (0-16) | 10.5 -> 6.7, but median/mean overshoots to 1.11 | wrong trade-off |
| two paths at different Doppler | 9.7-10.8 | no effect |

A measured channel with **narrow median/mean, narrow IQR and wide flutter at the same time**
is not expressible by any combination of these. Rician K comes closest and fails on the
trade-off, which points at the distribution *shape* rather than its scale: a Nakagami-m or
lognormal envelope can set the IQR independently of the median, whereas Rician and Rayleigh
cannot. That is the next thing to try, and it is a small change to the tap generator.

### What the HDSDR captures revealed (and what it was not)

The 2012-2017 HDSDR captures in the companion folder all show an envelope IQR of 12-28 dB,
against a Rayleigh 6.83 dB. Chasing that difference ruled out, one at a time:

| hypothesis | test | result |
|---|---|---|
| receiver AGC compressing the fades | AGC narrows a distribution; this one is *wider* | wrong direction |
| flat fading / QSB | sweep `--fade-depth` 0→14 dB | IQR only reaches 11.1 dB |
| fast fading | sweep `--doppler` 1→25 Hz | IQR stays 7.5-8.4 dB |
| frequency-selective fading | `--delays 0,20,45` and `0,60,140` ms | IQR 8-10 dB, no widening |
| a competing carrier in band | bandwidth sweep, `in_channel_peak_ratio_db` | -20 dB for *both* the clean and the bad carrier, so it does not discriminate |

A synthesised single-Rayleigh channel sits at **7-11 dB IQR regardless of every parameter
the model has**. The clean MacSDR carrier measures 10.5 / 7.3 / 6.8 dB at 100 / 200 / 300 Hz,
matching the model. The HDSDR carrier measures 26.9 / 23.0 / 13.1 dB, matching nothing the
model can produce.

So the conclusion is not "the HDSDR signal is a bit off": **those captures are not
reproducible by this channel model at all**, and the cause is in the HDSDR record chain
(not its AGC, which would go the other way — noise reduction, notch filtering or overload
remain candidates) rather than in the propagation. They should not be used as channel
references. The MacSDR captures can.

This is also why `envelope_db_iqr > 16 dB` now marks a carrier unusable: it is the sharpest
test available for "is this one Rayleigh channel", and it catches contamination that both
median/mean and the in-channel peak ratio miss.

### When a good capture turns up: the workflow

```sh
# 1. Is any carrier in here usable?
python3 calibrate.py scan capture.wav --center-mhz 7.025
#    -> take an offset that scores well and say "clean"

# 2. Measure it properly and get a starting point
python3 calibrate.py fit capture.wav --offset-hz -4847 --bw-hz 200

# 3. Generate the proposal, compare, and adjust
./build/gen_cw <the command fit printed> --out match.wav --labels match.json
python3 calibrate.py compare capture.wav match.wav --iq --offset-hz -4847 --bw-hz 200
```

`fit` measures six segments, reports mean and spread for every statistic, and prints a
`gen_cw` command to start from. On the reference capture it lands within 0.006 of the
measured median/mean and 0.08 Hz of the flutter on the first attempt:

| metric | real | `fit` proposal | delta |
|---|---|---|---|
| median/mean | 0.960 | 0.954 | −0.006 |
| phase flutter | 0.972 Hz | 1.054 Hz | +0.08 Hz |
| envelope IQR | 6.67 dB | 7.23 dB | +0.56 dB |
| S:N | 24.61 dB | 25.48 dB | +0.87 dB |
| fade rate | 0.072 Hz | 0.310 Hz | +0.24 Hz |

Adjust in this order, because the later ones barely affect the earlier ones:

1. **`--snr`** until the S:N delta is under ~1 dB. It is the only parameter that moves S:N
   much, and it is the one `fit` estimates least reliably (it uses a measured 3 kHz→200 Hz
   correction of 7.5 dB, which is empirical).
2. **`--channel` / `--doppler`** until the flutter and median/mean match. Doppler controls
   flutter directly.
3. **`--fade-depth`** only if the IQR is short. `fit` leaves it at 0 on purpose: a flat
   fade and multipath fading are not separable from the envelope alone, so setting it early
   fits one artefact with another.
4. **`--dah-ratio`** last, and only if you believe the operator was exact. `fit` reproduces
   the *measured* ratio, which the estimator reads about 0.3 low.

Two things to record when a capture turns out to be usable: the **metadata** (UTC, frequency,
receiver, filter width, AGC and noise-reduction settings) and the **segment spread**.
A capture whose segment-to-segment spread is large cannot support tight claims however good
its mean looks.

### Practical notes from real captures

- A 40 m CW band at night is crowded: the first carrier picked by "strongest peak" turned
  out to have five carriers inside 60 Hz. `--bw-hz` must be narrow enough to isolate one
  signal but wide enough not to distort the keying envelope; 500–800 Hz worked here.
- `--bw-hz` genuinely matters now (it did not in an earlier revision — a bug). Use the
  sweep in the tests as a sanity check: S:N should fall and the measured fade depth should
  shrink as the channel widens.
- A signal whose dit/dah duration ratio is far from 3 is usually not one signal; check the
  carrier count before trusting its statistics.

## Bug found and fixed: WGN silently became MPP

`--channel WGN --delays 0,2` produced exactly the same channel as `--channel MPP`
(Doppler 1 Hz, tick 64 Hz). The cause was ordering: `--delays` raised `paths` to 2, and a
generic "multipath needs a Doppler spread" correction then fired for every mode including
WGN. WGN therefore had a no-fading mode that could not be reached once delays were given.

Fixed by only inventing a Doppler spread for fading modes. `--channel WGN --delays 0,2` is
now **static multipath** — two constant-gain paths at 0 and 2 ms, which is a comb filter and
a legitimate thing to want. `--channel WGN --path-doppler 0.2,3` still fades, because that
is explicitly requested.

The same pass fixed a second inconsistency: with `--path-doppler`, the scalar
`doppler_spread_hz` in the labels reported a value derived from `paths` rather than from the
paths themselves. It now mirrors path 0, with the full vector in `path_doppler_hz`.

Both are covered by regression tests.

### Full option-interaction audit

The WGN/MPP collision prompted a sweep of every mode/option combination. Five more problems
turned up, all of the same family: accepted silently, did nothing (or crashed).

| combination | before | now |
|---|---|---|
| `--channel WGN --doppler 1` | **uncaught exception, process aborted** | error, exit 2 |
| `--channel MPP --doppler 0` | forced back to 1 Hz | honoured (static multipath) |
| `--channel WGN --rician-k 4` | label said 4, nothing applied | warning, label corrected to 0 |
| `--channel WGN --fade-depth 10` | label said 10, nothing applied | warning, label corrected to 0 |
| `--channel WGN --paths 4` | silently ignored | warning |

The abort was the worst of them: `--doppler` raised `paths` to 2 without extending the delay
list, so the `Renderer` constructor threw. It now reports instead, and the constructor is
wrapped so that any future mismatch cannot terminate the process.

Option handling is now validated once, after parsing and before rendering, and warnings go
to stderr rather than into the labels. The rule applied: **a value that is not going to be
used must not be reported as if it were.**

## Not implemented yet

- Receiver AGC (attack/release/hang) — the largest remaining realism gap, because on a
  deep fade a real AGC lifts the noise with the signal and this tool does not.
- Transmitter defects beyond risetime/jitter/drift: key clicks, chirp, PA non-linearity.
- QRM (a second CW signal — trivially a second renderer summed in) and QRN (impulse
  noise, power-line hum).
- Receiver front-end filter ringing as an explicit stage.
- Real-recording validation beyond the single 40 m capture used here. The estimator and the
  workflow are ready; more captures are the bottleneck, not the code.

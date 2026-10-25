#!/usr/bin/env python3
"""calibrate.py -- measure a CW recording's channel statistics, and compare two.

Purpose: decide whether audio produced by gen_cw is statistically like a real HF CW
recording. Two modes:

    analyze  <wav>                    channel statistics of one recording
    compare  <real.wav> <synth.wav>   side-by-side statistics + two-sample tests
    roundtrip [--workdir DIR]         closed-loop check: synthesise at known settings,
                                      re-measure, and report the estimation error

No physical calibration is attempted: compare() only needs statistics that are
*comparable* between the two recordings, not absolutely correct. The one figure that
does claim absolute meaning is snr_db, which is measured in the file's own bandwidth.

Requires numpy and scipy.
"""

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import signal, stats
from scipy.io import wavfile
from scipy.ndimage import binary_propagation, minimum_filter1d, uniform_filter1d

HERE = Path(__file__).resolve().parent
GEN = HERE / "build" / "gen_cw"


# --------------------------------------------------------------------------- io
def load_wav(path):
    """Read a WAV, returning (sample_rate, mono float64 in [-1, 1] for int formats)."""
    sr, x = wavfile.read(path)
    if x.dtype.kind == "i":
        x = x.astype(np.float64) / float(np.iinfo(x.dtype).max)
    elif x.dtype.kind == "f":
        x = x.astype(np.float64)
    else:
        raise ValueError(f"unsupported WAV dtype {x.dtype}")
    if x.ndim > 1 and x.shape[1] > 1:
        raise ValueError(
            f"{path} has {x.shape[1]} channels. calibrate expects mono audio; if this is "
            "an I/Q recording, pass --iq (I in channel 0, Q in channel 1). Averaging the "
            "two channels would destroy the signal."
        )
    if x.ndim > 1:
        x = x[:, 0]
    return sr, x


# ------------------------------------------------------------------- primitives
def rms_envelope(x, sr, win_ms=5.0):
    win = max(3, int(round(sr * win_ms / 1000.0)))
    return np.sqrt(np.maximum(uniform_filter1d(x * x, win, mode="nearest"), 0.0))


def rms_envelope_complex(z, sr, win_ms=5.0):
    win = max(3, int(round(sr * win_ms / 1000.0)))
    return np.sqrt(np.maximum(uniform_filter1d(np.abs(z) ** 2, win, mode="nearest"), 0.0))


def keydown_mask(x, sr):
    """Convenience wrapper: envelope of a real signal, then the shared splitter."""
    return keydown_mask_from_env(rms_envelope(x, sr, win_ms=4.0), sr)


def keydown_mask_from_env(env, sr):
    """Split key-down from key-up with hysteresis on the rms envelope.

    A single threshold is not enough. Set high, it cuts the bottom out of deep fades
    (which then look like 20 dB of extra noise); set low, it fills in the gaps at low
    S:N. So the mask is seeded from a high threshold (the geometric mean of floor and
    peak, i.e. halfway in dB) and then grown through a low threshold (1.4x the noise
    floor) only where it is connected to a seeded region. Fades survive, gaps do not.
    """
    lo = float(np.percentile(env, 5))
    hi = float(np.percentile(env, 95))
    if hi <= 0.0 or lo <= 0.0 or hi / max(lo, 1e-30) < 1.3:
        return np.ones(env.shape[0], dtype=bool)

    # Percentile-based hysteresis. Peak-relative thresholds were tried and are better on
    # a high-S:N synthetic (oracle-like envelope statistics) but break on real captures:
    # a fading peak makes thr_lo too high, the mask fragments, and the measured IQR jumps
    # from 6.6 to 18 dB. Percentiles tolerate a moving peak, so they stay.
    thr_lo = lo * 1.4
    thr_hi = math.sqrt(lo * hi)
    if thr_hi <= thr_lo:
        thr_hi = thr_lo * 1.2
    strong = env > thr_hi
    weak = env > thr_lo
    if not strong.any():
        strong = env > thr_lo
    mask = binary_propagation(strong, np.ones(3, dtype=bool), mask=weak)

    min_run = max(1, int(round(0.020 * sr)))
    mask = _close_short_gaps(mask, min_run)
    return _drop_short_runs(mask, min_run)


def _runs(mask):
    """Start/end index pairs of the True runs in a boolean array."""
    if mask.size == 0 or not mask.any():
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    d = np.diff(mask.astype(np.int8))
    starts = np.flatnonzero(d == 1) + 1
    ends = np.flatnonzero(d == -1) + 1
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mask.size]
    return starts, ends


def _close_short_gaps(mask, max_gap):
    starts, ends = _runs(~mask)
    out = mask.copy()
    for s, e in zip(starts, ends):
        if s > 0 and e < mask.size and (e - s) <= max_gap:
            out[s:e] = True
    return out


def _drop_short_runs(mask, min_run):
    starts, ends = _runs(mask)
    out = mask.copy()
    for s, e in zip(starts, ends):
        if (e - s) < min_run:
            out[s:e] = False
    return out


def erode(mask, n):
    if n <= 0:
        return mask
    return minimum_filter1d(mask, size=2 * n + 1, mode="nearest")


def tone_frequency(x, mask, sr):
    seg = x[mask]
    if seg.size < 1024:
        seg = x
    n = 1 << int(math.ceil(math.log2(min(seg.size, 1 << 16))))
    seg = seg[:n]
    spec = np.abs(np.fft.rfft(seg * np.hanning(seg.size)))
    k = int(np.argmax(spec))
    freqs = np.fft.rfftfreq(seg.size, 1.0 / sr)
    # Parabolic interpolation around the peak for sub-bin accuracy.
    if 0 < k < spec.size - 1:
        a, b, c = spec[k - 1], spec[k], spec[k + 1]
        denom = a - 2 * b + c
        delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
        return float(freqs[k] + delta * (freqs[1] - freqs[0]))
    return float(freqs[k])


def occupied_bandwidth(x, sr, fraction=0.99):
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    freqs = np.fft.rfftfreq(x.size, 1.0 / sr)
    cum = np.cumsum(spec)
    if cum[-1] <= 0:
        return 0.0
    lo = freqs[int(np.searchsorted(cum, (1 - fraction) / 2 * cum[-1]))]
    hi = freqs[int(np.searchsorted(cum, (1 + fraction) / 2 * cum[-1]))]
    return float(max(hi - lo, 0.0))


def snr_estimate(x, mask, sr):
    """Key-down signal power vs key-up noise power, both in the file's own bandwidth."""
    if mask.all() or not mask.any():
        return float("nan"), 0.0, 0.0
    p_on = float(np.mean(x[mask] ** 2))
    p_off = float(np.mean(x[~mask] ** 2))
    p_sig = max(p_on - p_off, 1e-30)
    return 10.0 * math.log10(p_sig / max(p_off, 1e-30)), p_sig, p_off


def envelope_db_series(x, mask, sr, win_ms=10.0):
    """Per-window key-down rms of a real signal, as (times, values)."""
    return envelope_series_from_env(rms_envelope(x, sr, win_ms=win_ms), mask, sr, win_ms)


def envelope_series_from_env(env, mask, sr, win_ms=10.0):
    """Per-window key-down rms, as (times, values). Windows must be fully key-down."""
    win = max(3, int(round(sr * win_ms / 1000.0)))
    hop = max(1, win // 2)
    if env.size <= win:
        return np.asarray([]), np.asarray([])
    inside = erode(mask, win // 2).astype(np.int64)
    # Vectorised "is this whole window inside": a window is full iff its running-sum
    # increment equals its length. A python loop over windows is 100x slower here.
    cs = np.concatenate(([0], np.cumsum(inside)))
    full = (cs[win:] - cs[:-win]) == win
    idx = np.flatnonzero(full)[::hop]
    if idx.size == 0:
        return np.asarray([]), np.asarray([])
    return (idx + win / 2.0) / sr, env[idx + win // 2]


def fade_spectrum(ts, vs, sr, grid_hz=20.0):
    """Locate the slow fading periodicity in the key-down envelope.

    The samples are irregular (they only exist while keying), so they are interpolated
    onto a uniform grid first. That is adequate for a line-like QSB component and is
    validated by roundtrip().
    """
    if ts.size < 8:
        return 0.0, 0.0, None, None
    db = 20.0 * np.log10(np.maximum(vs, 1e-30))
    db = db - db.mean()
    t0, t1 = ts[0], ts[-1]
    n = max(16, int(round((t1 - t0) * grid_hz)))
    if n < 16:
        return 0.0, 0.0, None, None
    grid = np.linspace(t0, t1, n)
    res = np.interp(grid, ts, db)
    res = res - res.mean()
    nfft = 1 << int(math.ceil(math.log2(n)))
    spec = np.abs(np.fft.rfft(res * np.hanning(n), nfft))
    freqs = np.fft.rfftfreq(nfft, 1.0 / grid_hz)
    if spec.size < 4:
        return 0.0, 0.0, None, None
    spec[0] = 0.0
    k = int(np.argmax(spec))
    peak_hz = float(freqs[k])
    med = float(np.median(spec[1:])) if spec.size > 2 else 0.0
    prominence = float(spec[k] / med) if med > 0 else 0.0
    return peak_hz, prominence, freqs, spec


def inst_freq_series(x, mask, sr, edge_ms=8.0, avg_ms=20.0):
    """Frequency of the tone inside key-down regions, measured over 20 ms windows.

    Its spread is the audible signature of Doppler spreading: a pure tone through a
    static channel gives zero spread, a fading one flutters.

    Two safeguards are needed. A sample-by-sample phase difference is swamped by noise
    once the signal fades (it reports hundreds of Hz), so the difference is taken over
    `avg_ms`. And points whose envelope has collapsed below 15% of the key-down median
    are dropped: there the phase belongs to the noise, not to the carrier.
    """
    edge = max(1, int(round(edge_ms * 1e-3 * sr)))
    half = max(1, int(round(0.5 * avg_ms * 1e-3 * sr)))
    # The difference spans +/- half, so the region must be eroded by half as well or the
    # window straddles a key-up gap and reports a garbage frequency.
    inner = erode(mask, edge + half)
    if int(inner.sum()) < 1024 or x.size <= 2 * half + 2:
        return np.asarray([])
    analytic = signal.hilbert(x)
    mag = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    dph = np.zeros(phase.size)
    dph[half:-half] = phase[2 * half :] - phase[: -2 * half]
    f = dph * (sr / (2.0 * math.pi * 2 * half))
    ref = float(np.median(mag[inner]))
    if ref <= 0.0:
        return np.asarray([])
    valid = inner & (mag > 0.15 * ref)
    f = f[valid]
    if f.size == 0:
        return f
    return f - float(np.median(f))


# -------------------------------------------------------------------------- I/Q
def load_iq(path, raw_fs=None):
    """Read an I/Q recording as (sample_rate, complex128).

    Accepts a stereo WAV (I on channel 0, Q on channel 1, as most SDR software writes),
    a .npy holding a complex array or an (N,2) array, or raw interleaved .cf32 / .cs16.
    Raw formats carry no sample rate, so `raw_fs` is required for them.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".npy":
        arr = np.load(path)
        if np.iscomplexobj(arr):
            if raw_fs is None:
                raise ValueError("--fs is required when loading a complex .npy")
            return float(raw_fs), arr.astype(np.complex128)
        if arr.ndim == 2 and arr.shape[1] == 2:
            if raw_fs is None:
                raise ValueError("--fs is required when loading an (N,2) .npy")
            return float(raw_fs), arr[:, 0].astype(np.float64) + 1j * arr[:, 1].astype(np.float64)
        raise ValueError(f"{path}: .npy must be complex or shaped (N, 2)")

    if ext in (".cf32", ".fc32", ".cfile"):
        raw = np.fromfile(path, dtype="<f4")
        if raw.size % 2:
            raw = raw[:-1]
        iq = raw[0::2].astype(np.float64) + 1j * raw[1::2].astype(np.float64)
        if raw_fs is None:
            raise ValueError("--fs is required for raw .cf32 input")
        return float(raw_fs), iq

    if ext == ".cs16":
        raw = np.fromfile(path, dtype="<i2")
        if raw.size % 2:
            raw = raw[:-1]
        iq = raw[0::2].astype(np.float64) + 1j * raw[1::2].astype(np.float64)
        iq /= float(np.iinfo(np.int16).max)
        if raw_fs is None:
            raise ValueError("--fs is required for raw .cs16 input")
        return float(raw_fs), iq

    # WAV: stereo means I/Q unless the two channels are essentially the same signal.
    sr, x = wavfile.read(path)
    if x.dtype.kind == "i":
        x = x.astype(np.float64) / float(np.iinfo(x.dtype).max)
    else:
        x = x.astype(np.float64)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError(f"{path} is not a stereo WAV; pass the audio file through the audio path")
    a, b = x[:, 0], x[:, 1]
    if a.std() > 0 and b.std() > 0:
        corr = abs(float(np.corrcoef(a, b)[0, 1]))
        if corr > 0.9:
            raise ValueError(
                f"{path}: channels correlate at {corr:.3f}, so this looks like duplicated "
                "audio rather than I/Q. Use the audio path, or force --iq."
            )
    return float(sr), a + 1j * b


def find_carrier(iq, sr, guard_hz=200.0):
    """Strongest narrowband peak, ignoring the DC spike that SDRs always have."""
    nperseg = int(min(16384, max(256, iq.size // 8)))
    f, pxx = signal.welch(iq, sr, nperseg=nperseg, return_onesided=False)
    order = np.argsort(f)
    f, pxx = f[order], pxx[order]
    valid = np.abs(f) > guard_hz
    if not valid.any():
        return 0.0
    return float(f[valid][int(np.argmax(pxx[valid]))])


def channelize(iq, sr, offset_hz, bw_hz, target_fs):
    """Mix the wanted carrier to 0 Hz, low-pass, and decimate to `target_fs`.

    Everything outside `bw_hz` is rejected before decimation, so the returned complex
    baseband contains the CW signal plus only the noise that shares its passband.
    """
    n = iq.size
    t = np.arange(n) / float(sr)
    z = iq * np.exp(-2j * math.pi * offset_hz * t)

    # Stage 1: coarse decimation to a few times the wanted channel. Polyphase, so it
    # stays linear in the input size; its anti-alias point is far above `bw_hz`, so the
    # channel itself is untouched.
    decim1 = max(1, int(sr // max(8.0 * bw_hz, 400.0)))
    if decim1 > 1:
        z = signal.resample_poly(z, 1, decim1)
    fs1 = sr / decim1

    # Stage 2: the channel filter that actually sets `bw_hz`. This must happen at the
    # decimated rate -- but it must happen, or --bw-hz silently becomes fs_out/2 and the
    # neighbouring signals on a crowded band leak straight into the measurement.
    cutoff = min(0.45, max(1e-4, (bw_hz / 2.0) / (fs1 / 2.0)))
    z = signal.lfilter(signal.firwin(129, cutoff), 1.0, z)

    decim2 = max(1, int(round(fs1 / float(target_fs))))
    if decim2 > 1:
        z = z[::decim2]
    return z, fs1 / decim2


def channel_purity_db(z, fs, main_hz=0.0, exclude_hz=20.0):
    """Strongest competing peak inside the channel, relative to the wanted carrier (dB).

    This is the direct test of "is there one signal here". A carrier list from the whole
    capture cannot answer it: keying sidebands land 30-40 Hz away and look like separate
    carriers, while a genuinely interfering signal may be too weak to appear in a
    full-band peak list at all.
    """
    if z.size < 256:
        return None
    nperseg = int(min(z.size, 1 << 15))
    f, pxx = signal.welch(z, fs, nperseg=nperseg, return_onesided=False)
    order = np.argsort(f)
    f, pxx = f[order], pxx[order]
    db = 10.0 * np.log10(pxx + 1e-30)
    main = np.abs(f - main_hz) <= exclude_hz
    if not main.any():
        return None
    others = db[~main]
    if others.size == 0:
        return None
    return float(np.max(others) - np.max(db[main]))


def snr_estimate_complex(z, mask):
    if mask.all() or not mask.any():
        return float("nan"), 0.0, 0.0
    p_on = float(np.mean(np.abs(z[mask]) ** 2))
    p_off = float(np.mean(np.abs(z[~mask]) ** 2))
    p_sig = max(p_on - p_off, 1e-30)
    return 10.0 * math.log10(p_sig / max(p_off, 1e-30)), p_sig, p_off


def inst_freq_complex(z, mask, sr, avg_ms=20.0):
    """Rate of change of the complex baseband phase: the direct signature of Doppler."""
    half = max(1, int(round(0.5 * avg_ms * 1e-3 * sr)))
    inner = erode(mask, half)
    if int(inner.sum()) < 256 or z.size <= 2 * half + 2:
        return np.asarray([])
    phase = np.unwrap(np.angle(z))
    dph = np.zeros(phase.size)
    dph[half:-half] = phase[2 * half :] - phase[: -2 * half]
    f = dph * (sr / (2.0 * math.pi * 2 * half))
    mag = np.abs(z)
    ref = float(np.median(mag[inner]))
    if ref <= 0.0:
        return np.asarray([])
    valid = inner & (mag > 0.15 * ref)
    f = f[valid]
    if f.size == 0:
        return f
    return f - float(np.median(f))


def two_cluster_centres(values, iters=30):
    """Split element durations into dit/dah clusters and return their centres.

    Taking p25 and p90 instead is biased: the percentiles land inside each cluster rather
    than at its centre, which understated the dah:dit ratio by about 0.27 on a synthetic
    signal with a known 2.77. A 1-D k-means on the two obvious modes is both simple and
    accurate enough for a quality check.
    """
    v = np.asarray(values, dtype=float)
    v = v[v > 0]
    if v.size < 16:
        return None, None
    lo = float(np.percentile(v, 20))
    hi = float(np.percentile(v, 80))
    if hi <= lo:
        return None, None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        a = v[v < mid]
        b = v[v >= mid]
        if a.size < 3 or b.size < 3:
            return None, None
        new_lo, new_hi = float(a.mean()), float(b.mean())
        if abs(new_lo - lo) < 1e-6 and abs(new_hi - hi) < 1e-6:
            lo, hi = new_lo, new_hi
            break
        lo, hi = new_lo, new_hi
    return lo, hi


def measure_complex(iq, sr_in, off, bw_hz, target_fs=None):
    """Every channel statistic for one carrier of an already-loaded capture.

    Split out of analyze_complex so scan() can measure many carriers without reloading
    the file. The keying figures (dit/dah/element count) are here because a signal is only
    a usable calibration reference if it is *one* well-formed Morse transmission.
    """
    if target_fs is None:
        target_fs = max(4.0 * bw_hz, 2000.0)
    z, fs = channelize(iq, sr_in, off, bw_hz, target_fs)

    env = rms_envelope_complex(z, fs, win_ms=4.0)
    mask = keydown_mask_from_env(env, fs)
    snr, p_sig, p_noise = snr_estimate_complex(z, mask)

    # The fading window must be long relative to one element, or the estimator reports the
    # within-element ripple as fades: on a known Rayleigh channel a 10 ms window reads
    # IQR 12.0 dB where 40 ms reads 6.9 dB (theory 6.83). Half a dit is the compromise,
    # clamped so that very fast keying still has enough samples per window.
    starts_env, ends_env = _runs(mask)
    durs_env = (ends_env - starts_env) / fs * 1000.0 if starts_env.size else np.asarray([])
    dit_env, _dah_env = two_cluster_centres(durs_env)
    win_ms = 10.0 if not dit_env else float(min(40.0, max(8.0, 0.5 * dit_env)))

    ts, vs = envelope_series_from_env(rms_envelope_complex(z, fs, win_ms=win_ms), mask, fs)
    env_db = 20.0 * np.log10(np.maximum(vs, 1e-30)) if vs.size else np.asarray([])
    fade_rate, fade_prom, _f, _s = fade_spectrum(ts, vs, fs)
    f_inst = inst_freq_complex(z, mask, fs)

    rayleigh_ratio = None
    if vs.size > 32:
        rayleigh_ratio = round(float(np.median(vs) / np.mean(vs)), 4)

    starts, ends = _runs(mask)
    durs = (ends - starts) / fs * 1000.0 if starts.size else np.asarray([])
    dit_ms, dah_ms = two_cluster_centres(durs)
    ratio = round(dah_ms / dit_ms, 3) if (dit_ms and dah_ms and dit_ms > 0) else None

    st = {
        "channel_sample_rate": round(float(fs), 1),
        "in_channel_peak_ratio_db": (round(channel_purity_db(z, fs), 1)
                                     if z.size else None),
        "iq_bw_hz": round(float(bw_hz), 1),
        "tone_hz": round(float(off), 1),
        "occupied_bandwidth_hz": round(float(bw_hz), 1),
        "duty": round(float(mask.mean()), 3),
        "elements": int(durs.size),
        # What matters for estimating a fading distribution is how much key-down time was
        # observed, not how many elements: 60 s of 5 wpm CW has few elements but plenty of
        # signal to measure.
        "keydown_seconds": round(float(mask.sum()) / float(fs), 2),
        "dit_ms": round(dit_ms, 1) if dit_ms else None,
        "dah_ms": round(dah_ms, 1) if dah_ms else None,
        "dit_dah_ratio": ratio,
        "snr_db": None if math.isnan(snr) else round(snr, 2),
        "signal_power": p_sig,
        "noise_power": p_noise,
        "envelope_db_std": round(float(env_db.std()), 2) if env_db.size else None,
        "envelope_db_p10": round(float(np.percentile(env_db, 10)), 2) if env_db.size else None,
        "envelope_db_p50": round(float(np.percentile(env_db, 50)), 2) if env_db.size else None,
        "envelope_db_p90": round(float(np.percentile(env_db, 90)), 2) if env_db.size else None,
        "envelope_db_span_p10_p90": (
            round(float(np.percentile(env_db, 90) - np.percentile(env_db, 10)), 2)
            if env_db.size else None
        ),
        "envelope_db_iqr": round(_iqr(env_db), 2) if env_db.size else None,
        "rayleigh_median_over_mean": rayleigh_ratio,
        "fade_rate_hz": round(fade_rate, 3),
        "fade_line_prominence": round(fade_prom, 2),
        "inst_freq_iqr_hz": round(_iqr(f_inst), 3) if f_inst.size else None,
        "inst_freq_p95_hz": round(float(np.percentile(np.abs(f_inst), 95)), 3) if f_inst.size else None,
    }
    return st, {"sr": fs, "x": z, "mask": mask, "env_db": env_db, "f_inst": f_inst}


def analyze_complex(path, offset_hz=None, bw_hz=500.0, target_fs=None, raw_fs=None):
    """Channel statistics measured directly on the complex baseband.

    This is stronger than the audio path: with I/Q the fading gain is observable rather
    than inferred, so |z| is literally |g(t)| during key-down and arg(z) is the channel
    phase. The Rayleigh shape of |g| and the Doppler flutter can both be measured
    without a Hilbert transform in the way.
    """
    sr_in, iq = load_iq(path, raw_fs)
    if iq.size < sr_in // 10:
        raise ValueError(f"{path}: too short to analyse")
    if iq.size > 50_000_000:
        print(
            f"warning: {iq.size / 1e6:.0f} Msamples; peak memory is roughly "
            f"{iq.size * 64 / 1e9:.1f} GB. Decimate the capture to <=192 kHz first — a "
            "500 Hz CW channel needs nothing more.",
            file=sys.stderr,
        )
    off = find_carrier(iq, sr_in) if offset_hz is None else float(offset_hz)
    st, detail = measure_complex(iq, sr_in, off, bw_hz, target_fs)
    st["file"] = str(path)
    st["input"] = "iq"
    st["iq_offset_hz"] = round(float(off), 1)
    st["sample_rate"] = int(sr_in)
    st["duration_s"] = round(iq.size / sr_in, 3)
    return st, detail


def is_complex_source(path, raw_fs=None):
    """True when the file looks like I/Q rather than mono audio."""
    p = Path(path)
    if p.suffix.lower() in (".cf32", ".fc32", ".cfile", ".cs16", ".iq"):
        return True
    if p.suffix.lower() == ".npy":
        try:
            return np.iscomplexobj(np.load(p))
        except Exception:
            return False
    if p.suffix.lower() == ".wav":
        try:
            sr, x = wavfile.read(p)
        except Exception:
            return False
        if x.ndim != 2 or x.shape[1] < 2:
            return False
        a, b = x[:, 0].astype(float), x[:, 1].astype(float)
        if a.std() == 0 or b.std() == 0:
            return False
        return abs(float(np.corrcoef(a, b)[0, 1])) <= 0.9
    return False


# ---------------------------------------------------------------------- analyze
def _iqr(v):
    return float(np.percentile(v, 75) - np.percentile(v, 25))


def analyze(path, iq=None, offset_hz=None, bw_hz=500.0, target_fs=None, raw_fs=None):
    """Measure a recording. I/Q sources are detected automatically unless `iq` is set."""
    if iq is None:
        iq = is_complex_source(path, raw_fs)
    if iq:
        return analyze_complex(path, offset_hz=offset_hz, bw_hz=bw_hz,
                               target_fs=target_fs, raw_fs=raw_fs)
    sr, x = load_wav(path)
    if x.size < sr // 10:
        raise ValueError(f"{path}: too short to analyse")
    mask = keydown_mask(x, sr)
    tone = tone_frequency(x, mask, sr)
    snr, p_sig, p_noise = snr_estimate(x, mask, sr)
    bw = occupied_bandwidth(x, sr)

    starts_a, ends_a = _runs(mask)
    durs_a = (ends_a - starts_a) / sr * 1000.0 if starts_a.size else np.asarray([])
    dit_a, dah_a = two_cluster_centres(durs_a)

    dit_win, _dah_win = two_cluster_centres(durs_a)
    win_ms_a = 10.0 if not dit_win else float(min(40.0, max(8.0, 0.5 * dit_win)))
    ts, vs = envelope_series_from_env(rms_envelope(x, sr, win_ms=win_ms_a), mask, sr, win_ms_a)
    env_db = 20.0 * np.log10(np.maximum(vs, 1e-30)) if vs.size else np.asarray([])
    fade_rate, fade_prom, freqs, spec = fade_spectrum(ts, vs, sr)

    f_inst = inst_freq_series(x, mask, sr)

    out = {
        "file": str(path),
        "input": "audio",
        "sample_rate": int(sr),
        "duration_s": round(x.size / sr, 3),
        "tone_hz": round(tone, 1),
        "occupied_bandwidth_hz": round(bw, 1),
        "duty": round(float(mask.mean()), 3),
        "elements": int(durs_a.size),
        "dit_ms": round(dit_a, 1) if dit_a else None,
        "dah_ms": round(dah_a, 1) if dah_a else None,
        "dit_dah_ratio": (round(dah_a / dit_a, 3) if (dit_a and dah_a and dit_a > 0) else None),
        "keydown_seconds": round(float(mask.sum()) / float(sr), 2),
        "keydown_fraction_of_file_bw": round(float(mask.mean()), 3),
        "snr_db": None if math.isnan(snr) else round(snr, 2),
        "signal_power": p_sig,
        "noise_power": p_noise,
        "envelope_db_mean": round(float(env_db.mean()), 2) if env_db.size else None,
        "envelope_db_std": round(float(env_db.std()), 2) if env_db.size else None,
        "envelope_db_p10": round(float(np.percentile(env_db, 10)), 2) if env_db.size else None,
        "envelope_db_p50": round(float(np.percentile(env_db, 50)), 2) if env_db.size else None,
        "envelope_db_p90": round(float(np.percentile(env_db, 90)), 2) if env_db.size else None,
        # p10-p90 is dominated by the noise floor in the fade nulls, so it moves with the
        # analysis bandwidth and must not be used to fit a channel. The IQR is far more
        # stable, and the median/mean ratio is the scale-free Rayleigh check
        # (0.939 for a pure Rayleigh envelope, 6.83 dB IQR).
        "envelope_db_span_p10_p90": (
            round(float(np.percentile(env_db, 90) - np.percentile(env_db, 10)), 2)
            if env_db.size else None
        ),
        "envelope_db_iqr": round(_iqr(env_db), 2) if env_db.size else None,
        "rayleigh_median_over_mean": (round(float(np.median(vs) / np.mean(vs)), 4)
                                      if vs.size else None),
        "fade_rate_hz": round(fade_rate, 3),
        "fade_line_prominence": round(fade_prom, 2),
        # Robust statistics only: a handful of points where the phase unwrap slips by
        # 2*pi inflate the standard deviation by an order of magnitude, so std is useless.
        "inst_freq_iqr_hz": round(_iqr(f_inst), 3) if f_inst.size else None,
        "inst_freq_p95_hz": round(float(np.percentile(np.abs(f_inst), 95)), 3) if f_inst.size else None,
    }
    return out, {"sr": sr, "x": x, "mask": mask, "env_db": env_db, "f_inst": f_inst}


# ---------------------------------------------------------------------- compare
SHARED_KEYS = [
    "tone_hz",
    "occupied_bandwidth_hz",
    "duty",
    "snr_db",
    "envelope_db_std",
    "envelope_db_iqr",
    "rayleigh_median_over_mean",
    "envelope_db_span_p10_p90",
    "fade_rate_hz",
    "fade_line_prominence",
    "inst_freq_iqr_hz",
    "inst_freq_p95_hz",
]


def compare(a_path, b_path, **kw):
    a, ad = analyze(a_path, **kw)
    b, bd = analyze(b_path, **kw)
    rows = []
    for k in SHARED_KEYS:
        va, vb = a.get(k), b.get(k)
        delta = None
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            delta = round(float(vb) - float(va), 3)
        rows.append({"metric": k, "a": va, "b": vb, "delta_b_minus_a": delta})

    # Distribution tests on the two quantities that carry the channel signature.
    tests = {}
    if ad["env_db"].size > 8 and bd["env_db"].size > 8:
        # Absolute level depends on recorder gain and is meaningless across files, so the
        # distributions are compared after removing each median. Otherwise KS reports a
        # perfect separation for two recordings that differ only in overall volume.
        ea = ad["env_db"] - float(np.median(ad["env_db"]))
        eb = bd["env_db"] - float(np.median(bd["env_db"]))
        d, p = stats.ks_2samp(ea, eb)
        tests["envelope_db_ks"] = {"statistic": round(float(d), 4), "p_value": round(float(p), 6)}
    if ad["f_inst"].size > 256 and bd["f_inst"].size > 256:
        d, p = stats.ks_2samp(ad["f_inst"], bd["f_inst"])
        tests["inst_freq_ks"] = {"statistic": round(float(d), 4), "p_value": round(float(p), 6)}

    notes = []
    if a.get("envelope_db_span_p10_p90") is not None:
        notes.append(
            "envelope_db_span_p10_p90 is noise-floor limited in the fade nulls and moves "
            "with the analysis bandwidth; do not fit a channel to it. Use "
            "rayleigh_median_over_mean (0.939 = pure Rayleigh) and envelope_db_iqr "
            "(6.83 dB = pure Rayleigh), which are stable across bandwidths."
        )
    if a.get("input") != b.get("input"):
        notes.append(
            f"input types differ ({a.get('input')} vs {b.get('input')}): S:N and the "
            "envelope statistics are measured in different bandwidths and are not "
            "directly comparable. Fade rate and duty are."
        )
    if a.get("input") == "iq" and a.get("iq_bw_hz") != b.get("iq_bw_hz"):
        notes.append("I/Q passbands differ; S:N is not comparable.")

    return {"a": a, "b": b, "metrics": rows, "tests": tests, "notes": notes}


def print_compare(res):
    a, b = res["a"], res["b"]
    print(f"A = {a['file']}")
    print(f"B = {b['file']}\n")
    print(f"{'metric':<26} {'A':>12} {'B':>12} {'B-A':>10}")
    print("-" * 64)
    for r in res["metrics"]:
        def fmt(v):
            if v is None:
                return "-"
            if isinstance(v, float):
                return f"{v:.3f}"
            return str(v)
        print(f"{r['metric']:<26} {fmt(r['a']):>12} {fmt(r['b']):>12} {fmt(r['delta_b_minus_a']):>10}")
    if res["tests"]:
        print()
        for name, t in res["tests"].items():
            # With tens of thousands of samples the KS p-value is 0 for any difference at
            # all, so it says nothing. Judge on the effect size instead: D is the largest
            # vertical gap between the two empirical distributions.
            verdict = "compatible" if t["statistic"] < 0.15 else "DIFFERENT"
            print(f"{name}: D={t['statistic']:.4f} (p={t['p_value']:.3g}, "
                  f"uninformative at this n)  -> {verdict}")
    else:
        print("\n(too little data for two-sample tests)")
    for note in res.get("notes", []):
        print(f"\nNOTE: {note}")


# -------------------------------------------------------------------------- fit
def _pct(values, q):
    return float(np.percentile(values, q)) if values else float("nan")


# Flutter (IQR, Hz) measured on a 200 Hz channel with no flat fade is
# 0.20 / 0.37 / 0.72 / 1.45 for MPG / MPM / MPP / MPD. Boundaries are the geometric
# midpoints, widened a little because the estimate itself moves by ~0.1 Hz on short or
# noisy captures.
FLUTTER_BOUNDARIES = (0.32, 0.55, 1.15)


def channel_for_flutter(flutter):
    """Pick the CCIR channel whose Doppler spread best matches a measured flutter."""
    if flutter is None:
        return "MPP", 1.0
    a, b, c = FLUTTER_BOUNDARIES
    if flutter < a:
        return "MPG", 0.1
    if flutter < b:
        return "MPM", 0.5
    if flutter < c:
        return "MPP", 1.0
    return "MPD", 2.0


def fit_capture(path, offset_hz=None, bw_hz=200.0, segments=6, raw_fs=None, seed=1):
    """Measure a capture in segments and propose `gen_cw` settings that would match it.

    This is a starting point, not an answer. The inverse problem has more unknowns than
    observables, so one round of `compare` will always be needed; what this removes is the
    guesswork about *where* to start.
    """
    sr, data = load_iq(path, raw_fs)
    if offset_hz is None:
        offset_hz = find_carrier(data, sr)
    seg_len = int(data.size // max(1, segments))
    rows = []
    for i in range(segments):
        chunk = data[i * seg_len:(i + 1) * seg_len]
        if chunk.size < sr // 4:
            continue
        st, _ = measure_complex(chunk, sr, offset_hz, bw_hz)
        if st.get("dit_dah_ratio"):
            rows.append(st)
    if len(rows) < 2:
        raise ValueError(f"{path}: only {len(rows)} usable segments; need at least 2")

    def agg(key):
        v = [r[key] for r in rows if r[key] is not None]
        if not v:
            return None, None
        return float(np.mean(v)), float(np.std(v))

    targets = {k: agg(k) for k in
               ("snr_db", "rayleigh_median_over_mean", "envelope_db_iqr", "fade_rate_hz",
                "inst_freq_iqr_hz", "dit_dah_ratio", "dit_ms", "duty", "keydown_seconds")}

    dit_ms = targets["dit_ms"][0]
    wpm = round(1200.0 / dit_ms, 1) if dit_ms else 18.0
    ratio = targets["dit_dah_ratio"][0] or 3.0
    flutter = targets["inst_freq_iqr_hz"][0] or 1.0

    channel, doppler = channel_for_flutter(flutter)

    # gen_cw's --snr is quoted in a 3 kHz band; the capture is measured in `bw_hz`.
    # Measured gain for 3 kHz -> 200 Hz is about 7.5 dB on this toolchain, so invert that.
    snr_measured = targets["snr_db"][0]
    band_gain = {200.0: 7.5, 400.0: 5.0, 500.0: 4.0, 800.0: 2.5}.get(float(bw_hz), 7.5)
    snr_gen = round(snr_measured - band_gain, 1) if snr_measured is not None else 15.0

    fade_rate = targets["fade_rate_hz"][0]
    fade_rate = round(fade_rate, 3) if fade_rate and fade_rate > 0.02 else None

    cmd = ["./build/gen_cw", "--text", "CQ CQ DE JA1ABC K", "--repeat", "15",
           "--wpm", f"{wpm}", "--dah-ratio", f"{ratio:.2f}",
           "--channel", channel, "--snr", f"{snr_gen}", "--bw", "3000",
           "--fade-depth", "0"]
    if fade_rate:
        cmd += ["--fade", f"{fade_rate}"]
    cmd += ["--seed", str(seed), "--out", "match.wav", "--labels", "match.json"]

    return {
        "file": str(path),
        "offset_hz": float(offset_hz),
        "bandwidth_hz": float(bw_hz),
        "segments_used": len(rows),
        "targets": {k: {"mean": v[0], "sd": v[1]} for k, v in targets.items()},
        "suggested_command": " ".join(cmd),
        "notes": [
            "--snr is a starting value derived from a measured bandwidth correction; expect "
            "to adjust it by 1-3 dB after the first compare.",
            "dah-ratio reproduces the *measured* ratio, which the estimator reads about 0.3 "
            "low; if the operator is likely exact, consider 3.00.",
            "fade-depth was left at 0: the fade sweep and the multipath fading are not "
            "separable from the envelope alone. Add it only if compare shows the IQR short.",
        ],
    }


def print_fit(res):
    t = res["targets"]
    print(f"{res['file']}")
    print(f"  offset {res['offset_hz']:.0f} Hz, {res['bandwidth_hz']:.0f} Hz channel, "
          f"{res['segments_used']} segments\n")
    print(f"{'metric':<28}{'mean':>10}{'sd':>9}")
    print("-" * 47)
    for k in ("snr_db", "rayleigh_median_over_mean", "envelope_db_iqr", "fade_rate_hz",
              "inst_freq_iqr_hz", "dit_dah_ratio", "dit_ms", "duty", "keydown_seconds"):
        m, sd = t[k]["mean"], t[k]["sd"]
        if m is None:
            continue
        print(f"{k:<28}{m:>10.3f}{sd:>9.3f}")
    print(f"\n  Rayleigh reference: median/mean {RAYLEIGH_RATIO}, IQR {RAYLEIGH_IQR_DB} dB")
    print(f"\nsuggested start:\n  {res['suggested_command']}")
    print("\nthen iterate:")
    print("  ./build/gen_cw ... --out match.wav   # run the command above")
    print("  python3 calibrate.py compare capture.wav match.wav --iq --offset-hz <off> --bw-hz <bw>")
    print("  adjust --snr first, then the channel, then --fade-depth\n")
    for n in res["notes"]:
        print(f"  note: {n}")


# ------------------------------------------------------------------------- scan
RAYLEIGH_RATIO = 0.9394   # median/mean of a Rayleigh envelope
RAYLEIGH_IQR_DB = 6.83    # p25-p75 of a Rayleigh envelope in dB (p10-p90 is 13.39)


def find_carriers(iq, sr, min_prominence=8.0, min_separation_hz=30.0, guard_hz=200.0):
    """Carriers in a whole capture, strongest first."""
    nperseg = int(min(65536, max(1024, iq.size // 8)))
    f, pxx = signal.welch(iq, sr, nperseg=nperseg, return_onesided=False)
    order = np.argsort(f)
    f, pxx = f[order], pxx[order]
    db = 10.0 * np.log10(pxx + 1e-30)
    df = f[1] - f[0]
    pk, _ = signal.find_peaks(db, prominence=min_prominence,
                              distance=max(1, int(round(min_separation_hz / df))))
    pk = pk[np.abs(f[pk]) > guard_hz]
    if pk.size == 0:
        return np.asarray([]), np.asarray([]), df
    order = np.argsort(db[pk])[::-1]
    return f[pk][order], db[pk][order], df


def nearest_neighbour_hz(freqs, off, levels=None, rel_db=40.0):
    """Distance to the closest carrier that could actually interfere.

    Peaks more than `rel_db` below the strongest carrier are ignored: on a quiet band the
    peak finder returns plenty of noise maxima, and counting those would make every signal
    look crowded.
    """
    if levels is None:
        others = [abs(off - v) for v in freqs if v != off]
    else:
        top = float(np.max(levels))
        others = [abs(off - v) for v, lv in zip(freqs, levels)
                  if v != off and lv >= top - rel_db]
    return min(others) if others else float("inf")


def quality_score(st, stability=None, neighbour_hz=None):
    """Can this carrier be used as a single-signal calibration reference?

    Returns (score out of 110, notes). Thresholds are deliberately loose: off-air signals
    are never textbook, and the purpose is to reject the clearly unusable, not to grade
    the good ones finely. A high score is not a proof of correctness -- it means nothing
    obvious is wrong.
    """
    score = 0
    notes = []

    r = st.get("dit_dah_ratio")
    if r is None:
        notes.append("no keying found")
    else:
        d = abs(r - 3.0)
        if d <= 0.30:
            score += 30
        elif d <= 0.60:
            score += 15
        else:
            notes.append(f"dit:dah {r:.2f} (Morse is 3.00)")

    duty = st.get("duty") or 0.0
    if 0.40 <= duty <= 0.70:
        score += 15
    else:
        notes.append(f"duty {duty:.2f}")

    kd = st.get("keydown_seconds")
    if kd is not None:
        if kd >= 15.0:
            score += 15
        elif kd >= 8.0:
            score += 8
        else:
            notes.append(f"only {kd:.0f} s of key-down")
    else:
        el = st.get("elements") or 0
        if el >= 200:
            score += 15
        elif el >= 80:
            score += 8
        else:
            notes.append(f"only {el} elements")

    snr = st.get("snr_db")
    if snr is None:
        notes.append("no S:N measurable")
    elif snr >= 20:
        score += 20
    elif snr >= 15:
        score += 10
    else:
        notes.append(f"S:N {snr:.1f} dB")

    rr = st.get("rayleigh_median_over_mean")
    if rr is None:
        notes.append("no envelope statistics")
    else:
        d = abs(rr - RAYLEIGH_RATIO)
        if d <= 0.06:
            score += 20
        elif d <= 0.15:
            score += 10
        else:
            notes.append(f"median/mean {rr:.2f} vs Rayleigh {RAYLEIGH_RATIO:.2f}")

    bw = st.get("iq_bw_hz") or 200.0
    separable = True
    if neighbour_hz is not None and neighbour_hz < 4.0 * bw:
        notes.append(f"nearest carrier only {neighbour_hz:.0f} Hz away")
    iqr = st.get("envelope_db_iqr")
    if iqr is not None and iqr > 16.0:
        # A Rayleigh envelope has IQR 6.83 dB; the measurement itself adds a couple of dB
        # (a pure-Rayleigh synthetic reads about 9 dB), so 16 dB is a loose ceiling that
        # only clearly non-Rayleigh stimuli exceed.
        notes.append(f"envelope IQR {iqr:.1f} dB (Rayleigh is 8.7): not a single Rayleigh channel")
        separable = False

    purity = st.get("in_channel_peak_ratio_db")
    if purity is not None and purity > -15.0:
        # Something else inside the channel is within 15 dB of the carrier. Filtering
        # narrower would damage the keying before it removed this.
        notes.append(f"competing peak {purity:.0f} dB inside the channel")
        separable = False

    if stability is not None:
        if stability <= 0.05:
            score += 10
        else:
            notes.append(f"median/mean moves {stability:.2f} with bandwidth")

    if not separable:
        score = min(score, 20)
    return score, notes, separable


def scan(path, bw_hz=200.0, raw_fs=None, center_mhz=None, max_carriers=25,
         check_stability=True):
    """Measure every carrier in a capture and rank them as calibration references."""
    sr, data = load_iq(path, raw_fs)
    freqs, levels, _df = find_carriers(data, sr)
    rows = []
    for off in list(freqs)[:max_carriers]:
        st, _ = measure_complex(data, sr, off, bw_hz)
        if (st.get("elements") or 0) < 20:
            continue
        nb = nearest_neighbour_hz(list(freqs), off, levels)
        stability = None
        if check_stability:
            st2, _ = measure_complex(data, sr, off, bw_hz * 2.0)
            a = st.get("rayleigh_median_over_mean")
            b = st2.get("rayleigh_median_over_mean")
            if a is not None and b is not None:
                stability = abs(a - b)
        score, notes, separable = quality_score(st, stability, nb)
        row = {
            "offset_hz": round(float(off), 1),
            "score": score,
            "separable": separable,
            "dit_dah_ratio": st.get("dit_dah_ratio"),
            "dit_ms": st.get("dit_ms"),
            "wpm_est": round(1200.0 / st["dit_ms"], 1) if st.get("dit_ms") else None,
            "duty": st.get("duty"),
            "elements": st.get("elements"),
            "keydown_seconds": st.get("keydown_seconds"),
            "snr_db": st.get("snr_db"),
            "rayleigh_median_over_mean": st.get("rayleigh_median_over_mean"),
            "envelope_db_iqr": st.get("envelope_db_iqr"),
            "fade_rate_hz": st.get("fade_rate_hz"),
            "inst_freq_iqr_hz": st.get("inst_freq_iqr_hz"),
            "neighbour_hz": round(float(nb), 0) if nb != float("inf") else None,
            "in_channel_peak_ratio_db": st.get("in_channel_peak_ratio_db"),
            "stability": round(stability, 3) if stability is not None else None,
            "notes": notes,
        }
        if center_mhz is not None:
            row["mhz"] = round(center_mhz + off / 1e6, 6)
        rows.append(row)
    rows.sort(key=lambda r: -r["score"])
    return {"file": str(path), "bandwidth_hz": bw_hz, "carriers": rows}


def print_scan(res):
    print(f"{res['file']}")
    print(f"channel bandwidth {res['bandwidth_hz']:.0f} Hz\n")
    hdr = f"{'MHz':>12}" if "mhz" in (res['carriers'][0] if res['carriers'] else {}) else f"{'off Hz':>10}"
    print(f"{'MHz':>12}{'off Hz':>9}{'score':>7}{'ratio':>7}{'wpm':>6}{'snr':>6}"
          f"{'rayl':>7}{'iqr':>6}{'nb Hz':>7}  notes")
    print("-" * 108)
    for r in res["carriers"]:
        mhz = f"{r['mhz']:.6f}" if "mhz" in r else ""
        ratio = f"{r['dit_dah_ratio']:.2f}" if r["dit_dah_ratio"] else "-"
        wpm = f"{r['wpm_est']:.1f}" if r["wpm_est"] else "-"
        snr = f"{r['snr_db']:.1f}" if r["snr_db"] is not None else "-"
        rayl = f"{r['rayleigh_median_over_mean']:.3f}" if r["rayleigh_median_over_mean"] else "-"
        iqr = f"{r['envelope_db_iqr']:.1f}" if r["envelope_db_iqr"] is not None else "-"
        nb = f"{r['neighbour_hz']:.0f}" if r["neighbour_hz"] is not None else "-"
        notes = "; ".join(r["notes"]) if r["notes"] else "clean"
        if not r.get("separable", True):
            notes = "UNUSABLE: " + notes
        print(f"{mhz:>12}{r['offset_hz']:>9.0f}{r['score']:>7}{ratio:>7}{wpm:>6}{snr:>6}"
              f"{rayl:>7}{iqr:>6}{nb:>7}  {notes}")
    print(f"\nRayleigh reference: median/mean {RAYLEIGH_RATIO}, IQR {RAYLEIGH_IQR_DB} dB")


# -------------------------------------------------------------------- roundtrip
# S:N is checked where nothing else moves; fading is checked at high S:N so that the
# envelope of a deep fade is not clipped by the noise floor.
ROUNDTRIP_CASES = [
    ("wgn-snr10", ["--channel", "WGN", "--snr", "10"], "snr"),
    ("wgn-snr0", ["--channel", "WGN", "--snr", "0"], "snr"),
    ("wgn-snr25", ["--channel", "WGN", "--snr", "25"], "snr"),
    ("wgn-snr30", ["--channel", "WGN", "--snr", "30"], "snr"),
    ("mpd-snr30", ["--channel", "MPD", "--snr", "30"], "snr"),
    ("fade12-0.5", ["--channel", "WGN", "--snr", "35", "--fade-depth", "12", "--fade", "0.5"], "fade"),
    ("fade15-1.0", ["--channel", "WGN", "--snr", "35", "--fade-depth", "15", "--fade", "1.0"], "fade"),
    ("fade20-0.2", ["--channel", "WGN", "--snr", "35", "--fade-depth", "20", "--fade", "0.2"], "fade"),
    ("mpp-doppler1", ["--channel", "MPP", "--snr", "30", "--fade-depth", "10", "--fade", "0.5"], "fade"),
    ("mpg-doppler01", ["--channel", "MPG", "--snr", "30", "--fade-depth", "10", "--fade", "0.5"], "fade"),
]


def fade_average_penalty_db(depth_db):
    """How much a cosine S:N sweep lowers the mean power relative to its peak.

    --snr is defined at a fade peak, but any recording averages over the sweep, so the
    measured S:N of a faded file must come out lower by exactly this much.
    """
    if depth_db <= 0:
        return 0.0
    theta = np.linspace(0.0, 2.0 * math.pi, 8192, endpoint=False)
    d = -0.5 * depth_db * (1.0 - np.cos(theta))
    return float(10.0 * math.log10(np.mean(10.0 ** (d / 10.0))))


def roundtrip(workdir):
    if not GEN.is_file():
        raise SystemExit(f"{GEN} not built; run make -C {HERE}")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    text = "CQ CQ DE JA1ABC K " * 6
    print(f"{'case':<13} {'kind':<5} {'want':>8} {'predicted':>10} {'measured':>9} {'err':>7}")
    print("-" * 60)
    rows = []
    for name, extra, kind in ROUNDTRIP_CASES:
        wav = workdir / f"rt_{name}.wav"
        labels = workdir / f"rt_{name}.json"
        cmd = [str(GEN), "--text", text, "--out", str(wav), "--labels", str(labels),
               "--quiet", "--seed", "11"] + extra
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(f"gen_cw failed for {name}: {proc.stderr}")
        lab = json.loads(labels.read_text())
        stats_out, _ = analyze(wav)
        row = {"case": name, "kind": kind, "file": str(wav),
               "channel": lab["channel"]["mode"]}

        if kind == "snr":
            want = lab["snr"]["requested_audio_band_db"]
            pred = want
            got = stats_out["snr_db"]
        else:
            want = lab["channel"]["fade_rate_hz"]
            pred = want
            got = stats_out["fade_rate_hz"]

        err = None if (got is None or pred is None) else round(got - pred, 3)
        print(f"{name:<13} {kind:<5} {want:>8.2f} {pred:>10.2f} "
              f"{'n/a' if got is None else f'{got:.2f}':>9} {'n/a' if err is None else f'{err:+.2f}':>7}")

        row.update({
            "want": want, "predicted": pred, "measured": got, "error": err,
            "requested_snr_db": lab["snr"]["requested_audio_band_db"],
            "measured_snr_db": stats_out["snr_db"],
            "fade_rate_hz": stats_out["fade_rate_hz"],
            "fade_depth_db": lab["channel"]["fade_depth_db"],
            "envelope_span_db": stats_out["envelope_db_span_p10_p90"],
            "inst_freq_iqr_hz": stats_out["inst_freq_iqr_hz"],
            "doppler_spread_hz": lab["channel"]["doppler_spread_hz"],
        })
        rows.append(row)

    print()
    print("S:N after the fade average (what a recording must show, not the peak value):")
    for r in rows:
        pen = fade_average_penalty_db(r["fade_depth_db"])
        if pen != 0.0 and r["measured_snr_db"] is not None:
            pred = r["requested_snr_db"] + pen
            print(f"  {r['case']:<13} peak {r['requested_snr_db']:>6.2f} "
                  f"penalty {pen:>6.2f} -> predicted {pred:>6.2f}  "
                  f"measured {r['measured_snr_db']:>6.2f}")

    # inst_freq_iqr is a relative indicator: the phase estimate has a noise floor set by
    # S:N, so it is only meaningful next to a no-fading recording at the same S:N.
    base = next((r["inst_freq_iqr_hz"] for r in rows
                 if r["doppler_spread_hz"] == 0 and r["case"] == "wgn-snr30"), None)
    print()
    print("Doppler spread versus the flutter it should produce (relative to a no-fading")
    print("recording at the same S:N -- this is an indicator, not an absolute estimate):")
    if base is not None:
        print(f"  {'wgn-snr30':<13} baseline (no fading)      -> inst_freq_iqr={base} Hz")
    for r in rows:
        if r["doppler_spread_hz"] and r["inst_freq_iqr_hz"] is not None:
            extra = "" if base is None else f"  (x{r['inst_freq_iqr_hz'] / base:.2f} baseline)"
            print(f"  {r['case']:<13} 2sigma={r['doppler_spread_hz']:<5} "
                  f"-> inst_freq_iqr={r['inst_freq_iqr_hz']} Hz{extra}")
    return rows


# ------------------------------------------------------------------------- main
def input_kw(args):
    iq = True if getattr(args, "iq", False) else (False if getattr(args, "mono", False) else None)
    return {
        "iq": iq,
        "offset_hz": getattr(args, "offset_hz", None),
        "bw_hz": getattr(args, "bw_hz", 500.0),
        "target_fs": getattr(args, "target_fs", None),
        "raw_fs": getattr(args, "fs", None),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_input_args(parser):
        parser.add_argument("--iq", action="store_true",
                            help="force I/Q input (auto-detected by default)")
        parser.add_argument("--mono", action="store_true",
                            help="force mono audio input (auto-detected by default)")
        parser.add_argument("--offset-hz", type=float, default=None,
                            help="I/Q carrier offset from centre; default: strongest peak")
        parser.add_argument("--bw-hz", type=float, default=500.0,
                            help="I/Q channel passband width (default 500)")
        parser.add_argument("--target-fs", type=float, default=None,
                            help="I/Q sample rate after decimation (default max(4*bw, 2000))")
        parser.add_argument("--fs", type=float, default=None,
                            help="sample rate for raw .cf32/.cs16/.npy input")

    a = sub.add_parser("analyze", help="channel statistics of one recording")
    a.add_argument("wav")
    a.add_argument("--json")
    add_input_args(a)

    c = sub.add_parser("compare", help="compare a real recording against a synthesised one")
    c.add_argument("real")
    c.add_argument("synth")
    c.add_argument("--json")
    add_input_args(c)

    ft = sub.add_parser("fit", help="measure a capture and propose gen_cw settings")
    ft.add_argument("file")
    ft.add_argument("--offset-hz", type=float, default=None)
    ft.add_argument("--bw-hz", type=float, default=200.0)
    ft.add_argument("--segments", type=int, default=6)
    ft.add_argument("--fs", type=float, default=None)
    ft.add_argument("--json")

    sc = sub.add_parser("scan", help="rank every carrier in a capture as a reference")
    sc.add_argument("file")
    sc.add_argument("--bw-hz", type=float, default=200.0, help="channel width (default 200)")
    sc.add_argument("--center-mhz", type=float, default=None,
                    help="capture centre frequency, for printing absolute MHz")
    sc.add_argument("--max-carriers", type=int, default=25)
    sc.add_argument("--no-stability", action="store_true",
                    help="skip the bandwidth-stability check (about 2x faster)")
    sc.add_argument("--fs", type=float, default=None)
    sc.add_argument("--json")

    r = sub.add_parser("roundtrip", help="closed-loop check of the estimators")
    r.add_argument("--workdir")
    r.add_argument("--json")

    args = ap.parse_args(argv)
    try:
        return _dispatch(args)
    except (ValueError, OSError, KeyError) as e:
        # A CLI must not print a traceback for a bad input file. Every subcommand funnels
        # its failures through here.
        print(f"error: {e}", file=sys.stderr)
        return 1


def _dispatch(args):

    if args.cmd == "analyze":
        stats_out, _ = analyze(args.wav, **input_kw(args))
        print(json.dumps(stats_out, indent=2))
        if args.json:
            Path(args.json).write_text(json.dumps(stats_out, indent=2))
        return 0

    if args.cmd == "compare":
        res = compare(args.real, args.synth, **input_kw(args))
        print_compare(res)
        if args.json:
            Path(args.json).write_text(json.dumps(res, indent=2))
        return 0

    if args.cmd == "fit":
        res = fit_capture(args.file, offset_hz=args.offset_hz, bw_hz=args.bw_hz,
                          segments=args.segments, raw_fs=args.fs)
        print_fit(res)
        if args.json:
            Path(args.json).write_text(json.dumps(res, indent=2))
        return 0

    if args.cmd == "scan":
        res = scan(args.file, bw_hz=args.bw_hz, raw_fs=args.fs,
                   center_mhz=args.center_mhz, max_carriers=args.max_carriers,
                   check_stability=not args.no_stability)
        print_scan(res)
        if args.json:
            Path(args.json).write_text(json.dumps(res, indent=2))
        return 0

    if args.cmd == "roundtrip":
        wd = args.workdir
        if wd:
            rows = roundtrip(wd)
        else:
            with tempfile.TemporaryDirectory(prefix="cwsim-rt-") as td:
                rows = roundtrip(td)
        if args.json:
            Path(args.json).write_text(json.dumps(rows, indent=2))
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())

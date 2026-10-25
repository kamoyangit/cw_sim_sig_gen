"""Statistical regression tests for tools/cwsim (labelled CW channel generator).

These are the tests that make the generator's claims checkable: they verify the
channel statistics against the Watterson/CCIR model the IONOS firmware implements,
and they verify the ground-truth labels against the rendered audio.

Run with:  make -C tools/cwsim test
"""

import json
import math
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

CWSIM = Path(__file__).resolve().parents[1]
BIN = CWSIM / "build" / "gen_cw"

MESSAGE = "CQ CQ DE JA1ABC K"


def require_bin():
    if not BIN.is_file():
        raise unittest.SkipTest(f"{BIN} not built; run make -C tools/cwsim")


def read_wav_float32(path):
    """Minimal float32 WAV reader so the tests do not depend on scipy."""
    import struct

    raw = Path(path).read_bytes()
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    pos = 12
    fmt = None
    data = None
    while pos + 8 <= len(raw):
        cid = raw[pos : pos + 4]
        size = struct.unpack("<I", raw[pos + 4 : pos + 8])[0]
        body = raw[pos + 8 : pos + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif cid == b"data":
            data = body
        pos += 8 + size + (size & 1)
    if fmt is None or data is None:
        raise ValueError("missing fmt or data chunk")
    tag, channels, rate, _byte_rate, _align, bits = fmt
    if tag != 3 or bits != 32:
        raise ValueError(f"expected float32 WAV, got tag={tag} bits={bits}")
    return rate, np.frombuffer(data, dtype="<f4").astype(np.float64).reshape(-1, channels)[:, 0]


class CwSimTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_bin()
        cls._tmp = tempfile.TemporaryDirectory(prefix="cwsim-test-")
        cls.tmp = Path(cls._tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    # -- helpers ---------------------------------------------------------
    def run_gen(self, name, extra, expect_ok=True):
        wav = self.tmp / f"{name}.wav"
        labels = self.tmp / f"{name}.json"
        cmd = [str(BIN), "--out", str(wav), "--labels", str(labels), "--quiet"] + extra
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if expect_ok:
            self.assertEqual(proc.returncode, 0, f"gen_cw failed: {proc.stderr}")
            rate, audio = read_wav_float32(wav)
            return rate, audio, json.loads(labels.read_text())
        return proc

    @staticmethod
    def keydown_mask(labels, n):
        mask = np.zeros(n, dtype=bool)
        for s in labels["symbols"]:
            if s["kind"] in ("dit", "dah"):
                mask[s["start"] : s["end"]] = True
        return mask

    @staticmethod
    def measure_snr_db(audio, labels):
        """Independent S:N estimate: noise from gaps, signal = key-down minus noise."""
        mask = CwSimTest.keydown_mask(labels, len(audio))
        p_noise = float(np.mean(audio[~mask] ** 2))
        p_kedown = float(np.mean(audio[mask] ** 2))
        p_signal = max(p_kedown - p_noise, 1e-30)
        return 10.0 * math.log10(p_signal / p_noise)

    # -- CLI surface -----------------------------------------------------
    def test_lists_cw_delay_presets(self):
        proc = subprocess.run([str(BIN), "--list-delay-presets"], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        names = [line.split(":")[0] for line in proc.stdout.strip().splitlines()]
        self.assertIn("ccir-mpp", names)
        self.assertIn("cw-long", names)
        # CW presets must reach well past the 4 ms data-mode maximum.
        longest = max(
            len(line.split(":")[1].split(",")) for line in proc.stdout.strip().splitlines()
        )
        self.assertGreaterEqual(longest, 4)

    def test_rejects_missing_output(self):
        proc = subprocess.run([str(BIN), "--text", "E"], capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)

    # -- labels ----------------------------------------------------------
    def test_labels_align_with_audio(self):
        rate, audio, labels = self.run_gen("align", ["--text", MESSAGE, "--channel", "WGN", "--snr", "10"])
        n = len(audio)
        self.assertGreater(n, 0)
        self.assertEqual(rate, 44100)
        self.assertEqual(labels["text"], MESSAGE)

        chars = labels["characters"]
        self.assertEqual(len(chars), len(MESSAGE.replace(" ", "")))
        self.assertEqual("".join(c["char"] for c in chars), MESSAGE.replace(" ", ""))

        prev_end = -1
        for c in chars:
            self.assertGreaterEqual(c["start"], prev_end)
            self.assertLess(c["start"], c["end"])
            self.assertLessEqual(c["end"], n, "label runs past the end of the audio")
            prev_end = c["end"]

        # The last element must be inside the file, i.e. the tail was padded.
        self.assertLess(labels["symbols"][-1]["end"], n)

        # Key-down samples must be clearly louder than the gaps.
        mask = self.keydown_mask(labels, n)
        self.assertAlmostEqual(float(np.mean(mask)), 0.5, delta=0.2)
        self.assertGreater(np.sqrt(np.mean(audio[mask] ** 2)), 3.0 * np.sqrt(np.mean(audio[~mask] ** 2)))

    def test_wgn_stays_distinct_from_mpp(self):
        """WGN is "no fading", MPP is 1 Hz of it. They must never be the same thing."""
        _r, _a, wgn = self.run_gen("wgn_plain", ["--text", MESSAGE, "--channel", "WGN",
                                                 "--snr", "20"])
        _r, _a, mpp = self.run_gen("mpp_plain", ["--text", MESSAGE, "--channel", "MPP",
                                                 "--snr", "20"])
        self.assertEqual(wgn["channel"]["doppler_spread_hz"], 0.0)
        self.assertEqual(wgn["channel"]["tick_rate_hz"], 0.0)
        self.assertEqual(wgn["channel"]["paths"], 1)
        self.assertEqual(mpp["channel"]["doppler_spread_hz"], 1.0)
        self.assertEqual(mpp["channel"]["tick_rate_hz"], 64.0)

    def test_wgn_with_explicit_delays_is_static_multipath(self):
        """Regression: --delays used to raise `paths`, which then triggered a generic
        "multipath needs Doppler" correction and silently turned WGN into MPP."""
        _r, _a, lab = self.run_gen("wgn_delays", ["--text", MESSAGE, "--channel", "WGN",
                                                  "--delays", "0,2", "--snr", "20"])
        ch = lab["channel"]
        self.assertEqual(ch["mode"], "WGN")
        self.assertEqual(ch["paths"], 2)
        self.assertEqual(ch["doppler_spread_hz"], 0.0, "WGN must not acquire a Doppler spread")
        self.assertEqual(ch["tick_rate_hz"], 0.0)

    def test_per_path_doppler_and_scalar_summary_agree(self):
        _r, _a, lab = self.run_gen("pdsum", ["--text", MESSAGE, "--channel", "WGN",
                                             "--path-doppler", "0.2,3", "--snr", "20"])
        ch = lab["channel"]
        self.assertEqual(ch["path_doppler_hz"], [0.2, 3.0])
        self.assertAlmostEqual(ch["doppler_spread_hz"], 0.2, delta=1e-9)
        self.assertAlmostEqual(ch["tick_rate_hz"], 12.8, delta=1e-9)  # 64 x 0.2

    def test_contradictory_options_are_rejected(self):
        """WGN has no fading, so a Doppler spread is a contradiction. This used to raise an
        uncaught exception instead of reporting anything."""
        proc = self.run_gen("wgn_dop", ["--text", "E", "--channel", "WGN", "--doppler", "1"],
                            expect_ok=False)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("WGN", proc.stderr)

    def test_ineffective_options_warn_and_do_not_leak_into_labels(self):
        _r, _a, lab = self.run_gen("wgn_k", ["--text", "E", "--channel", "WGN",
                                             "--rician-k", "4", "--snr", "20"])
        self.assertEqual(lab["channel"]["rician_k"], 0.0)

        _r, _a, lab2 = self.run_gen("mpp_fadeonly", ["--text", "E", "--channel", "MPP",
                                                     "--fade-depth", "10", "--snr", "20"])
        self.assertEqual(lab2["channel"]["fade_depth_db"], 0.0)

    def test_explicit_doppler_zero_gives_static_multipath(self):
        """An explicit --doppler 0 must survive; a generic correction used to force it back
        to 1 Hz, silently discarding the request."""
        _r, _a, lab = self.run_gen("mpp_dop0", ["--text", "E", "--channel", "MPP",
                                                "--doppler", "0", "--snr", "20"])
        self.assertEqual(lab["channel"]["doppler_spread_hz"], 0.0)
        self.assertEqual(lab["channel"]["tick_rate_hz"], 0.0)
        self.assertEqual(lab["channel"]["paths"], 2)

    def test_keying_timing_follows_wpm(self):
        rate, _audio, labels = self.run_gen("timing", ["--text", "PARIS", "--wpm", "20", "--snr", "40"])
        dit = [s for s in labels["symbols"] if s["kind"] == "dit"]
        dah = [s for s in labels["symbols"] if s["kind"] == "dah"]
        self.assertTrue(dit and dah)
        dit_len = np.mean([s["end"] - s["start"] for s in dit])
        dah_len = np.mean([s["end"] - s["start"] for s in dah])
        expected_dit = 1.2 / 20.0 * rate
        self.assertAlmostEqual(dit_len / expected_dit, 1.0, delta=0.03)
        self.assertAlmostEqual(dah_len / dit_len, 3.0, delta=0.05)

        # Element and gap units must add up to the keying span at the unit duration.
        total_units = sum(
            {"dit": 1, "dah": 3, "element_gap": 1, "char_gap": 3, "word_gap": 7}[s["kind"]]
            for s in labels["symbols"]
        )
        span = labels["symbols"][-1]["end"] - labels["symbols"][0]["start"]
        self.assertAlmostEqual(span / (total_units * expected_dit), 1.0, delta=0.02)

    def test_min_gain_db_tracks_the_rendered_fade(self):
        """min_gain_db must describe the composite channel of the rendered signal, not a
        single fading path: two Rayleigh paths add to something far shallower."""
        _rate, audio, labels = self.run_gen(
            "mingain", ["--text", MESSAGE, "--repeat", "2", "--channel", "MPP", "--snr", "45"]
        )
        mask = self.keydown_mask(labels, len(audio))
        ref = math.sqrt(float(np.mean(audio[mask] ** 2)))
        win, hop = 441, 220
        checked = 0
        for c in labels["characters"]:
            best = None
            for st in range(c["start"], c["end"] - win + 1, hop):
                if not mask[st : st + win].all():
                    continue  # skip windows straddling an element edge
                v = float(np.mean(audio[st : st + win] ** 2))
                best = v if best is None else min(best, v)
            if best is None:
                continue
            measured = 10.0 * math.log10(best / ref**2)
            self.assertAlmostEqual(c["min_gain_db"], measured, delta=1.5)
            checked += 1
        self.assertGreater(checked, 5)

    # -- S:N calibration -------------------------------------------------
    def test_snr_matches_request_across_range(self):
        for snr in (-5.0, 0.0, 5.0, 15.0, 30.0):
            with self.subTest(snr=snr):
                _rate, audio, labels = self.run_gen(
                    f"snr{int(snr)}", ["--text", MESSAGE, "--repeat", "3", "--channel", "WGN", "--snr", str(snr)]
                )
                measured = self.measure_snr_db(audio, labels)
                self.assertAlmostEqual(measured, snr, delta=0.5)

    def test_snr_is_referenced_to_the_audio_band_not_the_filter(self):
        _rate, _audio, labels = self.run_gen("bw", ["--text", MESSAGE, "--channel", "WGN", "--snr", "5"])
        snr = labels["snr"]
        # Narrowing the observation bandwidth must improve S:N by the bandwidth ratio.
        expected = snr["requested_audio_band_db"] + 10.0 * math.log10(3000.0 / 250.0)
        self.assertAlmostEqual(snr["in_filter_250hz_db"], expected, delta=0.01)
        self.assertAlmostEqual(
            snr["in_filter_500hz_db"],
            snr["requested_audio_band_db"] + 10.0 * math.log10(3000.0 / 500.0),
            delta=0.01,
        )

    def test_measured_snr_is_reported_honestly(self):
        for snr in (-10.0, 0.0, 20.0):
            with self.subTest(snr=snr):
                _rate, audio, labels = self.run_gen(
                    f"honest{int(snr)}", ["--text", MESSAGE, "--channel", "MPP", "--snr", str(snr)]
                )
                self.assertAlmostEqual(labels["snr"]["measured_audio_band_db"], snr, delta=0.05)
                # Independent check, tolerating the fade-driven variance.
                self.assertAlmostEqual(self.measure_snr_db(audio, labels), snr, delta=2.0)

    # -- channel statistics ----------------------------------------------
    def test_wgn_mode_has_no_fading(self):
        _rate, audio, labels = self.run_gen("wgn", ["--text", MESSAGE, "--channel", "WGN", "--snr", "40"])
        self.assertEqual(labels["channel"]["paths"], 1)
        self.assertEqual(labels["channel"]["tick_rate_hz"], 0.0)
        mask = self.keydown_mask(labels, len(audio))
        # Per-element peak amplitude must be constant in pure WGN mode.
        levels = []
        for s in labels["symbols"]:
            if s["kind"] != "dah" or s["end"] - s["start"] < 100:
                continue
            seg = audio[s["start"] + 50 : s["end"] - 50]
            if seg.size:
                levels.append(np.max(np.abs(seg)))
        levels = np.array(levels)
        self.assertGreater(levels.size, 3)
        self.assertLess(np.std(levels) / np.mean(levels), 0.02, "WGN mode must not modulate level")

    def test_fading_amplitude_is_rayleigh(self):
        """Complex Gaussian taps give a Rayleigh envelope; check scale-free quantiles."""
        track = self.tmp / "rayleigh.csv"
        _rate, _audio, labels = self.run_gen(
            "rayleigh",
            ["--text", MESSAGE, "--repeat", "6", "--channel", "MPP", "--snr", "40", "--fadetrack", str(track)],
        )
        rows = np.loadtxt(track, delimiter=",", skiprows=1)
        gain_db = rows[:, 2]
        # Drop the fade warm-up region already excluded by the generator.
        amp = 10.0 ** (gain_db / 20.0)
        amp = amp[amp > 0]
        self.assertGreater(amp.size, 500)

        # Rayleigh: median / mean = sqrt(2 ln 2) / sqrt(pi/2) = 0.9393.
        ratio = np.median(amp) / np.mean(amp)
        self.assertAlmostEqual(ratio, 0.9393, delta=0.06)

        # 10/90 quantiles normalised by the median: sqrt(2 ln(1/0.9)) / sqrt(2 ln 2) = 0.3899
        # and sqrt(2 ln 10) / sqrt(2 ln 2) = 1.8226.
        q10, q90 = np.quantile(amp, [0.10, 0.90])
        self.assertAlmostEqual(q10 / np.median(amp), 0.3899, delta=0.08)
        self.assertAlmostEqual(q90 / np.median(amp), 1.8226, delta=0.08)

    def test_doppler_update_rate_is_64x_spread(self):
        for channel, spread in (("MPG", 0.1), ("MPM", 0.5), ("MPP", 1.0), ("MPD", 2.0)):
            with self.subTest(channel=channel):
                _rate, _audio, labels = self.run_gen(
                    f"tick{channel}", ["--text", "E", "--channel", channel, "--snr", "40"]
                )
                self.assertAlmostEqual(labels["channel"]["doppler_spread_hz"], spread, delta=1e-9)
                self.assertAlmostEqual(labels["channel"]["tick_rate_hz"], 64.0 * spread, delta=1e-9)

    def test_doppler_spectrum_is_scale_invariant(self):
        """The 128-tap Gauss FIR runs at 64x the spread, so the normalised spectrum of
        the tap gain must be the same shape for every CCIR channel. This is the single
        strongest check that the fading generator matches the firmware design."""
        widths = {}
        for channel in ("MPP", "MPD"):
            track = self.tmp / f"psd{channel}.csv"
            _rate, _audio, labels = self.run_gen(
                f"psd{channel}",
                ["--text", "E", "--repeat", "40", "--channel", channel, "--snr", "40", "--fadetrack", str(track)],
            )
            tick_rate = labels["channel"]["tick_rate_hz"]
            rows = np.loadtxt(track, delimiter=",", skiprows=1)
            amp = 10.0 ** (rows[:, 2] / 20.0)
            amp = amp - amp.mean()
            psd = np.abs(np.fft.rfft(amp * np.hanning(amp.size))) ** 2
            freqs = np.fft.rfftfreq(amp.size, d=1.0)  # cycles per tick
            psd = psd[1:]
            freqs = freqs[1:]
            cum = np.cumsum(psd)
            half = 0.5 * cum[-1]
            idx = int(np.searchsorted(cum, half))
            widths[channel] = freqs[idx]
            self.assertGreater(tick_rate, 0.0)

        self.assertAlmostEqual(widths["MPP"] / widths["MPD"], 1.0, delta=0.25)

    def test_per_path_doppler_is_honoured(self):
        """Paths may fade at different rates; that is the only way to get a narrow envelope
        and wide phase flutter out of the same channel."""
        track = self.tmp / "pd.csv"
        _r, _a, labels = self.run_gen(
            "pd", ["--text", MESSAGE, "--repeat", "3", "--channel", "MPP",
                   "--path-doppler", "0.5,8", "--snr", "40", "--fadetrack", str(track)]
        )
        self.assertEqual(labels["channel"]["paths"], 2)
        self.assertEqual(labels["channel"]["path_doppler_hz"], [0.5, 8.0])
        self.assertEqual(labels["channel"]["tick_rate_hz"], 32.0)  # 64 x path 0
        rows = np.loadtxt(track, delimiter=",", skiprows=1)
        # A fast path must decorrelate faster than a slow one: compare the lag-1 correlation
        # of the tap power at each path's own tick rate by using the shared file only as a
        # sanity check that the curve is not constant.
        amp = 10.0 ** (rows[:, 2] / 20.0)
        self.assertGreater(float(np.std(amp) / np.mean(amp)), 0.1)

    def test_rician_k_narrows_the_fading_distribution(self):
        """K moves power into a steady path, so the tap magnitude must stop behaving like
        Rayleigh: the spread shrinks and the median/mean ratio rises towards 1."""
        iqr = {}
        ratio = {}
        for k in ("0", "16"):
            track = self.tmp / f"rician{k}.csv"
            _r, _a, labels = self.run_gen(
                f"rician{k}", ["--text", MESSAGE, "--repeat", "6", "--channel", "MPP",
                               "--snr", "40", "--rician-k", k, "--fadetrack", str(track)]
            )
            self.assertAlmostEqual(labels["channel"]["rician_k"], float(k), delta=1e-9)
            rows = np.loadtxt(track, delimiter=",", skiprows=1)
            amp = 10.0 ** (rows[:, 2] / 20.0)
            amp = amp[amp > 0]
            db = 20.0 * np.log10(np.maximum(amp, 1e-30))
            iqr[k] = float(np.percentile(db, 75) - np.percentile(db, 25))
            ratio[k] = float(np.median(amp) / np.mean(amp))
        self.assertLess(iqr["16"], iqr["0"] - 2.0)
        # The IQR is the decisive signal (it halves); median/mean moves less because the
        # K=0 baseline is already close to 1 on a finite sample.
        self.assertGreater(ratio["16"], ratio["0"] + 0.02)

    def test_two_and_four_path_have_equal_mean_power(self):
        """IONOS fltSTD (4.0 vs 4.0/sqrt(2)) exists to keep 2-path and 4-path output
        level identical; verified here through the raw hardware staging."""
        p = {}
        for paths in ("2", "4"):
            _rate, audio, labels = self.run_gen(
                f"paths{paths}",
                ["--text", MESSAGE, "--repeat", "3", "--channel", "MPP", "--paths", paths,
                 "--snr", "-20", "--hardware-staging"],
            )
            p[paths] = float(np.mean(audio**2))
            self.assertEqual(labels["channel"]["paths"], int(paths))
        ratio_db = 10.0 * math.log10(p["2"] / p["4"])
        self.assertAlmostEqual(ratio_db, 0.0, delta=0.5)

    def test_flat_fade_depth_matches_request(self):
        depth = 12.0
        _rate, audio, labels = self.run_gen(
            "fade",
            ["--text", MESSAGE, "--repeat", "4", "--channel", "WGN", "--snr", "40",
             "--fade-depth", str(depth), "--fade", "0.5"],
        )
        # Measure per-element envelope so inter-element gaps (noise only) do not enter.
        levels = []
        for sym in labels["symbols"]:
            if sym["kind"] not in ("dit", "dah"):
                continue
            seg = np.abs(audio[sym["start"] : sym["end"]])
            if seg.size < 200:
                continue
            levels.append(np.percentile(seg, 95))
        levels = np.asarray(levels)
        # The fade is a cosine sweep, so its extremes are dwell points: use 2/98%.
        observed = 20.0 * math.log10(np.percentile(levels, 98) / np.percentile(levels, 2))
        self.assertAlmostEqual(observed, depth, delta=2.5)

    def test_offset_shifts_the_tone(self):
        _rate, audio, labels = self.run_gen(
            "offset", ["--text", "EEEE", "--channel", "WGN", "--snr", "40",
                       "--tone", "700", "--offset", "100"]
        )
        mask = self.keydown_mask(labels, len(audio))
        seg = audio[mask][: 8192]
        spec = np.abs(np.fft.rfft(seg * np.hanning(seg.size)))
        freqs = np.fft.rfftfreq(seg.size, 1.0 / 44100)
        peak = freqs[int(np.argmax(spec))]
        self.assertAlmostEqual(peak, 800.0, delta=30.0)

    def test_generation_is_deterministic(self):
        _r1, a1, _l1 = self.run_gen("det1", ["--text", MESSAGE, "--channel", "MPP", "--snr", "3", "--seed", "7"])
        _r2, a2, _l2 = self.run_gen("det2", ["--text", MESSAGE, "--channel", "MPP", "--snr", "3", "--seed", "7"])
        self.assertTrue(np.array_equal(a1, a2))

    def test_seed_changes_the_realisation(self):
        _r1, a1, _l1 = self.run_gen("seed1", ["--text", MESSAGE, "--channel", "MPP", "--snr", "3", "--seed", "1"])
        _r2, a2, _l2 = self.run_gen("seed2", ["--text", MESSAGE, "--channel", "MPP", "--snr", "3", "--seed", "2"])
        self.assertFalse(np.array_equal(a1, a2))

    def test_hardware_staging_and_normalised_differ(self):
        _r, a_norm, _l = self.run_gen("norm", ["--text", MESSAGE, "--channel", "MPP", "--snr", "40"])
        _r, a_hw, _l = self.run_gen(
            "hw", ["--text", MESSAGE, "--channel", "MPP", "--snr", "40", "--hardware-staging"]
        )
        self.assertNotAlmostEqual(float(np.mean(a_norm**2)), float(np.mean(a_hw**2)), places=4)


if __name__ == "__main__":
    unittest.main()

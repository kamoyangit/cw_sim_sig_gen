"""Closed-loop tests for tools/cwsim/calibrate.py.

The estimator is only trustworthy if it recovers parameters it was not told about. Every
test here synthesises a recording at known settings, throws the labels away, and checks
what the measurement reports.
"""

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

CWSIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CWSIM))

import calibrate as C  # noqa: E402

BIN = CWSIM / "build" / "gen_cw"
TEXT = "CQ CQ DE JA1ABC K " * 2


def require_bin():
    if not BIN.is_file():
        raise unittest.SkipTest(f"{BIN} not built; run make -C {CWSIM}")


class CalibrateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_bin()
        cls._tmp = tempfile.TemporaryDirectory(prefix="cwsim-cal-")
        cls.tmp = Path(cls._tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def synth(self, name, extra):
        wav = self.tmp / f"{name}.wav"
        labels = self.tmp / f"{name}.json"
        cmd = [str(BIN), "--text", TEXT, "--out", str(wav), "--labels", str(labels),
               "--quiet", "--seed", "5"] + extra
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return wav, json.loads(labels.read_text())

    # -- S:N -------------------------------------------------------------
    def test_snr_estimator_recovers_the_requested_value(self):
        for snr in (0.0, 10.0, 25.0):
            with self.subTest(snr=snr):
                wav, _lab = self.synth(f"snr{int(snr)}", ["--channel", "WGN", "--snr", str(snr)])
                got = C.analyze(wav)[0]["snr_db"]
                self.assertAlmostEqual(got, snr, delta=1.0)

    # -- fading ----------------------------------------------------------
    def test_fade_rate_estimator_recovers_the_requested_value(self):
        for rate in (0.2, 0.5, 1.0):
            with self.subTest(rate=rate):
                wav, _lab = self.synth(
                    f"rate{rate}", ["--channel", "WGN", "--snr", "35",
                                    "--fade-depth", "15", "--fade", str(rate)]
                )
                got = C.analyze(wav)[0]["fade_rate_hz"]
                self.assertAlmostEqual(got, rate, delta=0.05)

    def test_fade_average_penalty_predicts_the_measured_snr(self):
        depth = 15.0
        wav, lab = self.synth("depth", ["--channel", "WGN", "--snr", "35",
                                        "--fade-depth", str(depth), "--fade", "0.5"])
        stats_out = C.analyze(wav)[0]
        predicted = lab["snr"]["requested_audio_band_db"] + C.fade_average_penalty_db(depth)
        self.assertAlmostEqual(stats_out["snr_db"], predicted, delta=2.0)

    def test_envelope_span_tracks_the_fade_depth(self):
        spans = {}
        for depth in (6.0, 18.0):
            wav, _lab = self.synth(
                f"span{int(depth)}", ["--channel", "WGN", "--snr", "40",
                                      "--fade-depth", str(depth), "--fade", "0.5"]
            )
            spans[depth] = C.analyze(wav)[0]["envelope_db_span_p10_p90"]
        self.assertGreater(spans[18.0], spans[6.0] + 6.0)

    def test_doppler_increases_phase_flutter(self):
        """More Doppler spreading must produce more frequency flutter. Checked as an
        ordering, because the metric is relative to an S:N-dependent noise floor."""
        vals = {}
        for channel in ("WGN", "MPG", "MPD"):
            wav, lab = self.synth(
                f"dop{channel}", ["--channel", channel, "--snr", "30",
                                  "--fade-depth", "10", "--fade", "0.5"]
            )
            vals[channel] = C.analyze(wav)[0]["inst_freq_iqr_hz"]
        self.assertGreater(vals["MPG"], vals["WGN"])
        self.assertGreater(vals["MPD"], vals["MPG"])

    # -- compare ---------------------------------------------------------
    def test_compare_flags_identical_and_different_generators(self):
        same_a, _ = self.synth("cmp_a", ["--channel", "MPP", "--snr", "15", "--seed", "1"])
        same_b, _ = self.synth("cmp_b", ["--channel", "MPP", "--snr", "15", "--seed", "2"])
        diff, _ = self.synth("cmp_c", ["--channel", "WGN", "--snr", "15"])

        res_same = C.compare(same_a, same_b)
        res_diff = C.compare(same_a, diff)

        d_same = res_same["tests"]["envelope_db_ks"]["statistic"]
        d_diff = res_diff["tests"]["envelope_db_ks"]["statistic"]
        self.assertLess(d_same, 0.2, "two draws of the same channel should match")
        self.assertGreater(d_diff, d_same, "a fading channel must differ from a flat one")

    # -- I/Q input --------------------------------------------------------
    def to_iq(self, wav_path, out_path, carrier_hz=1500.0, noise=0.005, seed=0):
        """Turn mono audio into a plausible SDR capture: analytic, shifted, wideband noise."""
        from scipy import signal
        from scipy.io import wavfile
        sr, x = C.load_wav(wav_path)
        t = np.arange(x.size) / sr
        z = signal.hilbert(x) * np.exp(2j * np.pi * carrier_hz * t)
        rng = np.random.default_rng(seed)
        z = z + noise * (rng.standard_normal(z.size) + 1j * rng.standard_normal(z.size))
        wavfile.write(str(out_path), sr, np.stack([z.real, z.imag], axis=1).astype(np.float32))
        return out_path

    def test_iq_is_detected_and_the_fade_rate_survives(self):
        wav, _lab = self.synth("iqsrc", ["--channel", "MPP", "--snr", "12",
                                         "--fade-depth", "12", "--fade", "0.5"])
        iq = self.to_iq(wav, self.tmp / "iqsrc_iq.wav")
        self.assertTrue(C.is_complex_source(iq))
        self.assertFalse(C.is_complex_source(wav))

        audio_stats = C.analyze(wav)[0]
        iq_stats = C.analyze(iq)[0]

        self.assertEqual(iq_stats["input"], "iq")
        # The single tone was shifted by the carrier, so the detected offset must be
        # carrier + tone. This checks channelisation, not just detection.
        self.assertAlmostEqual(iq_stats["iq_offset_hz"], 1500.0 + audio_stats["tone_hz"], delta=60.0)
        # Fade rate is bandwidth-independent, so it must be identical.
        self.assertAlmostEqual(iq_stats["fade_rate_hz"], audio_stats["fade_rate_hz"], delta=0.05)
        # Duty is threshold- and bandwidth-sensitive (the audio path keys a 3 kHz band,
        # the I/Q path a 500 Hz one), so it is only expected to agree loosely.
        self.assertAlmostEqual(iq_stats["duty"], audio_stats["duty"], delta=0.15)

    def test_iq_measures_fade_depth_closer_than_audio(self):
        depth = 15.0
        wav, _lab = self.synth("iqdepth", ["--channel", "WGN", "--snr", "12",
                                           "--fade-depth", str(depth), "--fade", "0.5"])
        iq = self.to_iq(wav, self.tmp / "iqdepth_iq.wav", noise=0.002)
        audio_span = C.analyze(wav)[0]["envelope_db_span_p10_p90"]
        iq_span = C.analyze(iq)[0]["envelope_db_span_p10_p90"]
        # A 500 Hz I/Q passband rejects most of the noise, so the fade floor is visible
        # instead of being buried, and the measured depth is closer to the truth.
        self.assertLess(abs(iq_span - depth), abs(audio_span - depth) + 0.5)

    def test_stereo_duplicated_audio_is_not_treated_as_iq(self):
        from scipy.io import wavfile
        sr = 44100
        mono = (0.3 * np.sin(2 * np.pi * 600 * np.arange(sr) / sr)).astype(np.float32)
        path = self.tmp / "dup.wav"
        wavfile.write(str(path), sr, np.stack([mono, mono], axis=1))
        self.assertFalse(C.is_complex_source(path))

    def test_iq_compound_note_appears_in_compare(self):
        wav, _lab = self.synth("cmpiq", ["--channel", "MPP", "--snr", "15", "--fade-depth", "10", "--fade", "0.5"])
        iq = self.to_iq(wav, self.tmp / "cmpiq_iq.wav")
        res = C.compare(wav, iq)
        self.assertTrue(any("not" in n and "comparable" in n for n in res["notes"]))

    # -- carrier scan -----------------------------------------------------
    @staticmethod
    def textbook(**over):
        st = {"dit_dah_ratio": 3.02, "duty": 0.52, "elements": 300, "snr_db": 22.0,
              "rayleigh_median_over_mean": 0.94, "iq_bw_hz": 200.0,
              "in_channel_peak_ratio_db": -32.0, "keydown_seconds": 40.0,
              "envelope_db_iqr": 8.4}
        st.update(over)
        return st

    def test_quality_score_accepts_a_textbook_signal(self):
        score, notes, separable = C.quality_score(self.textbook(), stability=0.02,
                                                  neighbour_hz=1500.0)
        self.assertGreaterEqual(score, 100)
        self.assertTrue(separable)
        self.assertEqual(notes, [])

    def test_quality_score_rejects_a_wide_envelope(self):
        """A Rayleigh envelope has IQR 6.83 dB and a clean synthetic reads about 9 dB, so
        23 dB is far outside anything one Rayleigh channel produces."""
        score, notes, separable = C.quality_score(self.textbook(envelope_db_iqr=23.0),
                                                  stability=0.02, neighbour_hz=1500.0)
        self.assertFalse(separable)
        self.assertLessEqual(score, 20)
        self.assertTrue(any("IQR" in n for n in notes))

    def test_quality_score_rejects_an_unseparable_carrier(self):
        """Two carriers 40 Hz apart cannot be separated at 200 Hz, however good the rest
        of the evidence looks."""
        st = self.textbook(in_channel_peak_ratio_db=-6.0)   # something 6 dB down in band
        score, notes, separable = C.quality_score(st, stability=0.02, neighbour_hz=40.0)
        self.assertFalse(separable)
        self.assertLessEqual(score, 20)
        self.assertTrue(any("competing peak" in n for n in notes))

    def test_quality_score_flags_wrong_keying_ratio(self):
        score, notes, separable = C.quality_score(self.textbook(dit_dah_ratio=5.6),
                                                  stability=0.02, neighbour_hz=1500.0)
        self.assertTrue(separable)
        self.assertLess(score, 100)
        self.assertTrue(any("dit:dah" in n for n in notes))

    def test_quality_score_flags_a_non_rayleigh_envelope(self):
        score, notes, _ = C.quality_score(self.textbook(rayleigh_median_over_mean=0.42),
                                          stability=0.02, neighbour_hz=1500.0)
        self.assertLess(score, 100)
        self.assertTrue(any("Rayleigh" in n for n in notes))

    def test_scan_finds_a_synthetic_carrier(self):
        wav, lab = self.synth("scan", ["--channel", "MPP", "--snr", "25", "--fade-depth", "4",
                                       "--fade", "0.05", "--wpm", "18"])
        iq = self.to_iq(wav, self.tmp / "scan_iq.wav", noise=0.002)
        res = C.scan(iq, bw_hz=200.0, check_stability=False, max_carriers=4)
        self.assertTrue(res["carriers"])
        best = res["carriers"][0]
        self.assertGreaterEqual(best["score"], 60)
        # to_iq() shifts by +1500 Hz, so the tone ends up at 1500 + tone.
        self.assertAlmostEqual(best["offset_hz"], 1500.0 + lab["cw"]["tone_hz"], delta=90.0)

    # -- fit --------------------------------------------------------------
    def test_fit_proposes_usable_settings(self):
        wav, lab = self.synth("fitsrc", ["--channel", "MPP", "--snr", "20",
                                         "--fade-depth", "6", "--fade", "0.1", "--wpm", "18"])
        iq = self.to_iq(wav, self.tmp / "fitsrc_iq.wav", noise=0.002)
        res = C.fit_capture(iq, offset_hz=1500.0 + lab["cw"]["tone_hz"], bw_hz=200.0,
                            segments=3)
        self.assertGreaterEqual(res["segments_used"], 2)
        self.assertIn("--channel", res["suggested_command"])
        # The proposed channel must be one the generator actually has.
        self.assertRegex(res["suggested_command"], r"--channel (WGN|MPG|MPM|MPP|MPD)")
        # And the measured envelope must be Rayleigh-like, since that is what was generated.
        self.assertAlmostEqual(res["targets"]["rayleigh_median_over_mean"]["mean"], 0.939,
                              delta=0.12)
        self.assertGreater(len(res["notes"]), 1)

    def test_channel_for_flutter_covers_all_four(self):
        """MPM was once missing from the branch, making one of the four channels
        unreachable from `fit`. These are the measured flutter values for each channel."""
        cases = [(0.20, "MPG"), (0.37, "MPM"), (0.72, "MPP"), (1.45, "MPD")]
        for flutter, want in cases:
            with self.subTest(flutter=flutter):
                got, doppler = C.channel_for_flutter(flutter)
                self.assertEqual(got, want)
                self.assertGreater(doppler, 0.0)

    def test_channel_for_flutter_handles_the_edges(self):
        self.assertEqual(C.channel_for_flutter(None)[0], "MPP")
        self.assertEqual(C.channel_for_flutter(0.0)[0], "MPG")
        self.assertEqual(C.channel_for_flutter(99.0)[0], "MPD")

    def test_fit_rejects_a_capture_without_keying(self):
        import scipy.io.wavfile as wavfile
        sr = 48000
        tone = (0.3 * np.exp(2j * np.pi * 800 * np.arange(sr * 4) / sr)).astype(np.complex64)
        path = self.tmp / "tone.cf32"
        tone.tofile(path)
        with self.assertRaises(ValueError):
            C.fit_capture(path, raw_fs=sr, bw_hz=200.0, segments=3)

    def test_cli_reports_bad_input_without_a_traceback(self):
        """A CLI is judged on what it prints when handed garbage."""
        for bad in (str(CWSIM / "corpus.txt"), str(self.tmp / "does_not_exist.wav")):
            with self.subTest(input=bad):
                proc = subprocess.run([sys.executable, str(CWSIM / "calibrate.py"),
                                       "analyze", bad], capture_output=True, text=True)
                self.assertNotEqual(proc.returncode, 0)
                self.assertNotIn("Traceback", proc.stderr)
                self.assertIn("error:", proc.stderr)

    def test_analyze_rejects_too_short_input(self):
        short = self.tmp / "short.wav"
        import scipy.io.wavfile as wavfile
        wavfile.write(str(short), 44100, np.zeros(1000, dtype=np.float32))
        with self.assertRaises(ValueError):
            C.analyze(short)

    def test_analyze_handles_silence_without_crashing(self):
        silent = self.tmp / "silent.wav"
        import scipy.io.wavfile as wavfile
        wavfile.write(str(silent), 44100, np.zeros(44100 * 2, dtype=np.float32))
        stats_out = C.analyze(silent)[0]
        self.assertEqual(stats_out["sample_rate"], 44100)
        # No keying to find, so no S:N can be reported.
        self.assertIsNone(stats_out["snr_db"])


if __name__ == "__main__":
    unittest.main()

// cwchannel.h -- offline Watterson/IONOS channel core for CW (Morse) data generation
//
// Extracted from src/HFSim_BFD_2_05/HFSim_BFD_2_05.ino (IONOS HF/VHF Channel Simulator,
// (C) 2020 Amateur Radio Safety Foundation Inc., MIT licence) so that the same channel
// statistics can be produced offline, without a Teensy, at arbitrary length.
//
// The maths is preserved deliberately; the real-time servo/calibration plumbing is not.
// See README.md for what differs and why.
#ifndef CWSIM_CWCHANNEL_H
#define CWSIM_CWCHANNEL_H

#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cwsim {

// Channel types as defined by CCIR/ITU-R F.520 and implemented by IONOS.
enum class Mode { WGN, MPG, MPM, MPP, MPD, CUSTOM };

const char* mode_name(Mode m);
bool mode_from_name(const std::string& name, Mode* out);

// How the tap gains are moved between tick-rate updates.
//  Linear        : sample-wise interpolation (closest to the Watterson ideal).
//  ZeroOrderHold : hold each tick constant, as the Teensy does, because AudioMixer4
//                  gains only change once per audio block.
enum class FadeInterp { Linear, ZeroOrderHold };

struct ChannelPreset {
    Mode mode = Mode::WGN;
    int paths = 1;                    // 1 (WGN), 2 or 4
    double doppler_spread_hz = 0.0;   // CCIR "2 sigma" figure, Hz
    std::vector<double> delay_ms;     // delay of path k relative to path 0
};

// IONOS/CCIR F.520 preset. paths must be 2 (or 1 for WGN) or 4.
ChannelPreset ionos_preset(Mode mode, int paths);

struct ChannelConfig {
    ChannelPreset preset = ionos_preset(Mode::WGN, 1);

    // Optional override of the per-path relative delays; non-empty wins over
    // preset.delay_ms. This is how the CW-oriented long-delay taps are expressed.
    std::vector<double> delay_ms_override;

    // IONOS delays the I tap by 59 samples (1.3378684807 ms) to line it up with the
    // 119-tap Hilbert FIR on the Q path. Kept so generated audio can be compared
    // sample-for-sample against hardware captures.
    double base_delay_ms = 1.3378684807;

    // Per-path Doppler spread (2 sigma, Hz). Empty means every path uses
    // preset.doppler_spread_hz. Giving paths different values is the one thing the plain
    // Watterson model cannot express, and it is what the measured capture seems to need:
    // a slow path narrows the envelope distribution while a fast path widens the phase
    // flutter, and no single Doppler value produces both at once.
    std::vector<double> path_doppler_hz;

    // Rician K (linear power ratio of a steady path to the scattered paths). K = 0 is a
    // pure Watterson/Rayleigh channel; larger K narrows the envelope distribution towards
    // a constant. A measured 40 m capture showed median/mean 0.992 and IQR 7.3 dB against
    // Rayleigh's 0.939 / 8.7 dB, i.e. narrower than pure Rayleigh, which is what K > 0
    // expresses. The steady path carries K/(K+1) of the mean power.
    double rician_k = 0.0;

    // Flat fading (IONOS "FADE DEPTH"/"FADE FREQ"), applied as an S:N envelope.
    double fade_depth_db = 0.0;
    double fade_rate_hz = 0.0;

    double snr_db = 20.0;          // audio-band S:N, referenced to key-down signal power
    double audio_bw_hz = 3000.0;   // 3000 or 6000, matching the IONOS input LP filter
    double freq_offset_hz = 0.0;   // IONOS "OFFSET"

    FadeInterp fade_interp = FadeInterp::Linear;

    // Replicate the IONOS absolute gain chain instead of normalising by power.
    // IONOS mixes signal and noise with gains that sum to 1, so its displayed S:N is
    // exact only when the driving audio sits at the design level (~1.19 V p-p for
    // 3 kHz). Off by default; see README "S:N reference".
    bool hardware_staging = false;

    std::uint64_t seed = 1;

    bool has_fading() const { return preset.paths > 1 && preset.doppler_spread_hz > 0.0; }
    const std::vector<double>& delays_ms() const;
    int paths() const { return preset.paths < 1 ? 1 : preset.paths; }
};

struct FadingDiagnostics {
    double tick_rate_hz = 0.0;              // path 0, for reporting
    std::vector<double> path_tick_rate_hz;  // per path; 0 when the path does not fade
    // gain_db[path][tick]: complex tap magnitude in dB at each fading update.
    std::vector<std::vector<float>> gain_db;
    // Mean of |g|^2 across all paths and ticks (linear power gain of the fading section).
    double mean_power_gain = 0.0;
};

// Offline renderer: construct, render a whole buffer, inspect diagnostics.
// Not thread-safe, not real-time. Delays and band limiting are done in the frequency
// domain over the whole buffer, so long buffers give exact fractional delays.
class Renderer {
  public:
    Renderer(const ChannelConfig& cfg, double sample_rate);

    // in and out may alias. `active` optionally marks the key-down samples; when set,
    // S:N is referenced to the mean square of those samples only (noise is continuous).
    void render(const float* in, std::size_t n, std::vector<float>& out,
                const std::vector<std::uint8_t>* active = nullptr);

    const ChannelConfig& config() const { return cfg_; }
    const FadingDiagnostics& fading() const { return diag_; }
    double sample_rate() const { return fs_; }

    // Signal-only output of the last render(), including fading but no noise.
    const std::vector<float>& signal_only() const { return signal_only_; }
    double signal_power() const { return signal_power_; }
    double noise_power() const { return noise_power_; }
    // Realised audio-band S:N of the last render (dB).
    double measured_snr_db() const { return measured_snr_db_; }
    // Equivalent S:N inside a receiver filter of width cw_bw_hz.
    double snr_in_filter_db(double cw_bw_hz) const;

  private:
    void build_fading(std::size_t n);
    void build_noise(std::size_t n, std::size_t guard, double target_power);

    ChannelConfig cfg_;
    double fs_;
    FadingDiagnostics diag_;

    std::vector<std::vector<double>> taps_i_;  // [path][tick]
    std::vector<std::vector<double>> taps_q_;  // [path][tick]
    std::vector<float> signal_only_;
    std::vector<float> noise_;

    double signal_power_ = 0.0;
    double noise_power_ = 0.0;
    double measured_snr_db_ = 0.0;
};

// CW-oriented tap delay presets. Delays beyond the IONOS range are allowed because a
// PC has no audio delay-line limit, and HF CW "hollow"/echo is a 10-100 ms phenomenon
// that the data-mode presets never exercise.
struct CwDelayPreset {
    const char* name;
    std::vector<double> delays_ms;
};
std::vector<CwDelayPreset> cw_delay_presets();

}  // namespace cwsim

#endif  // CWSIM_CWCHANNEL_H

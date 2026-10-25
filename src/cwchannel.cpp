// cwchannel.cpp -- offline Watterson/IONOS channel core for CW data generation.
// See cwchannel.h and README.md.
#include "cwchannel.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>

namespace cwsim {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;

// ---------------------------------------------------------------------------
// 128-tap Gaussian-spectrum FIR, from IONOS HFSim_BFD_2_05.ino
// ("Iowa Hills FIR Filter Designer Version 7.0", designed at a 64 Hz sample rate,
// Fc = 0.5856 Hz, ~Gauss -0.900, 9 poles, group delay ~0.864 s).
// Sum(h) = 1.0, Sum(h^2) = 0.019412401884695396.
// ---------------------------------------------------------------------------
const double kGaussFir[128] = {
    11.75559267133204600E-12, 201.8800495613742780E-12, 1.723633362394617620E-9, 9.815423109243150530E-9,
    42.19820040519088170E-9,  146.9342948623463490E-9,  433.8503956649552150E-9, 1.122118806000393040E-6,
    2.604091536729610910E-6,  5.522713327023963000E-6,  10.85739046564233410E-6, 20.01141264924749310E-6,
    34.89306870616279350E-6,  57.98247725939228300E-6,  92.37678934582388020E-6, 141.8076728400771460E-6,

    210.6266900063565690E-6,  303.7561533907517630E-6,  426.6051229102419030E-6, 584.9522379382009380E-6,
    784.7989367901134300E-6,  0.001032198204838678,    0.001333065243166745,    0.001692977321877409,
    0.002116970561258896,     0.002609341477645015,    0.003173460865247882,    0.003811607001168640,
    0.004524824309342139,     0.005312812558055807,    0.006173850455688009,    0.007104756211217515,

    0.008100886298035966,     0.009156172355120720,    0.010263194925878348,    0.011413291611735991,
    0.012596696236577472,     0.013802704802828978,    0.015019863385625786,    0.016236172665450827,
    0.017439303542147160,     0.018616818198095349,    0.019756391073966928,    0.020846024470695095,
    0.021874253876569307,     0.022830338616651881,    0.023704434009553566,    0.024487741869950012,

    0.025172636890277961,     0.025752767148979578,    0.026223127704178725,    0.026580106921515002,
    0.026821505836160390,     0.026946531447567135,    0.026955765379786157,    0.026851109801616951,
    0.026635712883536795,     0.026313876369100774,    0.025890948056565766,    0.025373202123371387,
    0.024767710285281262,     0.024082206768594926,    0.023324949994399220,    0.022504583735910824,

    0.021630000321890532,     0.020710208229666707,    0.019754206149457228,    0.018770865316331563,
    0.017768821605931590,     0.016756378583127243,    0.015741422386651591,    0.014731349034225070,
    0.013733004447687327,     0.012752637231260112,    0.011795863992392318,    0.010867646776884902,
    0.009972282000446704,     0.009113400098909145,    0.008293974989634013,    0.007516342337046732,

    0.006782225544921226,     0.006092768355660706,    0.005448572920512081,    0.004849742212182516,
    0.004295925680163602,     0.003786367096475506,    0.003319953602663025,    0.002895265044807468,
    0.002510622769190125,     0.002164137144270543,    0.001853753172183700,    0.001577293652557479,
    0.001332499460868328,     0.001117066600796574,    928.6797833839573290E-6, 765.0423737761393570E-6,

    623.9026277671727030E-6,  503.0762143350814310E-6,  400.4650862100223210E-6, 314.0728178353742240E-6,
    242.0165786791055780E-6,  182.5359497431699650E-6,  133.9988224980702400E-6, 94.90464268873809320E-6,
    63.88527699708468790E-6,  39.70378899074398050E-6,  21.25141280107635570E-6, 7.543009276742865590E-6,
    -2.288719293693219470E-6, -8.999992637884092870E-6, -13.24344714850232660E-6, -15.57613368708467760E-6,

    -16.46781142685044590E-6, -16.30935284928609970E-6, -15.42109793945524210E-6, -14.06101870492278390E-6,
    -12.43257788819570920E-6, -10.69218773541830810E-6, -8.956195575428226970E-6, -7.307342471035109810E-6,
    -5.800659111239630410E-6, -4.468779263172822700E-6, -3.326665397016021600E-6, -2.375753489237729440E-6,
    -1.607534498745384790E-6, -1.006598637105850620E-6, -553.1753923550552370E-9, -225.2074190014706120E-9};

constexpr int kGaussFirTaps = 128;
// Sum(h^2); pinned here and asserted by the tests so a typo in the table is caught.
[[maybe_unused]] constexpr double kGaussFirPowerSum = 0.019412401884695396;

// IONOS AdjustS_N() calibration constants.
constexpr double kSnCalibrationDb = 4.28;
// Band-limited white noise rms in the IONOS staging, at 1000 counts full scale.
constexpr double kIonosNoiseRms3k = 0.235;
constexpr double kIonosNoiseRms6k = 0.313;
// IONOS mixes signal and noise with gains that sum to 1, applied to these raw stages.
constexpr double kIonosWgnSigGain = 0.8498;
constexpr double kIonosPathGain = 0.5;

// ---------------------------------------------------------------------------
// Deterministic RNG (SplitMix64): reproducible across platforms, unlike std::.
// ---------------------------------------------------------------------------
class Rng {
  public:
    explicit Rng(std::uint64_t seed) : s_(seed) {}
    std::uint64_t next_u64() {
        std::uint64_t z = (s_ += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    double uniform01() { return static_cast<double>(next_u64() >> 11) * 0x1.0p-53; }
    // Uniform in [-1,1], matching IONOS random(-100000,100001) * 1e-5.
    double uniform_pm1() { return 2.0 * uniform01() - 1.0; }
    double normal() {
        double u1 = uniform01();
        if (u1 < 1e-300) u1 = 1e-300;
        const double u2 = uniform01();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(kTwoPi * u2);
    }

  private:
    std::uint64_t s_;
};

// Complex Gaussian tap weight, replicating the IONOS rejection-sampled Box-Muller
// exactly, including the 0.1 <= |z|^2 < 1 annulus that truncates the tails.
void ionos_complex_gaussian(Rng& rng, double std_dev, double* out_i, double* out_q) {
    double zi = 0.0, zq = 0.0, m2 = 2.0;
    while (m2 >= 1.0 || m2 < 0.1) {
        zi = rng.uniform_pm1();
        zq = rng.uniform_pm1();
        m2 = zi * zi + zq * zq;
    }
    const double mag = std::sqrt(m2);
    const double root = std::sqrt(-2.0 * std::log(m2));
    *out_i = (zi * root * std_dev) / mag;
    *out_q = (zq * root * std_dev) / mag;
}

// Twiddle table, cached per (size, direction). Recomputing cos/sin inside the butterfly
// loop dominated render time for long buffers; the generator is a batch tool, so a
// process-wide cache is both safe and worth several times the throughput.
const std::vector<std::complex<double>>& twiddles(std::size_t n, bool inverse) {
    static std::vector<std::complex<double>> fwd, inv;
    static std::size_t fwd_n = 0, inv_n = 0;
    std::vector<std::complex<double>>& tab = inverse ? inv : fwd;
    std::size_t& known = inverse ? inv_n : fwd_n;
    if (known != n) {
        tab.resize(n / 2);
        const double sign = inverse ? 1.0 : -1.0;
        for (std::size_t i = 0; i < n / 2; ++i) {
            const double ang = sign * kTwoPi * static_cast<double>(i) / static_cast<double>(n);
            tab[i] = std::complex<double>(std::cos(ang), std::sin(ang));
        }
        known = n;
    }
    return tab;
}

// Radix-2 FFT, in place. Size must be a power of two.
void fft(std::vector<std::complex<double>>& a, bool inverse) {
    const std::size_t n = a.size();
    if (n < 2) return;
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    const auto& wt = twiddles(n, inverse);
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const std::size_t half = len / 2;
        const std::size_t stride = n / len;
        for (std::size_t i = 0; i < n; i += len) {
            for (std::size_t k = 0; k < half; ++k) {
                const std::complex<double> u = a[i + k];
                const std::complex<double> v = a[i + k + half] * wt[k * stride];
                a[i + k] = u + v;
                a[i + k + half] = u - v;
            }
        }
    }
    if (inverse) {
        const double inv = 1.0 / static_cast<double>(n);
        for (auto& v : a) v *= inv;
    }
}

std::size_t next_pow2(std::size_t v) {
    std::size_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

double mean_square(const float* x, std::size_t n, const std::vector<std::uint8_t>* active) {
    double acc = 0.0;
    std::size_t cnt = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (active && !(*active)[i]) continue;
        acc += static_cast<double>(x[i]) * static_cast<double>(x[i]);
        ++cnt;
    }
    return cnt ? acc / static_cast<double>(cnt) : 0.0;
}

double bin_frequency(std::size_t k, std::size_t npad, double fs) {
    const double n = static_cast<double>(npad);
    return (k <= npad / 2) ? static_cast<double>(k) * fs / n
                           : (static_cast<double>(k) - n) * fs / n;
}

}  // namespace

const char* mode_name(Mode m) {
    switch (m) {
        case Mode::WGN: return "WGN";
        case Mode::MPG: return "MPG";
        case Mode::MPM: return "MPM";
        case Mode::MPP: return "MPP";
        case Mode::MPD: return "MPD";
        case Mode::CUSTOM: return "CUSTOM";
    }
    return "?";
}

bool mode_from_name(const std::string& name, Mode* out) {
    std::string u;
    for (char c : name) u.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
    if (u == "WGN") { *out = Mode::WGN; return true; }
    if (u == "MPG") { *out = Mode::MPG; return true; }
    if (u == "MPM") { *out = Mode::MPM; return true; }
    if (u == "MPP") { *out = Mode::MPP; return true; }
    if (u == "MPD") { *out = Mode::MPD; return true; }
    if (u == "CUSTOM") { *out = Mode::CUSTOM; return true; }
    return false;
}

ChannelPreset ionos_preset(Mode mode, int paths) {
    ChannelPreset p;
    p.mode = mode;
    if (mode == Mode::WGN) {
        p.paths = 1;
        p.doppler_spread_hz = 0.0;
        p.delay_ms = {0.0};
        return p;
    }
    p.paths = (paths == 4) ? 4 : 2;
    switch (mode) {
        case Mode::MPG: p.doppler_spread_hz = 0.1; break;
        case Mode::MPM: p.doppler_spread_hz = 0.5; break;
        case Mode::MPP: p.doppler_spread_hz = 1.0; break;
        case Mode::MPD: p.doppler_spread_hz = 2.0; break;
        default: p.doppler_spread_hz = 1.0; break;
    }
    // IONOS SetIQTapDelays(): 2-path uses one delay of .5/1/2/4 ms; 4-path spaces
    // four taps by half that increment (DL6MAA's recommendation).
    const double base = (mode == Mode::MPG) ? 0.5 : (mode == Mode::MPM) ? 1.0 : (mode == Mode::MPP) ? 2.0 : 4.0;
    if (p.paths == 2) {
        p.delay_ms = {0.0, base};
    } else {
        const double step = base * 0.5;
        p.delay_ms = {0.0, step, 2.0 * step, 3.0 * step};
    }
    return p;
}

const std::vector<double>& ChannelConfig::delays_ms() const {
    return delay_ms_override.empty() ? preset.delay_ms : delay_ms_override;
}

std::vector<CwDelayPreset> cw_delay_presets() {
    return {
        {"ccir-mpg", {0.0, 0.5}},
        {"ccir-mpm", {0.0, 1.0}},
        {"ccir-mpp", {0.0, 2.0}},
        {"ccir-mpd", {0.0, 4.0}},
        {"cw-short", {0.0, 5.0, 12.0}},
        {"cw-mid", {0.0, 15.0, 35.0}},
        {"cw-long", {0.0, 30.0, 70.0}},
        {"cw-roundworld", {0.0, 45.0, 110.0, 180.0}},
    };
}

Renderer::Renderer(const ChannelConfig& cfg, double sample_rate) : cfg_(cfg), fs_(sample_rate) {
    if (fs_ <= 0.0) throw std::invalid_argument("sample_rate must be positive");
    if (cfg_.paths() < 1) throw std::invalid_argument("paths must be >= 1");
    if (cfg_.delays_ms().size() != static_cast<std::size_t>(cfg_.paths()))
        throw std::invalid_argument("delay count must equal path count");
    if (cfg_.audio_bw_hz <= 0.0 || cfg_.audio_bw_hz >= fs_)
        throw std::invalid_argument("audio_bw_hz must be in (0, sample_rate)");
    if (!cfg_.path_doppler_hz.empty() &&
        cfg_.path_doppler_hz.size() != static_cast<std::size_t>(cfg_.paths()))
        throw std::invalid_argument("path_doppler_hz must be empty or match the path count");
    diag_.tick_rate_hz = cfg_.has_fading() ? 64.0 * cfg_.preset.doppler_spread_hz : 0.0;
    diag_.path_tick_rate_hz.assign(static_cast<std::size_t>(cfg_.paths()), diag_.tick_rate_hz);
    if (!cfg_.path_doppler_hz.empty()) {
        for (int p = 0; p < cfg_.paths(); ++p) {
            const double sp = cfg_.path_doppler_hz[static_cast<std::size_t>(p)];
            diag_.path_tick_rate_hz[static_cast<std::size_t>(p)] = (sp > 0.0) ? 64.0 * sp : 0.0;
        }
        diag_.tick_rate_hz = diag_.path_tick_rate_hz.empty() ? 0.0 : diag_.path_tick_rate_hz[0];
    }
}

void Renderer::build_fading(std::size_t n) {
    const int paths = cfg_.paths();
    diag_.gain_db.assign(static_cast<std::size_t>(paths), {});
    diag_.mean_power_gain = 0.0;
    taps_i_.assign(static_cast<std::size_t>(paths), {});
    taps_q_.assign(static_cast<std::size_t>(paths), {});

    if (!cfg_.has_fading()) {
        for (int p = 0; p < paths; ++p) {
            taps_i_[p] = {1.0, 1.0};
            taps_q_[p] = {0.0, 0.0};
            diag_.gain_db[p] = {0.0f, 0.0f};
        }
        diag_.mean_power_gain = 1.0;
        return;
    }

    if (diag_.path_tick_rate_hz.size() != static_cast<std::size_t>(paths))
        diag_.path_tick_rate_hz.assign(static_cast<std::size_t>(paths), diag_.tick_rate_hz);

    // fltSTD keeps the long-term average output equal between 2-path and 4-path operation:
    // 2 * 4.0^2 * Sum(h^2) == 4 * (4.0/sqrt(2))^2 * Sum(h^2).
    const double flt_std = (paths == 2) ? 4.0 : (4.0 / std::sqrt(2.0));
    const std::size_t warmup = 2 * kGaussFirTaps;

    Rng rng(cfg_.seed);
    double power_acc = 0.0;
    std::size_t power_cnt = 0;

    // Each path is generated on its own tick grid, so paths may fade at different rates.
    for (int p = 0; p < paths; ++p) {
        const double tick_rate = diag_.path_tick_rate_hz[static_cast<std::size_t>(p)];
        if (tick_rate <= 0.0) {
            taps_i_[p] = {1.0, 1.0};
            taps_q_[p] = {0.0, 0.0};
            diag_.gain_db[p] = {0.0f, 0.0f};
            continue;
        }
        const std::size_t ticks =
            static_cast<std::size_t>(std::ceil(static_cast<double>(n) / fs_ * tick_rate)) +
            kGaussFirTaps + 2;
        const std::size_t total_ticks = warmup + ticks;

        taps_i_[p].assign(ticks, 0.0);
        taps_q_[p].assign(ticks, 0.0);
        diag_.gain_db[p].assign(ticks, 0.0f);

        std::vector<double> di(kGaussFirTaps, 0.0), dq(kGaussFirTaps, 0.0);
        for (std::size_t k = 0; k < total_ticks; ++k) {
            double zi = 0.0, zq = 0.0;
            ionos_complex_gaussian(rng, flt_std, &zi, &zq);
            for (int i = kGaussFirTaps - 1; i > 0; --i) {
                di[i] = di[i - 1];
                dq[i] = dq[i - 1];
            }
            di[0] = zi;
            dq[0] = zq;
            double si = 0.0, sq = 0.0;
            for (int i = 0; i < kGaussFirTaps; ++i) {
                si += di[i] * kGaussFir[i];
                sq += dq[i] * kGaussFir[i];
            }
            if (k < warmup) continue;
            const std::size_t kk = k - warmup;
            taps_i_[p][kk] = si;
            taps_q_[p][kk] = sq;
            const double m2 = si * si + sq * sq;
            diag_.gain_db[p][kk] = static_cast<float>(10.0 * std::log10(m2 + 1e-30));
            power_acc += m2;
            ++power_cnt;
        }
    }
    diag_.mean_power_gain = power_cnt ? power_acc / static_cast<double>(power_cnt) : 0.0;

    // Rician: split the mean power between a steady path and the scattered paths, keeping
    // the total at unity so that S:N and every downstream level stay meaningful.
    if (cfg_.rician_k > 0.0 && power_cnt > 0) {
        const double k = cfg_.rician_k;
        const double los_amp = std::sqrt(k / (k + 1.0));
        const double sc_amp = std::sqrt((1.0 / (k + 1.0)) / diag_.mean_power_gain);
        for (int p = 0; p < paths; ++p) {
            for (std::size_t i = 0; i < taps_i_[p].size(); ++i) {
                taps_i_[p][i] = taps_i_[p][i] * sc_amp + los_amp;
                taps_q_[p][i] *= sc_amp;
            }
            for (std::size_t i = 0; i < diag_.gain_db[p].size(); ++i) {
                const double m2 = taps_i_[p][i] * taps_i_[p][i] + taps_q_[p][i] * taps_q_[p][i];
                diag_.gain_db[p][i] = static_cast<float>(10.0 * std::log10(m2 + 1e-30));
            }
        }
        diag_.mean_power_gain = 1.0;
    }
}

void Renderer::build_noise(std::size_t n, std::size_t guard, double target_power) {
    noise_.assign(n, 0.0f);
    if (n == 0) return;
    if (target_power <= 0.0) return;

    const std::size_t npad = next_pow2(n + 2 * guard);
    std::vector<std::complex<double>> buf(npad, {0.0, 0.0});
    Rng rng(cfg_.seed ^ 0x5DEECE66DULL);
    for (std::size_t i = 0; i < n; ++i) buf[guard + i] = {rng.normal(), 0.0};

    fft(buf, false);
    const double half_bw = 0.5 * cfg_.audio_bw_hz;
    for (std::size_t k = 0; k < npad; ++k) {
        if (std::fabs(bin_frequency(k, npad, fs_)) > half_bw) buf[k] = {0.0, 0.0};
    }
    fft(buf, true);

    for (std::size_t i = 0; i < n; ++i) noise_[i] = static_cast<float>(buf[guard + i].real());
    const double p = mean_square(noise_.data(), n, nullptr);
    if (p > 0.0) {
        const double s = std::sqrt(target_power / p);
        for (auto& v : noise_) v = static_cast<float>(v * s);
    }
}

void Renderer::render(const float* in, std::size_t n, std::vector<float>& out,
                      const std::vector<std::uint8_t>* active) {
    if (n > 0 && in == nullptr) throw std::invalid_argument("in must not be null");
    out.assign(n, 0.0f);
    signal_only_.assign(n, 0.0f);
    noise_.clear();
    signal_power_ = 0.0;
    noise_power_ = 0.0;
    measured_snr_db_ = 0.0;
    if (n == 0) return;

    build_fading(n);

    const std::size_t guard = 8192;
    const std::size_t npad = next_pow2(n + 2 * guard);
    const auto& delays = cfg_.delays_ms();
    const int paths = cfg_.paths();

    // Analytic signal of the input, so that a complex tap gain can rotate phase as
    // well as scale amplitude. This is what produces CW tone flutter, not just QSB.
    std::vector<std::complex<double>> base(npad, {0.0, 0.0});
    for (std::size_t i = 0; i < n; ++i) base[guard + i] = {static_cast<double>(in[i]), 0.0};
    fft(base, false);
    for (std::size_t k = 1; k < npad / 2; ++k) base[k] *= 2.0;
    for (std::size_t k = npad / 2 + 1; k < npad; ++k) base[k] = {0.0, 0.0};
    fft(base, true);

    // IONOS applies the OFFSET before the multipath taps (mixPathSel precedes delayI/Q).
    if (cfg_.freq_offset_hz != 0.0) {
        for (std::size_t i = 0; i < npad; ++i) {
            const double t = (static_cast<double>(i) - static_cast<double>(guard)) / fs_;
            const double ph = kTwoPi * cfg_.freq_offset_hz * t;
            base[i] *= std::complex<double>(std::cos(ph), std::sin(ph));
        }
    }

    std::vector<float> acc(n, 0.0f);
    for (int p = 0; p < paths; ++p) {
        std::vector<std::complex<double>> buf = base;
        fft(buf, false);
        const double tau = (cfg_.base_delay_ms + delays[static_cast<std::size_t>(p)]) * 1e-3;
        for (std::size_t k = 0; k < npad; ++k) {
            const double f = bin_frequency(k, npad, fs_);
            const double ph = -kTwoPi * f * tau;
            buf[k] *= std::complex<double>(std::cos(ph), std::sin(ph));
        }
        fft(buf, true);

        const auto& ti = taps_i_[static_cast<std::size_t>(p)];
        const auto& tq = taps_q_[static_cast<std::size_t>(p)];
        const std::size_t tick_max = ti.size() - 1;
        const double path_rate = diag_.path_tick_rate_hz.empty()
                                     ? diag_.tick_rate_hz
                                     : diag_.path_tick_rate_hz[static_cast<std::size_t>(p)];
        for (std::size_t i = 0; i < n; ++i) {
            const double tick = (path_rate > 0.0) ? static_cast<double>(i) * path_rate / fs_ : 0.0;
            std::size_t k0 = static_cast<std::size_t>(tick);
            if (k0 > tick_max) k0 = tick_max;
            const std::size_t k1 = (k0 < tick_max) ? k0 + 1 : k0;
            double frac = tick - static_cast<double>(k0);
            if (cfg_.fade_interp == FadeInterp::ZeroOrderHold) frac = 0.0;
            if (frac < 0.0) frac = 0.0;
            if (frac > 1.0) frac = 1.0;
            const double gi = ti[k0] * (1.0 - frac) + ti[k1] * frac;
            const double gq = tq[k0] * (1.0 - frac) + tq[k1] * frac;
            const std::complex<double> z = buf[guard + i] * std::complex<double>(gi, gq);
            acc[i] = static_cast<float>(static_cast<double>(acc[i]) + z.real());
        }
    }

    // Channel power normalisation: a fading channel has unit long-term power gain, so
    // speech/morse level is preserved and S:N is referenced to the input level.
    double scale = 1.0;
    if (!cfg_.hardware_staging) {
        double p_in = 0.0;
        for (std::size_t i = 0; i < n; ++i) p_in += static_cast<double>(in[i]) * static_cast<double>(in[i]);
        p_in /= static_cast<double>(n);
        const double p_out = mean_square(acc.data(), n, nullptr);
        if (p_in > 0.0 && p_out > 0.0) scale = std::sqrt(p_in / p_out);
    } else {
        scale = (paths == 1) ? (kIonosWgnSigGain * kIonosPathGain) : 1.0;
    }

    // Flat fading: IONOS Fade() sweeps the target S:N by depth/2 * (1 - cos(2 pi f t)).
    std::vector<double> fade_gain(n, 1.0);
    if (cfg_.fade_depth_db > 0.0 && cfg_.fade_rate_hz > 0.0) {
        for (std::size_t i = 0; i < n; ++i) {
            const double t = static_cast<double>(i) / fs_;
            double d = -0.5 * cfg_.fade_depth_db * (1.0 - std::cos(kTwoPi * cfg_.fade_rate_hz * t));
            if (cfg_.hardware_staging) d = std::floor(d);  // IONOS stores this as an int dB
            if (d < -40.0) d = -40.0;
            fade_gain[i] = std::pow(10.0, d / 20.0);
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        const double v = static_cast<double>(acc[i]) * scale * fade_gain[i];
        signal_only_[i] = static_cast<float>(v);
    }

    // S:N is referenced to key-down signal power, before the fade is applied, so the
    // requested value means "S:N at a fade peak" (IONOS intTargetSN semantics).
    double ps = 0.0;
    {
        std::size_t cnt = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (active && !(*active)[i]) continue;
            const double v = static_cast<double>(acc[i]) * scale;
            ps += v * v;
            ++cnt;
        }
        ps = cnt ? ps / static_cast<double>(cnt) : 0.0;
    }
    signal_power_ = ps;

    for (std::size_t i = 0; i < n; ++i) out[i] = signal_only_[i];

    if (cfg_.hardware_staging) {
        // Reproduce IONOS: fixed-bandwidth noise level, then g_s + g_n = 1.
        const double noise_rms = (cfg_.audio_bw_hz <= 4500.0) ? kIonosNoiseRms3k : kIonosNoiseRms6k;
        const double bw_corr = 10.0 * std::log10(cfg_.audio_bw_hz / (cfg_.audio_bw_hz + 300.0));
        const double ratio = std::pow(10.0, (cfg_.snr_db + kSnCalibrationDb + bw_corr) / 20.0);
        const double g_s = ratio / (1.0 + ratio);
        const double g_n = 1.0 - g_s;
        build_noise(n, guard, noise_rms * noise_rms);
        for (std::size_t i = 0; i < n; ++i)
            out[i] = static_cast<float>(g_s * out[i] + g_n * noise_[i]);
        noise_power_ = g_n * g_n * noise_rms * noise_rms;
        signal_power_ = g_s * g_s * ps;
    } else {
        const double pn = ps / std::pow(10.0, cfg_.snr_db / 10.0);
        build_noise(n, guard, pn);
        for (std::size_t i = 0; i < n; ++i)
            out[i] = static_cast<float>(out[i] + noise_[i]);
        noise_power_ = pn;
    }

    measured_snr_db_ = (noise_power_ > 0.0) ? 10.0 * std::log10(signal_power_ / noise_power_) : 0.0;
}

double Renderer::snr_in_filter_db(double cw_bw_hz) const {
    if (cw_bw_hz <= 0.0) return measured_snr_db_;
    return measured_snr_db_ + 10.0 * std::log10(cfg_.audio_bw_hz / cw_bw_hz);
}

}  // namespace cwsim

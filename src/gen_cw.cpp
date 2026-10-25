// gen_cw.cpp -- CLI: render labelled CW through an IONOS/Watterson channel.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <memory>
#include <string>
#include <vector>

#include "cwchannel.h"
#include "cwkeyer.h"

namespace {

struct Options {
    std::string text;
    std::string text_file;
    int repeat = 1;

    cwsim::CwKeyerConfig keyer;

    std::string channel = "WGN";
    int paths = 0;  // 0 => preset default
    double doppler_hz = -1.0;
    std::vector<double> path_doppler;
    std::string delay_preset;
    std::vector<double> delay_override;
    double rician_k = 0.0;
    double fade_depth_db = 0.0;
    double fade_rate_hz = 0.0;
    double snr_db = 20.0;
    double audio_bw_hz = 3000.0;
    double freq_offset_hz = 0.0;
    std::string interp = "linear";
    bool hardware_staging = false;
    std::uint64_t seed = 1;

    double sample_rate = 44100.0;
    std::string out_wav;
    std::string out_labels;
    std::string fadetrack;
    bool quiet = false;
};

void usage() {
    std::cout <<
        "gen_cw -- labelled CW through an IONOS/Watterson HF channel\n"
        "\n"
        "Text:\n"
        "  --text <string>          message to key\n"
        "  --text-file <path>       read message from file\n"
        "  --repeat <n>             repeat the message n times (default 1)\n"
        "\n"
        "CW keying:\n"
        "  --wpm <n>                effective speed (default 20)\n"
        "  --char-wpm <n>           element speed for Farnsworth (default = wpm)\n"
        "  --tone <hz>              sidetone (default 700)\n"
        "  --risetime <ms>          raised-cosine envelope, per element (default 5)\n"
        "  --dah-ratio <f>          dah:dit duration ratio (ITU 3.0; a measured signal gave 2.77)\n"
        "  --amplitude <f>          key-down amplitude (default 0.5)\n"
        "  --jitter <ms>            uniform +/- timing jitter on every element\n"
        "  --drift <hz>             linear tone drift across the message\n"
        "\n"
        "Channel:\n"
        "  --channel <WGN|MPG|MPM|MPP|MPD>   CCIR F.520 preset (default WGN)\n"
        "  --paths <2|4>            multipath count for MPG..MPD\n"
        "  --doppler <hz>           override the preset 2-sigma Doppler spread\n"
        "  --path-doppler <a,b,..>  per-path Doppler spread, e.g. 0.5,8 for a slow and a fast path\n"
        "  --delay-preset <name>    see --list-delay-presets\n"
        "  --delays <a,b,c>         explicit per-path delays in ms (overrides preset)\n"
        "  --rician-k <f>           steady-path power ratio (0 = pure Rayleigh)\n"
        "  --fade-depth <db>        flat fading depth\n"
        "  --fade <hz>              flat fading rate\n"
        "  --offset <hz>            carrier offset\n"
        "  --interp <linear|zoh>    tap gain interpolation (default linear)\n"
        "  --hardware-staging       replicate IONOS absolute gain chain\n"
        "\n"
        "Levels:\n"
        "  --snr <db>               audio-band S:N at a fade peak (default 20)\n"
        "  --bw <3000|6000>         audio bandwidth (default 3000)\n"
        "\n"
        "Output:\n"
        "  --out <file.wav>         float32 WAV\n"
        "  --labels <file.json>     ground-truth labels\n"
        "  --fadetrack <file.csv>   tap gain curve (path 0)\n"
        "  --fs <n>                 sample rate (default 44100)\n"
        "  --seed <n>               RNG seed (default 1)\n"
        "  --quiet                  suppress the summary\n"
        "  --list-delay-presets     print the CW delay presets and exit\n";
}

bool need_value(int i, int argc, char** argv) {
    if (i + 1 >= argc) {
        std::cerr << "error: " << argv[i] << " needs a value\n";
        return false;
    }
    return true;
}

std::vector<double> parse_csv(const std::string& s) {
    std::vector<double> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        out.push_back(std::atof(item.c_str()));
    }
    return out;
}

std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char b[8];
                    std::snprintf(b, sizeof(b), "\\u%04x", c);
                    o += b;
                } else {
                    o += c;
                }
        }
    }
    return o;
}

void put_le32(std::ostream& os, std::uint32_t v) {
    char b[4] = {static_cast<char>(v & 0xFF), static_cast<char>((v >> 8) & 0xFF),
                 static_cast<char>((v >> 16) & 0xFF), static_cast<char>((v >> 24) & 0xFF)};
    os.write(b, 4);
}
void put_le16(std::ostream& os, std::uint16_t v) {
    char b[2] = {static_cast<char>(v & 0xFF), static_cast<char>((v >> 8) & 0xFF)};
    os.write(b, 2);
}

bool write_wav_float32(const std::string& path, const std::vector<float>& x, double fs) {
    std::ofstream os(path, std::ios::binary);
    if (!os) return false;
    const std::uint16_t channels = 1, bits = 32;
    const std::uint32_t rate = static_cast<std::uint32_t>(std::llround(fs));
    const std::uint32_t data_bytes = static_cast<std::uint32_t>(x.size() * 4);
    os.write("RIFF", 4);
    put_le32(os, 36 + data_bytes);
    os.write("WAVE", 4);
    os.write("fmt ", 4);
    put_le32(os, 16);
    put_le16(os, 3);  // WAVE_FORMAT_IEEE_FLOAT
    put_le16(os, channels);
    put_le32(os, rate);
    put_le32(os, rate * channels * bits / 8);
    put_le16(os, static_cast<std::uint16_t>(channels * bits / 8));
    put_le16(os, bits);
    os.write("data", 4);
    put_le32(os, data_bytes);
    os.write(reinterpret_cast<const char*>(x.data()), static_cast<std::streamsize>(data_bytes));
    return os.good();
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--help" || a == "-h") { usage(); return 0; }
        else if (a == "--list-delay-presets") {
            for (const auto& p : cwsim::cw_delay_presets()) {
                std::cout << p.name << ":";
                for (std::size_t k = 0; k < p.delays_ms.size(); ++k)
                    std::cout << (k ? "," : "") << p.delays_ms[k];
                std::cout << "\n";
            }
            return 0;
        }
        else if (a == "--text" && need_value(i, argc, argv)) opt.text = argv[++i];
        else if (a == "--text-file" && need_value(i, argc, argv)) opt.text_file = argv[++i];
        else if (a == "--repeat" && need_value(i, argc, argv)) opt.repeat = std::max(1, std::atoi(argv[++i]));
        else if (a == "--wpm" && need_value(i, argc, argv)) opt.keyer.wpm = std::atof(argv[++i]);
        else if (a == "--char-wpm" && need_value(i, argc, argv)) opt.keyer.char_wpm = std::atof(argv[++i]);
        else if (a == "--tone" && need_value(i, argc, argv)) opt.keyer.tone_hz = std::atof(argv[++i]);
        else if (a == "--risetime" && need_value(i, argc, argv)) opt.keyer.risetime_ms = std::atof(argv[++i]);
        else if (a == "--dah-ratio" && need_value(i, argc, argv)) opt.keyer.dah_ratio = std::atof(argv[++i]);
        else if (a == "--amplitude" && need_value(i, argc, argv)) opt.keyer.amplitude = std::atof(argv[++i]);
        else if (a == "--jitter" && need_value(i, argc, argv)) opt.keyer.timing_jitter_ms = std::atof(argv[++i]);
        else if (a == "--drift" && need_value(i, argc, argv)) opt.keyer.tone_drift_hz = std::atof(argv[++i]);
        else if (a == "--channel" && need_value(i, argc, argv)) opt.channel = argv[++i];
        else if (a == "--paths" && need_value(i, argc, argv)) opt.paths = std::atoi(argv[++i]);
        else if (a == "--doppler" && need_value(i, argc, argv)) opt.doppler_hz = std::atof(argv[++i]);
        else if (a == "--path-doppler" && need_value(i, argc, argv)) opt.path_doppler = parse_csv(argv[++i]);
        else if (a == "--delay-preset" && need_value(i, argc, argv)) opt.delay_preset = argv[++i];
        else if (a == "--delays" && need_value(i, argc, argv)) opt.delay_override = parse_csv(argv[++i]);
        else if (a == "--rician-k" && need_value(i, argc, argv)) opt.rician_k = std::atof(argv[++i]);
        else if (a == "--fade-depth" && need_value(i, argc, argv)) opt.fade_depth_db = std::atof(argv[++i]);
        else if (a == "--fade" && need_value(i, argc, argv)) opt.fade_rate_hz = std::atof(argv[++i]);
        else if (a == "--snr" && need_value(i, argc, argv)) opt.snr_db = std::atof(argv[++i]);
        else if (a == "--bw" && need_value(i, argc, argv)) opt.audio_bw_hz = std::atof(argv[++i]);
        else if (a == "--offset" && need_value(i, argc, argv)) opt.freq_offset_hz = std::atof(argv[++i]);
        else if (a == "--interp" && need_value(i, argc, argv)) opt.interp = argv[++i];
        else if (a == "--hardware-staging") opt.hardware_staging = true;
        else if (a == "--seed" && need_value(i, argc, argv)) opt.seed = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--fs" && need_value(i, argc, argv)) opt.sample_rate = std::atof(argv[++i]);
        else if (a == "--out" && need_value(i, argc, argv)) opt.out_wav = argv[++i];
        else if (a == "--labels" && need_value(i, argc, argv)) opt.out_labels = argv[++i];
        else if (a == "--fadetrack" && need_value(i, argc, argv)) opt.fadetrack = argv[++i];
        else if (a == "--quiet") opt.quiet = true;
        else {
            std::cerr << "error: unknown option '" << a << "' (try --help)\n";
            return 2;
        }
    }

    if (opt.out_wav.empty()) {
        std::cerr << "error: --out <file.wav> is required\n";
        return 2;
    }

    // --- message ------------------------------------------------------------
    std::string text = opt.text;
    if (!opt.text_file.empty()) {
        std::ifstream f(opt.text_file);
        if (!f) { std::cerr << "error: cannot read " << opt.text_file << "\n"; return 1; }
        std::stringstream ss;
        ss << f.rdbuf();
        text = ss.str();
    }
    if (text.empty()) {
        std::cerr << "error: no message (use --text or --text-file)\n";
        return 2;
    }
    {
        std::string joined;
        for (int r = 0; r < opt.repeat; ++r) {
            if (r) joined += "   ";
            joined += text;
        }
        text = joined;
    }

    // --- channel ------------------------------------------------------------
    cwsim::Mode mode = cwsim::Mode::WGN;
    if (!cwsim::mode_from_name(opt.channel, &mode)) {
        std::cerr << "error: unknown channel '" << opt.channel << "'\n";
        return 2;
    }
    cwsim::ChannelConfig cfg;
    cfg.preset = cwsim::ionos_preset(mode, opt.paths > 0 ? opt.paths : 2);
    const bool doppler_explicit = opt.doppler_hz >= 0.0;
    if (opt.doppler_hz >= 0.0) {
        cfg.preset.doppler_spread_hz = opt.doppler_hz;
        if (opt.doppler_hz > 0.0 && cfg.preset.paths == 1) cfg.preset.paths = 2;
    }
    if (!opt.delay_preset.empty()) {
        bool found = false;
        for (const auto& p : cwsim::cw_delay_presets()) {
            if (opt.delay_preset == p.name) {
                cfg.delay_ms_override = p.delays_ms;
                cfg.preset.paths = static_cast<int>(p.delays_ms.size());
                found = true;
                break;
            }
        }
        if (!found) { std::cerr << "error: unknown delay preset '" << opt.delay_preset << "'\n"; return 2; }
    }
    if (!opt.path_doppler.empty()) {
        cfg.path_doppler_hz = opt.path_doppler;
        cfg.preset.paths = static_cast<int>(opt.path_doppler.size());
        if (!opt.delay_override.empty() && opt.delay_override.size() != opt.path_doppler.size()) {
            std::cerr << "error: --path-doppler and --delays must have the same length\n";
            return 2;
        }
        if (opt.delay_override.empty()) {
            cfg.delay_ms_override = std::vector<double>(opt.path_doppler.size(), 0.0);
            for (std::size_t k = 1; k < opt.path_doppler.size(); ++k)
                cfg.delay_ms_override[k] = 2.0 * static_cast<double>(k);
        }
    }
    if (!opt.delay_override.empty()) {
        cfg.delay_ms_override = opt.delay_override;
        cfg.preset.paths = static_cast<int>(opt.delay_override.size());
    }
    // Only a fading channel needs a Doppler spread invented for it. Applying this to WGN
    // turned "--channel WGN --delays 0,2" into MPP silently: the two are then identical,
    // which defeats the point of having a no-fading mode at all. WGN with explicit delays
    // is static multipath (a comb filter), which is a legitimate configuration.
    if (!doppler_explicit && cfg.preset.mode != cwsim::Mode::WGN && cfg.preset.paths > 1 &&
        cfg.preset.doppler_spread_hz <= 0.0)
        cfg.preset.doppler_spread_hz = 1.0;

    // Keep the scalar summary consistent with a per-path specification.
    if (!cfg.path_doppler_hz.empty()) cfg.preset.doppler_spread_hz = cfg.path_doppler_hz[0];
    cfg.rician_k = opt.rician_k;
    cfg.fade_depth_db = opt.fade_depth_db;
    cfg.fade_rate_hz = opt.fade_rate_hz;
    cfg.snr_db = opt.snr_db;
    cfg.audio_bw_hz = opt.audio_bw_hz;
    cfg.freq_offset_hz = opt.freq_offset_hz;
    cfg.fade_interp = (opt.interp == "zoh") ? cwsim::FadeInterp::ZeroOrderHold : cwsim::FadeInterp::Linear;
    cfg.hardware_staging = opt.hardware_staging;
    cfg.seed = opt.seed;
    opt.keyer.seed = opt.seed;

    // --- keying -------------------------------------------------------------
    cwsim::CwKeyed keyed;
    try {
        keyed = cwsim::key_text(text, opt.keyer, opt.sample_rate);
    } catch (const std::exception& e) {
        std::cerr << "error: keying failed: " << e.what() << "\n";
        return 1;
    }
    if (keyed.audio.empty()) {
        std::cerr << "error: message contains no keyable characters\n";
        return 2;
    }

    // --- validate -----------------------------------------------------------
    // Every combination below used to be accepted silently while doing nothing, or in one
    // case crashing: `--channel WGN --doppler 1` raised `paths` to 2 without extending the
    // delay list, so the Renderer constructor threw and the process terminated.
    if (mode == cwsim::Mode::WGN) {
        if (opt.doppler_hz > 0.0) {
            std::cerr << "error: --channel WGN has no fading, so --doppler cannot be applied.\n"
                         "       Use --channel MPP (or --path-doppler) for a fading channel.\n";
            return 2;
        }
        if (opt.paths > 1) {
            std::cerr << "warning: --paths is ignored for WGN (it is always a single path).\n";
        }
        if (opt.rician_k > 0.0) {
            std::cerr << "warning: --rician-k has no effect on WGN; there is no fading to make\n"
                         "         Rician. Use a multipath channel instead.\n";
            cfg.rician_k = 0.0;  // keep the labels honest rather than reporting an unused value
        }
    }
    if (opt.fade_depth_db > 0.0 && opt.fade_rate_hz <= 0.0) {
        std::cerr << "warning: --fade-depth has no effect without --fade; ignoring it.\n";
        cfg.fade_depth_db = 0.0;
    }
    if (opt.fade_rate_hz > 0.0 && opt.fade_depth_db <= 0.0) {
        std::cerr << "warning: --fade has no effect without --fade-depth; ignoring it.\n";
        cfg.fade_rate_hz = 0.0;
    }
    if (!opt.delay_preset.empty() && mode == cwsim::Mode::WGN) {
        std::cerr << "warning: --delay-preset gives WGN static multipath with no fading.\n";
    }

    // --- render -------------------------------------------------------------
    // Pad the tail so the longest tap delay does not truncate the last elements; the
    // labels are shifted by the path-0 delay afterwards so they line up with the audio.
    const auto& path_delays = cfg.delays_ms();
    double max_delay_ms = cfg.base_delay_ms;
    for (double d : path_delays) max_delay_ms = std::max(max_delay_ms, cfg.base_delay_ms + d);
    const std::size_t tail = static_cast<std::size_t>(std::llround(max_delay_ms * 1e-3 * opt.sample_rate)) +
                             static_cast<std::size_t>(std::llround(0.05 * opt.sample_rate));
    std::vector<float> padded = keyed.audio;
    padded.resize(keyed.audio.size() + tail, 0.0f);
    std::vector<std::uint8_t> padded_active = keyed.active;
    padded_active.resize(keyed.active.size() + tail, 0);

    const std::size_t label_offset = static_cast<std::size_t>(
        std::llround((cfg.base_delay_ms + path_delays.at(0)) * 1e-3 * opt.sample_rate));

    std::unique_ptr<cwsim::Renderer> renderer_holder;
    try {
        renderer_holder.reset(new cwsim::Renderer(cfg, opt.sample_rate));
    } catch (const std::exception& e) {
        std::cerr << "error: invalid channel configuration: " << e.what() << "\n";
        return 2;
    }
    cwsim::Renderer& renderer = *renderer_holder;
    std::vector<float> out;
    try {
        renderer.render(padded.data(), padded.size(), out, &padded_active);
    } catch (const std::exception& e) {
        std::cerr << "error: render failed: " << e.what() << "\n";
        return 1;
    }

    if (!write_wav_float32(opt.out_wav, out, opt.sample_rate)) {
        std::cerr << "error: cannot write " << opt.out_wav << "\n";
        return 1;
    }

    if (!opt.fadetrack.empty() && !renderer.fading().gain_db.empty()) {
        std::ofstream f(opt.fadetrack);
        if (f) {
            f << "tick,time_s,gain_db\n";
            const auto& curve = renderer.fading().gain_db[0];
            const double tick_rate = renderer.fading().tick_rate_hz;
            for (std::size_t k = 0; k < curve.size(); ++k)
                f << k << "," << (tick_rate > 0.0 ? static_cast<double>(k) / tick_rate : 0.0) << ","
                  << curve[k] << "\n";
        }
    }

    // --- labels -------------------------------------------------------------
    if (!opt.out_labels.empty()) {
        std::ofstream f(opt.out_labels);
        if (!f) { std::cerr << "error: cannot write " << opt.out_labels << "\n"; return 1; }
        const auto& delays = cfg.delays_ms();
        const auto& fd = renderer.fading();

        f << "{\n";
        f << "  \"generator\": \"cwsim 0.1\",\n";
        f << "  \"text\": \"" << json_escape(text) << "\",\n";
        f << "  \"sample_rate\": " << opt.sample_rate << ",\n";
        f << "  \"duration_s\": " << (static_cast<double>(out.size()) / opt.sample_rate) << ",\n";
        f << "  \"channel\": {\n";
        f << "    \"mode\": \"" << cwsim::mode_name(cfg.preset.mode) << "\",\n";
        f << "    \"paths\": " << cfg.paths() << ",\n";
        f << "    \"doppler_spread_hz\": " << cfg.preset.doppler_spread_hz << ",\n";
        f << "    \"path_doppler_hz\": [";
        for (std::size_t k = 0; k < fd.path_tick_rate_hz.size(); ++k)
            f << (k ? ", " : "") << (fd.path_tick_rate_hz[k] / 64.0);
        f << "],\n";
        f << "    \"delays_ms\": [";
        for (std::size_t k = 0; k < delays.size(); ++k) f << (k ? ", " : "") << delays[k];
        f << "],\n";
        f << "    \"base_delay_ms\": " << cfg.base_delay_ms << ",\n";
        f << "    \"rician_k\": " << cfg.rician_k << ",\n";
        f << "    \"fade_depth_db\": " << cfg.fade_depth_db << ",\n";
        f << "    \"fade_rate_hz\": " << cfg.fade_rate_hz << ",\n";
        f << "    \"freq_offset_hz\": " << cfg.freq_offset_hz << ",\n";
        f << "    \"interp\": \"" << (cfg.fade_interp == cwsim::FadeInterp::Linear ? "linear" : "zoh") << "\",\n";
        f << "    \"tick_rate_hz\": " << fd.tick_rate_hz << ",\n";
        f << "    \"mean_power_gain\": " << fd.mean_power_gain << ",\n";
        f << "    \"hardware_staging\": " << (cfg.hardware_staging ? "true" : "false") << ",\n";
        f << "    \"seed\": " << cfg.seed << "\n";
        f << "  },\n";
        f << "  \"snr\": {\n";
        f << "    \"requested_audio_band_db\": " << cfg.snr_db << ",\n";
        f << "    \"measured_audio_band_db\": " << renderer.measured_snr_db() << ",\n";
        f << "    \"audio_bandwidth_hz\": " << cfg.audio_bw_hz << ",\n";
        f << "    \"in_filter_250hz_db\": " << renderer.snr_in_filter_db(250.0) << ",\n";
        f << "    \"in_filter_500hz_db\": " << renderer.snr_in_filter_db(500.0) << ",\n";
        f << "    \"reference\": \"key-down signal power vs total noise power in audio_bandwidth_hz\"\n";
        f << "  },\n";
        f << "  \"cw\": {\n";
        f << "    \"wpm\": " << opt.keyer.wpm << ",\n";
        f << "    \"char_wpm\": " << (opt.keyer.char_wpm > 0.0 ? opt.keyer.char_wpm : opt.keyer.wpm) << ",\n";
        f << "    \"tone_hz\": " << opt.keyer.tone_hz << ",\n";
        f << "    \"risetime_ms\": " << opt.keyer.risetime_ms << ",\n";
        f << "    \"dah_ratio\": " << opt.keyer.dah_ratio << ",\n";
        f << "    \"amplitude\": " << opt.keyer.amplitude << ",\n";
        f << "    \"timing_jitter_ms\": " << opt.keyer.timing_jitter_ms << ",\n";
        f << "    \"tone_drift_hz\": " << opt.keyer.tone_drift_hz << "\n";
        f << "  },\n";

        // Per-character channel state, measured on the rendered signal rather than on a
        // single path: two independent Rayleigh paths combine to something much
        // shallower than either, so a per-path number would overstate the fade.
        const std::vector<float>& sig = renderer.signal_only();
        double p_long = 0.0;
        std::size_t p_cnt = 0;
        for (std::size_t i = 0; i < sig.size() && i < padded_active.size(); ++i) {
            if (!padded_active[i]) continue;
            p_long += static_cast<double>(sig[i]) * static_cast<double>(sig[i]);
            ++p_cnt;
        }
        p_long = p_cnt ? p_long / static_cast<double>(p_cnt) : 0.0;

        f << "  \"characters\": [\n";
        for (std::size_t k = 0; k < keyed.characters.size(); ++k) {
            const auto& c = keyed.characters[k];
            double min_gain = 0.0;
            if (p_long > 0.0 && c.end > c.start && c.end <= sig.size()) {
                const std::size_t len = c.end - c.start;
                std::size_t win = static_cast<std::size_t>(std::llround(0.010 * opt.sample_rate));
                win = std::max<std::size_t>(16, std::min<std::size_t>(win, std::max<std::size_t>(1, len / 2)));
                const std::size_t hop = std::max<std::size_t>(1, win / 2);
                double worst = 1e300;
                for (std::size_t st = c.start; st + win <= c.end; st += hop) {
                    // Only average over key-down samples: a character's span includes its
                    // inter-element gaps, and silence there is not a fade.
                    double acc = 0.0;
                    std::size_t cnt = 0;
                    for (std::size_t j = 0; j < win; ++j) {
                        if (st + j >= padded_active.size() || !padded_active[st + j]) continue;
                        const double v = sig[st + j];
                        acc += v * v;
                        ++cnt;
                    }
                    // Require a fully key-down window: a window that straddles an element
                    // edge would measure the raised-cosine rise/fall, not the channel.
                    if (cnt != win) continue;
                    worst = std::min(worst, acc / static_cast<double>(cnt));
                }
                if (worst > 1e299) {
                    // Very short element: fall back to the whole span.
                    double acc = 0.0;
                    std::size_t cnt = 0;
                    for (std::size_t j = c.start; j < c.end; ++j) {
                        if (!padded_active[j]) continue;
                        const double v = sig[j];
                        acc += v * v;
                        ++cnt;
                    }
                    if (cnt) worst = acc / static_cast<double>(cnt);
                }
                if (worst < 1e299) min_gain = 10.0 * std::log10(worst / p_long + 1e-30);
            }
            f << "    {\"index\": " << k << ", \"char\": \"" << json_escape(std::string(1, c.character))
              << "\", \"start\": " << (c.start + label_offset) << ", \"end\": " << (c.end + label_offset)
              << ", \"min_gain_db\": " << min_gain << "}";
            f << (k + 1 < keyed.characters.size() ? ",\n" : "\n");
        }
        f << "  ],\n";

        f << "  \"symbols\": [\n";
        for (std::size_t k = 0; k < keyed.symbols.size(); ++k) {
            const auto& s = keyed.symbols[k];
            f << "    {\"kind\": \"" << cwsim::symbol_kind_name(s.kind) << "\", \"char\": \""
              << (s.character ? json_escape(std::string(1, s.character)) : "")
              << "\", \"start\": " << (s.start + label_offset) << ", \"end\": " << (s.end + label_offset)
              << "}";
            f << (k + 1 < keyed.symbols.size() ? ",\n" : "\n");
        }
        f << "  ]\n";
        f << "}\n";
    }

    if (!opt.quiet) {
        std::cout << "channel   : " << cwsim::mode_name(cfg.preset.mode) << "  paths=" << cfg.paths()
                  << "  spread=" << cfg.preset.doppler_spread_hz << " Hz"
                  << "  tick=" << renderer.fading().tick_rate_hz << " Hz\n";
        std::cout << "fading    : mean|g|^2=" << renderer.fading().mean_power_gain;
        if (cfg.fade_depth_db > 0.0)
            std::cout << "  fade " << cfg.fade_depth_db << " dB @ " << cfg.fade_rate_hz << " Hz";
        std::cout << "\n";
        std::cout << "snr       : requested " << cfg.snr_db << " dB  measured " << renderer.measured_snr_db()
                  << " dB  (audio band " << cfg.audio_bw_hz << " Hz)\n";
        std::cout << "            in a 250 Hz filter: " << renderer.snr_in_filter_db(250.0) << " dB\n";
        std::cout << "keying    : " << keyed.characters.size() << " chars, " << keyed.symbols.size()
                  << " symbols, " << opt.keyer.wpm << " wpm, " << opt.keyer.tone_hz << " Hz tone\n";
        std::cout << "duration  : " << (static_cast<double>(out.size()) / opt.sample_rate) << " s\n";
        std::cout << "wrote     : " << opt.out_wav << "\n";
    }
    return 0;
}

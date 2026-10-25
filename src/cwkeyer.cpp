// cwkeyer.cpp -- Morse keyer with per-symbol ground truth.
#include "cwkeyer.h"

#include <algorithm>
#include <cctype>
#include <cmath>

namespace cwsim {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;

struct MorseEntry {
    char c;
    const char* pattern;
};

// ITU-R M.1677 patterns for the characters a CW decoder is normally asked to copy.
const MorseEntry kMorse[] = {
    {'A', ".-"},      {'B', "-..."},    {'C', "-.-."},    {'D', "-.."},     {'E', "."},
    {'F', "..-."},    {'G', "--."},     {'H', "...."},    {'I', ".."},      {'J', ".---"},
    {'K', "-.-"},     {'L', ".-.."},    {'M', "--"},      {'N', "-."},      {'O', "---"},
    {'P', ".--."},    {'Q', "--.-"},    {'R', ".-."},     {'S', "..."},     {'T', "-"},
    {'U', "..-"},     {'V', "...-"},    {'W', ".--"},     {'X', "-..-"},    {'Y', "-.--"},
    {'Z', "--.."},    {'0', "-----"},   {'1', ".----"},   {'2', "..---"},   {'3', "...--"},
    {'4', "....-"},   {'5', "....."},   {'6', "-...."},   {'7', "--..."},   {'8', "---.."},
    {'9', "----."},   {'/', "-..-."},   {'?', "..--.."},  {'.', ".-.-.-"},  {',', "--..--"},
    {':', "---..."},  {';', "-.-.-."},  {'\'', ".----."}, {'-', "-....-"},  {'(', "-.--."},
    {')', "-.--.-"},  {'"', ".-..-."},  {'=', "-...-"},   {'+', ".-.-."},   {'@', ".--.-."},
};

struct Event {
    double units = 0.0;
    bool on = false;
    SymbolKind kind = SymbolKind::Dit;
    char character = '\0';
};

class Rng {
  public:
    explicit Rng(std::uint64_t seed) : s_(seed) {}
    double uniform01() {
        std::uint64_t z = (s_ += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z ^= z >> 31;
        return static_cast<double>(z >> 11) * 0x1.0p-53;
    }

  private:
    std::uint64_t s_;
};

}  // namespace

const char* symbol_kind_name(SymbolKind k) {
    switch (k) {
        case SymbolKind::Dit: return "dit";
        case SymbolKind::Dah: return "dah";
        case SymbolKind::ElementGap: return "element_gap";
        case SymbolKind::CharGap: return "char_gap";
        case SymbolKind::WordGap: return "word_gap";
    }
    return "?";
}

const char* morse_pattern(char c) {
    const char u = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    for (const auto& e : kMorse)
        if (e.c == u) return e.pattern;
    return nullptr;
}

CwKeyed key_text(const std::string& text, const CwKeyerConfig& cfg, double sample_rate) {
    CwKeyed out;
    out.text = text;
    if (sample_rate <= 0.0 || cfg.wpm <= 0.0) return out;

    const double char_wpm = (cfg.char_wpm > cfg.wpm) ? cfg.char_wpm : cfg.wpm;
    const double unit_char = 1.2 / char_wpm;  // seconds per element unit
    const double dah_ratio = (cfg.dah_ratio > 0.0) ? cfg.dah_ratio : 3.0;
    const double unit_eff = 1.2 / cfg.wpm;

    // --- build the element/gap sequence -------------------------------------
    std::vector<Event> events;
    double pending_gap_units = 0.0;
    std::size_t n_chars = 0;
    for (char raw : text) {
        const char c = static_cast<char>(std::toupper(static_cast<unsigned char>(raw)));
        if (c == ' ' || c == '\n' || c == '\t') {
            pending_gap_units = 7.0;
            continue;
        }
        const char* pat = morse_pattern(c);
        if (pat == nullptr) continue;

        if (n_chars > 0) {
            Event g;
            g.units = (pending_gap_units > 0.0) ? pending_gap_units : 3.0;
            g.on = false;
            g.kind = (pending_gap_units > 0.0) ? SymbolKind::WordGap : SymbolKind::CharGap;
            events.push_back(g);
        }
        pending_gap_units = 0.0;

        bool first_elem = true;
        for (const char* s = pat; *s; ++s) {
            if (!first_elem) {
                Event g;
                g.units = 1.0;
                g.on = false;
                g.kind = SymbolKind::ElementGap;
                g.character = c;
                events.push_back(g);
            }
            Event e;
            e.units = (*s == '.') ? 1.0 : dah_ratio;
            e.on = true;
            e.kind = (*s == '.') ? SymbolKind::Dit : SymbolKind::Dah;
            e.character = c;
            events.push_back(e);
            first_elem = false;
        }
        ++n_chars;
    }
    if (events.empty()) return out;

    // --- Farnsworth: stretch character/word gaps to hit the requested speed --
    double total_units = 0.0;
    int gap_count = 0;
    for (const auto& e : events) {
        total_units += e.units;
        if (e.kind == SymbolKind::CharGap || e.kind == SymbolKind::WordGap) ++gap_count;
    }
    // Farnsworth: elements keep their character speed, and the surplus time is spread
    // over the character/word gaps. With char_wpm == wpm the surplus is zero.
    double extra_per_gap = 0.0;
    if (gap_count > 0) {
        const double t_target = total_units * unit_eff;
        const double t_flat = total_units * unit_char;
        extra_per_gap = std::max(0.0, (t_target - t_flat) / static_cast<double>(gap_count));
    }

    // --- realise sample boundaries -----------------------------------------
    Rng rng(cfg.seed);
    const double rise = std::max(0.2e-3, cfg.risetime_ms * 1e-3);
    std::size_t pos = 0;
    std::vector<std::size_t> starts(events.size()), ends(events.size());
    for (std::size_t k = 0; k < events.size(); ++k) {
        double secs = events[k].units * unit_char;
        if (events[k].kind == SymbolKind::CharGap || events[k].kind == SymbolKind::WordGap)
            secs += extra_per_gap;
        if (cfg.timing_jitter_ms > 0.0) {
            const double j = (rng.uniform01() * 2.0 - 1.0) * cfg.timing_jitter_ms * 1e-3;
            secs = std::max(1e-4, secs + j);
        }
        const std::size_t len = static_cast<std::size_t>(std::llround(secs * sample_rate));
        starts[k] = pos;
        ends[k] = pos + std::max<std::size_t>(len, 1);
        pos = ends[k];
    }

    out.audio.assign(pos, 0.0f);
    out.active.assign(pos, 0);
    out.symbols.reserve(events.size());

    const double total_s = static_cast<double>(pos) / sample_rate;
    double phase = 0.0;
    for (std::size_t k = 0; k < events.size(); ++k) {
        CwSymbol sym;
        sym.kind = events[k].kind;
        sym.character = events[k].character;
        sym.start = starts[k];
        sym.end = ends[k];
        out.symbols.push_back(sym);

        if (events[k].on) {
            const std::size_t len = ends[k] - starts[k];
            for (std::size_t j = 0; j < len; ++j) {
                const std::size_t i = starts[k] + j;
                const double t = static_cast<double>(i) / sample_rate;
                const double f = cfg.tone_hz + cfg.tone_drift_hz * (total_s > 0.0 ? t / total_s : 0.0);
                // Raised-cosine rise/fall; degenerates gracefully for very short elements.
                const double d = static_cast<double>(std::min(j, len - 1 - j)) / sample_rate;
                const double w = std::min(1.0, d / rise);
                const double env = 0.5 * (1.0 - std::cos(kPi * w));
                out.audio[i] = static_cast<float>(cfg.amplitude * env * std::sin(phase));
                out.active[i] = 1;
                phase += kTwoPi * f / sample_rate;
            }
        } else {
            // Keep the oscillator running so phase is continuous across gaps.
            const std::size_t len = ends[k] - starts[k];
            phase += kTwoPi * cfg.tone_hz * static_cast<double>(len) / sample_rate;
        }
    }

    // --- per-character labels ----------------------------------------------
    std::vector<CwCharacter> chars;
    for (std::size_t k = 0; k < events.size(); ++k) {
        if (!events[k].on) continue;
        if (!chars.empty() && chars.back().character == events[k].character) {
            chars.back().end = ends[k];
        } else {
            CwCharacter ch;
            ch.character = events[k].character;
            ch.start = starts[k];
            ch.end = ends[k];
            chars.push_back(ch);
        }
    }
    out.characters = std::move(chars);
    out.duration_s = total_s;
    return out;
}

}  // namespace cwsim

// cwkeyer.h -- Morse keyer with per-symbol ground truth.
//
// Produces an ideal CW audio envelope and, alongside it, the sample-exact index of
// every element. That labelling is the reason this generator exists: the text is
// known, so the decoder's training/verification target is free.
#ifndef CWSIM_CWKEYER_H
#define CWSIM_CWKEYER_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cwsim {

enum class SymbolKind { Dit, Dah, ElementGap, CharGap, WordGap };

const char* symbol_kind_name(SymbolKind k);

struct CwSymbol {
    SymbolKind kind = SymbolKind::Dit;
    char character = '\0';  // owning character, for Dit/Dah
    std::size_t start = 0;  // sample index, inclusive
    std::size_t end = 0;    // sample index, exclusive
};

struct CwCharacter {
    char character = '\0';
    std::size_t start = 0;
    std::size_t end = 0;
};

struct CwKeyerConfig {
    double wpm = 20.0;          // effective speed
    double char_wpm = 0.0;      // element speed for Farnsworth; 0 => same as wpm
    double tone_hz = 700.0;
    double risetime_ms = 5.0;   // raised-cosine rise and fall per element
    // Dah:dit duration ratio. The ITU value is 3.0, but real operators are not exact:
    // a measured 40 m signal gave 2.773 +/- 0.053 over six independent 30 s segments,
    // so the ideal ratio is a modelling choice, not a fact.
    double dah_ratio = 3.0;
    double amplitude = 0.5;
    double timing_jitter_ms = 0.0;  // uniform +/- jitter on every element boundary
    double tone_drift_hz = 0.0;     // linear drift across the message
    std::uint64_t seed = 1;
};

struct CwKeyed {
    std::string text;
    std::vector<float> audio;
    std::vector<std::uint8_t> active;  // 1 during key-down
    std::vector<CwSymbol> symbols;
    std::vector<CwCharacter> characters;
    double duration_s = 0.0;
};

// Morse pattern for a character, or nullptr if unsupported.
const char* morse_pattern(char c);

// `text` is upper-cased internally; unsupported characters are skipped.
CwKeyed key_text(const std::string& text, const CwKeyerConfig& cfg, double sample_rate);

}  // namespace cwsim

#endif  // CWSIM_CWKEYER_H

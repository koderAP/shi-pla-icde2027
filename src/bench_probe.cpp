// One-off measurements for the revision-4 questions.
// The copy of the index finishes before the delete clock starts.
// Zeroing the piece tail is off unless --zero is set, and it is one 8-byte store.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <sys/resource.h>
#include <vector>

#include "pgm/pgm_index.hpp"
#include "shi/shi_pla.hpp"

namespace {

std::vector<uint64_t> load_file(const std::string &path, size_t max_keys, bool uint32_keys) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    uint64_t count = 0;
    in.read(reinterpret_cast<char *>(&count), sizeof(count));
    if (!in) throw std::runtime_error("short header");
    size_t n = static_cast<size_t>(count);
    if (max_keys > 0 && max_keys < n) n = max_keys;
    std::vector<uint64_t> keys(n);
    if (!uint32_keys) {
        in.read(reinterpret_cast<char *>(keys.data()), static_cast<std::streamsize>(n * sizeof(uint64_t)));
    } else {
        std::vector<uint32_t> raw(n);
        in.read(reinterpret_cast<char *>(raw.data()), static_cast<std::streamsize>(n * sizeof(uint32_t)));
        for (size_t i = 0; i < n; ++i) keys[i] = raw[i];
    }
    if (!in) throw std::runtime_error("short payload");
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    return keys;
}

shi::SHIIndex build_index(const std::vector<uint64_t> &keys, uint32_t modulus, bool upper_anchors = true,
                          bool drop_half = false) {
    return shi::SHIIndex(keys, 16, 0x5A175EEDULL, modulus, shi::PLAMode::OptimalHull, 4, upper_anchors, drop_half);
}

bool packed_keys_equal(const shi::SHIIndex &a, const shi::SHIIndex &b) {
    if (a.size() != b.size()) return false;
    std::vector<uint64_t> left(65536), right(65536);
    size_t off = 0;
    while (off < a.size()) {
        size_t n = std::min(left.size(), a.size() - off);
        if (a.copy_keys(off, left.data(), n) != n) return false;
        if (b.copy_keys(off, right.data(), n) != n) return false;
        if (std::memcmp(left.data(), right.data(), n * sizeof(uint64_t)) != 0) return false;
        off += n;
    }
    return true;
}

void identity_checks(const std::vector<uint64_t> &keys, size_t checks, uint32_t modulus, bool upper_anchors,
                     bool drop_half, bool zero) {
    std::mt19937_64 rng(20261022);
    std::uniform_int_distribution<size_t> pick(0, keys.size() - 1);
    size_t repr_match = 0;
    size_t key_match = 0;
    size_t both = 0;
    double us = 0;
    for (size_t i = 0; i < checks; ++i) {
        size_t rank = pick(rng);
        uint64_t key = keys[rank];
        std::vector<uint64_t> survivors;
        survivors.reserve(keys.size() - 1);
        survivors.insert(survivors.end(), keys.begin(), keys.begin() + static_cast<std::ptrdiff_t>(rank));
        survivors.insert(survivors.end(), keys.begin() + static_cast<std::ptrdiff_t>(rank + 1), keys.end());
        shi::SHIIndex live = build_index(keys, modulus, upper_anchors, drop_half);
        live.set_zero_tail(zero);
        auto t0 = std::chrono::steady_clock::now();
        live.erase(key);
        auto t1 = std::chrono::steady_clock::now();
        us += std::chrono::duration<double, std::micro>(t1 - t0).count();
        shi::SHIIndex fresh = build_index(survivors, modulus, upper_anchors, drop_half);
        bool repr = live.serialize_bytes() == fresh.serialize_bytes();
        bool packed = packed_keys_equal(live, fresh);
        if (repr) ++repr_match;
        if (packed) ++key_match;
        if (repr && packed) ++both;
        std::cout << "CHECK " << i << " rank " << rank << " repr " << repr << " packed_keys " << packed
                  << " us " << std::chrono::duration<double, std::micro>(t1 - t0).count() << std::endl;
    }
    std::cout << "IDENTITY n " << checks
              << " drop_half " << (drop_half ? 1 : 0)
              << " zero " << (zero ? 1 : 0)
              << " modulus " << modulus
              << " repr_match " << repr_match
              << " packed_keys_match " << key_match
              << " both " << both
              << " mean_us " << (checks ? us / static_cast<double>(checks) : 0)
              << std::endl;
}

long page_faults() {
    struct rusage ru {};
    if (getrusage(RUSAGE_SELF, &ru) != 0) return -1;
    return ru.ru_minflt + ru.ru_majflt;
}

void print_levels(const std::vector<uint64_t> &keys, bool drop_half) {
    pgm::PGMIndex<uint64_t, 16, 4> pgm(keys);
    const auto &off = pgm.debug_levels_offsets();
    std::cout << "PGM bytes " << pgm.size_in_bytes()
              << " height " << pgm.height()
              << " sizeof_segment " << sizeof(pgm.debug_segments()[0])
              << " all_segments " << pgm.debug_segments().size();
    for (size_t l = 0; l + 1 < off.size(); ++l) {
        size_t span = off[l + 1] - off[l];
        size_t nseg = span == 0 ? 0 : span - 1;
        std::cout << " level" << l << " " << nseg;
    }
    std::cout << std::endl;

    for (uint32_t m : {0u, 1024u, 4096u, 16384u}) {
        shi::SHIIndex idx = build_index(keys, m, true, drop_half);
        auto counts = idx.level_segment_counts();
        std::cout << (drop_half ? "SHI_DROP2" : "SHI") << " M " << m << " bytes " << idx.model_bytes() << " levels";
        for (size_t c : counts) std::cout << " " << c;
        std::cout << std::endl;
    }
}

struct Acc {
    size_t n = 0;
    size_t start_changed = 0;
    size_t tails = 0;
    long faults = 0;
    long maj_only = 0;
    double us = 0;
    double us_changed = 0;
    double us_same = 0;
    double move_ns = 0;
    double merge_ns = 0;
    double prefix_ns = 0;
    double refit_ns = 0;
    double fit_ns = 0;
    double upper_ns = 0;
    double span_ns = 0;
    size_t span_rewritten = 0;
    size_t uncleared = 0;
};

void add_sample(Acc &a, double us, bool changed, bool tail, long dfault, long dmaj,
                const shi::KeyStore::EditTrace &ed, uint64_t refit_ns, uint64_t fit_ns, uint64_t upper_ns,
                uint64_t span_ns, size_t span_rewritten) {
    a.n += 1;
    a.us += us;
    a.faults += dfault;
    a.maj_only += dmaj;
    a.move_ns += static_cast<double>(ed.move_ns);
    a.merge_ns += static_cast<double>(ed.merge_ns);
    a.prefix_ns += static_cast<double>(ed.prefix_ns);
    a.refit_ns += static_cast<double>(refit_ns);
    a.fit_ns += static_cast<double>(fit_ns);
    a.upper_ns += static_cast<double>(upper_ns);
    a.span_ns += static_cast<double>(span_ns);
    a.span_rewritten += span_rewritten;
    if (tail && ed.vacated_after != 0) a.uncleared += 1;
    if (changed) {
        a.start_changed += 1;
        a.us_changed += us;
    } else {
        a.us_same += us;
    }
    if (tail) a.tails += 1;
}

void report(const char *tag, const Acc &a) {
    double n = static_cast<double>(a.n);
    size_t same = a.n - a.start_changed;
    std::cout << tag
              << " n " << a.n
              << " start_key_changed " << a.start_changed
              << " mean_us_changed " << (a.start_changed ? a.us_changed / static_cast<double>(a.start_changed) : 0)
              << " mean_us_same " << (same ? a.us_same / static_cast<double>(same) : 0)
              << " tails " << a.tails
              << " mean_us " << (n ? a.us / n : 0)
              << " faults " << a.faults
              << " major " << a.maj_only
              << " mean_move_us " << (n ? a.move_ns / n / 1e3 : 0)
              << " mean_merge_us " << (n ? a.merge_ns / n / 1e3 : 0)
              << " mean_prefix_us " << (n ? a.prefix_ns / n / 1e3 : 0)
              << " mean_refit_us " << (n ? a.refit_ns / n / 1e3 : 0)
              << " mean_fit_us " << (n ? a.fit_ns / n / 1e3 : 0)
              << " mean_upper_us " << (n ? a.upper_ns / n / 1e3 : 0)
              << " mean_span_us " << (n ? a.span_ns / n / 1e3 : 0)
              << " span_rewritten " << a.span_rewritten
              << " uncleared " << a.uncleared
              << std::endl;
}

std::vector<size_t> sample_ranks(size_t n, size_t k) {
    std::mt19937_64 rng(20261022);
    std::uniform_int_distribution<size_t> pick(0, n - 1);
    std::vector<size_t> ranks;
    ranks.reserve(k);
    while (ranks.size() < k) {
        size_t r = pick(rng);
        if (std::find(ranks.begin(), ranks.end(), r) == ranks.end()) ranks.push_back(r);
    }
    return ranks;
}

bool starts_differ(const std::vector<uint64_t> &a, const std::vector<uint64_t> &b) {
    return a != b;
}

void sample_deletes(const std::vector<uint64_t> &keys, size_t k, bool zero, uint32_t modulus, bool upper_anchors) {
    shi::SHIIndex base = build_index(keys, modulus, upper_anchors);
    std::vector<size_t> ranks = sample_ranks(keys.size(), k);
    Acc acc;
    for (size_t i = 0; i < ranks.size(); ++i) {
        size_t rank = ranks[i];
        uint64_t key = keys[rank];
        shi::SHIIndex live = base;
        live.set_zero_tail(zero);
        bool tail = live.key_is_piece_tail(key);
        std::vector<uint64_t> before = live.leaf_start_keys();
        struct rusage ru0 {};
        getrusage(RUSAGE_SELF, &ru0);
        auto t0 = std::chrono::steady_clock::now();
        shi::RefitStats st = live.erase(key);
        auto t1 = std::chrono::steady_clock::now();
        struct rusage ru1 {};
        getrusage(RUSAGE_SELF, &ru1);
        std::vector<uint64_t> after = live.leaf_start_keys();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        long df = (ru1.ru_minflt - ru0.ru_minflt) + (ru1.ru_majflt - ru0.ru_majflt);
        long dm = ru1.ru_majflt - ru0.ru_majflt;
        add_sample(acc, us, starts_differ(before, after), tail, df, dm, live.last_edit(), live.last_refit_ns(), st.fit_ns, st.upper_ns, st.span_ns, st.span_rewritten);
        if ((i + 1) % 32 == 0 || i + 1 == ranks.size()) {
            std::cout << "PROGRESS sample " << (i + 1) << "/" << ranks.size() << std::endl;
        }
    }
    std::cout << "MOD " << modulus << " upper_anchors " << (upper_anchors ? 1 : 0) << " bytes " << base.model_bytes();
    auto counts = base.level_segment_counts();
    for (size_t c : counts) std::cout << " " << c;
    std::cout << std::endl;
    report(zero ? "SAMPLE_ZERO" : "SAMPLE", acc);
    if (!upper_anchors) {
        size_t match = 0;
        size_t checks = std::min<size_t>(8, ranks.size());
        for (size_t i = 0; i < checks; ++i) {
            size_t rank = ranks[i];
            std::vector<uint64_t> survivors;
            survivors.reserve(keys.size() - 1);
            survivors.insert(survivors.end(), keys.begin(), keys.begin() + static_cast<std::ptrdiff_t>(rank));
            survivors.insert(survivors.end(), keys.begin() + static_cast<std::ptrdiff_t>(rank + 1), keys.end());
            shi::SHIIndex live = build_index(keys, modulus, false);
            live.erase(keys[rank]);
            shi::SHIIndex fresh = build_index(survivors, modulus, false);
            if (live.serialize_bytes() == fresh.serialize_bytes()) ++match;
        }
        std::cout << "UPPER_OFF_MATCH " << match << " of " << checks << std::endl;
    }
}

void stream_deletes(std::vector<uint64_t> keys, size_t n_del, bool zero, bool upper_anchors, bool audit) {
    shi::SHIIndex idx = build_index(keys, 4096, upper_anchors);
    idx.set_compact_limit(32);
    idx.set_zero_tail(zero);
    if (n_del > keys.size() - 1) n_del = keys.size() - 1;
    std::mt19937_64 rng(20261022);
    size_t tails = 0;
    size_t uncleared = 0;
    double us = 0;
    auto wall0 = std::chrono::steady_clock::now();
    size_t done = 0;
    while (done < n_del && keys.size() > 1) {
        std::uniform_int_distribution<size_t> pick(0, keys.size() - 1);
        size_t r = pick(rng);
        uint64_t key = keys[r];
        bool tail = idx.key_is_piece_tail(key);
        auto t0 = std::chrono::steady_clock::now();
        idx.erase(key);
        auto t1 = std::chrono::steady_clock::now();
        if (tail) ++tails;
        if (tail && idx.last_edit().vacated_after != 0) ++uncleared;
        us += std::chrono::duration<double, std::micro>(t1 - t0).count();
        keys[r] = keys.back();
        keys.pop_back();
        ++done;
        if (done % 10000 == 0) {
            auto wall = std::chrono::steady_clock::now();
            double sec = std::chrono::duration<double>(wall - wall0).count();
            std::cout << "STREAM done " << done
                      << " mean_us " << (us / static_cast<double>(done))
                      << " tails " << tails
                      << " uncleared " << uncleared
                      << " boundaries " << idx.boundary_count()
                      << " wall_s " << sec
                      << std::endl;
        }
    }
    std::sort(keys.begin(), keys.end());
    shi::SHIIndex fresh = build_index(keys, 4096, upper_anchors);
    int match = idx.serialize_bytes() == fresh.serialize_bytes();
    std::mt19937_64 qrng(20261023);
    std::uniform_int_distribution<size_t> qpick(0, keys.size() - 1);
    size_t queries = std::min<size_t>(1000000, keys.size());
    size_t miss = 0;
    auto l0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < queries; ++i) {
        if (!idx.contains(keys[qpick(qrng)])) ++miss;
    }
    auto l1 = std::chrono::steady_clock::now();
    double lookup_ns = std::chrono::duration<double, std::nano>(l1 - l0).count() / static_cast<double>(queries);
    size_t audit_miss = 0;
    if (audit) {
        for (uint64_t k : keys) {
            if (!idx.contains(k)) ++audit_miss;
        }
    }
    std::cout << "STREAM_DONE n " << done
              << " mean_us " << (done ? us / static_cast<double>(done) : 0)
              << " tails " << tails
              << " uncleared " << uncleared
              << " boundaries " << idx.boundary_count()
              << " slides " << idx.compaction_count()
              << " byte_match " << match
              << " lookup_ns " << lookup_ns
              << " lookup_miss " << miss
              << " audit_miss " << audit_miss
              << " audited " << (audit ? keys.size() : 0)
              << std::endl;
}

} // namespace

int main(int argc, char **argv) {
    std::string path;
    size_t max_keys = 0;
    bool uint32_keys = false;
    bool levels = false;
    bool zero = false;
    bool upper_anchors = true;
    bool drop_half = false;
    bool audit = false;
    bool sweep = false;
    size_t sample = 0;
    size_t stream = 0;
    size_t checks = 0;
    size_t repeats = 1;
    uint32_t modulus = 4096;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--file" && i + 1 < argc) path = argv[++i];
        else if (a == "--n" && i + 1 < argc) max_keys = std::stoull(argv[++i]);
        else if (a == "--u32") uint32_keys = true;
        else if (a == "--levels") levels = true;
        else if (a == "--zero") zero = true;
        else if (a == "--no-upper-anchors") upper_anchors = false;
        else if (a == "--drop-half") drop_half = true;
        else if (a == "--audit") audit = true;
        else if (a == "--sweep") sweep = true;
        else if (a == "--sample" && i + 1 < argc) sample = std::stoull(argv[++i]);
        else if (a == "--checks" && i + 1 < argc) checks = std::stoull(argv[++i]);
        else if (a == "--stream" && i + 1 < argc) stream = std::stoull(argv[++i]);
        else if (a == "--repeats" && i + 1 < argc) repeats = std::stoull(argv[++i]);
        else if (a == "--modulus" && i + 1 < argc) modulus = static_cast<uint32_t>(std::stoul(argv[++i]));
        else {
            std::cerr << "usage: bench_probe --file PATH [--u32] [--n N] [--levels] [--sample K] [--checks K] [--zero] [--stream N] [--sweep] [--audit] [--no-upper-anchors] [--drop-half] [--repeats R] [--modulus M]\n";
            return 2;
        }
    }
    if (path.empty()) return 2;
    std::vector<uint64_t> keys = load_file(path, max_keys, uint32_keys);
    std::cout << "keys " << keys.size() << std::endl;
    if (levels) print_levels(keys, drop_half);
    if (sweep) {
        for (uint32_t m : {0u, 1024u, 4096u, 16384u}) sample_deletes(keys, sample == 0 ? 32 : sample, zero, m, upper_anchors);
    } else if (sample > 0) {
        for (size_t r = 0; r < repeats; ++r) {
            std::cout << "REPEAT " << r << std::endl;
            sample_deletes(keys, sample, zero, modulus, upper_anchors);
        }
    }
    if (checks > 0) identity_checks(keys, checks, modulus, upper_anchors, drop_half, zero);
    if (stream > 0) stream_deletes(std::move(keys), stream, zero, upper_anchors, audit);
    return 0;
}

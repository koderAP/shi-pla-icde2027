#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "pgm/pgm_index_dynamic.hpp"
#include "shi/datasets.hpp"
#include "shi/shi_pla.hpp"

namespace {

std::vector<uint64_t> load_file(const std::string &path, size_t max_keys, bool uint32_keys) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }
    uint64_t count = 0;
    in.read(reinterpret_cast<char *>(&count), sizeof(count));
    if (!in) {
        throw std::runtime_error("short header in " + path);
    }
    size_t n = static_cast<size_t>(count);
    if (max_keys > 0 && max_keys < n) {
        n = max_keys;
    }
    std::vector<uint64_t> keys(n);
    if (!uint32_keys) {
        in.read(reinterpret_cast<char *>(keys.data()), static_cast<std::streamsize>(n * sizeof(uint64_t)));
    } else {
        std::vector<uint32_t> raw(n);
        in.read(reinterpret_cast<char *>(raw.data()), static_cast<std::streamsize>(n * sizeof(uint32_t)));
        for (size_t i = 0; i < n; ++i) {
            keys[i] = raw[i];
        }
    }
    if (!in) {
        throw std::runtime_error("short payload in " + path);
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    return keys;
}

size_t audit_lookups(const shi::SHIIndex &idx, const std::vector<uint64_t> &keys) {
    size_t misses = 0;
    for (size_t rank = 0; rank < keys.size(); ++rank) {
        if (idx.lookup(keys[rank]) != rank) {
            ++misses;
        }
    }
    return misses;
}

} // namespace

int main(int argc, char **argv) {
    std::string path;
    std::string generate;
    size_t max_keys = 0;
    size_t n_deletes = 30;
    bool uint32_keys = false;
    bool do_audit = true;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char *flag) {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + flag);
            }
            return std::string(argv[++i]);
        };
        if (a == "--file") path = need("--file");
        else if (a == "--generate") generate = need("--generate");
        else if (a == "--n") max_keys = static_cast<size_t>(std::stoull(need("--n")));
        else if (a == "--deletes") n_deletes = static_cast<size_t>(std::stoull(need("--deletes")));
        else if (a == "--u32") uint32_keys = true;
        else if (a == "--no-audit") do_audit = false;
        else throw std::runtime_error("unknown arg " + a);
    }

    std::vector<uint64_t> keys;
    if (!path.empty()) {
        keys = load_file(path, max_keys, uint32_keys);
    } else if (!generate.empty()) {
        size_t n = max_keys == 0 ? 200000 : max_keys;
        keys = shi::generate_dataset(generate, n, 20261022ULL);
    } else {
        std::cerr << "usage: bench_e2e --file PATH [--n N] [--u32] [--deletes K]\n"
                  << "       bench_e2e --generate NAME --n N [--deletes K]\n";
        return 2;
    }

    const size_t n = keys.size();
    std::cout << "keys " << n << " unique_span " << (n ? keys.back() - keys.front() : 0) << std::endl;

    auto t_build0 = std::chrono::steady_clock::now();
    shi::SHIIndex index(keys, 16, 0x5A175EEDULL, 4096, shi::PLAMode::OptimalHull, 4);
    auto t_build1 = std::chrono::steady_clock::now();
    double build_ms = std::chrono::duration<double, std::milli>(t_build1 - t_build0).count();

    pgm::PGMIndex<uint64_t, 16> pgm(keys);
    auto t_pgm0 = std::chrono::steady_clock::now();
    pgm::PGMIndex<uint64_t, 16> pgm_timed(keys);
    auto t_pgm1 = std::chrono::steady_clock::now();
    double pgm_build_ms = std::chrono::duration<double, std::milli>(t_pgm1 - t_pgm0).count();

    std::cout << "built shi_ms " << build_ms << " segments " << index.segment_count() << std::endl;

    size_t lookup_misses = 0;
    if (do_audit) {
        lookup_misses = audit_lookups(index, keys);
        std::cout << "audit_done misses " << lookup_misses << std::endl;
    }

    std::mt19937_64 rng(20261022);
    std::uniform_int_distribution<size_t> pick(n / 20, n - n / 20 - 1);
    std::vector<size_t> positions;
    positions.reserve(n_deletes);
    while (positions.size() < n_deletes && positions.size() < n / 2) {
        size_t p = pick(rng);
        if (std::find(positions.begin(), positions.end(), p) == positions.end()) {
            positions.push_back(p);
        }
    }

    size_t identical = 0;
    double e2e_us = 0;
    double model_us = 0;
    double rebuild_us = 0;
    double refit_sum = 0;
    size_t refit_max = 0;
    for (size_t pos : positions) {
        uint64_t del = keys[pos];
        std::vector<uint64_t> survivors;
        survivors.reserve(n - 1);
        survivors.insert(survivors.end(), keys.begin(), keys.begin() + pos);
        survivors.insert(survivors.end(), keys.begin() + pos + 1, keys.end());

        shi::SHIIndex live(keys, 16, 0x5A175EEDULL, 4096, shi::PLAMode::OptimalHull, 4);
        auto e0 = std::chrono::steady_clock::now();
        auto st = live.erase(del);
        auto e1 = std::chrono::steady_clock::now();
        e2e_us += std::chrono::duration<double, std::micro>(e1 - e0).count();
        refit_sum += static_cast<double>(st.segments_refit);
        refit_max = std::max(refit_max, st.segments_refit);

        shi::SHIIndex model_only(keys, 16, 0x5A175EEDULL, 4096, shi::PLAMode::OptimalHull, 4);
        model_only.set_keys_raw(survivors);
        auto m0 = std::chrono::steady_clock::now();
        model_only.refit_only(del);
        auto m1 = std::chrono::steady_clock::now();
        model_us += std::chrono::duration<double, std::micro>(m1 - m0).count();

        auto r0 = std::chrono::steady_clock::now();
        shi::SHIIndex fresh(survivors, 16, 0x5A175EEDULL, 4096, shi::PLAMode::OptimalHull, 4);
        auto r1 = std::chrono::steady_clock::now();
        rebuild_us += std::chrono::duration<double, std::micro>(r1 - r0).count();
        if (live.serialize_bytes() == fresh.serialize_bytes()) {
            ++identical;
        }
    }
    double denom = static_cast<double>(std::max<size_t>(1, positions.size()));

    size_t pgm_probes = std::min<size_t>(positions.size(), 20);
    std::vector<std::pair<uint64_t, uint64_t>> pairs(n);
    for (size_t i = 0; i < n; ++i) pairs[i] = {keys[i], i + 1};
    pgm::DynamicPGMIndex<uint64_t, uint64_t> dyn(pairs.begin(), pairs.end());
    size_t before = dyn.size_in_bytes();
    size_t tombstones = 0;
    for (size_t i = 0; i < pgm_probes; ++i) {
        uint64_t del = keys[positions[i]];
        dyn.erase(del);
        bool found = false;
        for (const auto &level : dyn.debug_levels()) {
            for (const auto &item : level) {
                if (item.first == del && item.deleted()) {
                    found = true;
                    break;
                }
            }
            if (found) break;
        }
        if (found) ++tombstones;
    }
    size_t after = dyn.size_in_bytes();
    double bytes_per_delete = pgm_probes == 0 ? 0.0 : static_cast<double>(after - before) / static_cast<double>(pgm_probes);

    std::cout << "shi_segments " << index.segments().size()
              << " shi_levels " << index.level_count()
              << " shi_model_bytes " << index.model_bytes()
              << " pgm_segments " << pgm.segments_count()
              << " pgm_model_bytes " << pgm.size_in_bytes()
              << "\n";
    std::cout << "shi_build_ms " << build_ms
              << " pgm_build_ms " << pgm_build_ms
              << " lookup_misses " << lookup_misses
              << " lookup_audited " << (do_audit ? n : 0)
              << "\n";
    std::cout << "deletes " << positions.size()
              << " byte_identical " << identical
              << " refit_mean " << (refit_sum / denom)
              << " refit_max " << refit_max
              << " e2e_delete_us " << (e2e_us / denom)
              << " model_only_us " << (model_us / denom)
              << " rebuild_us " << (rebuild_us / denom)
              << " e2e_speedup " << ((e2e_us > 0) ? (rebuild_us / e2e_us) : 0)
              << "\n";
    std::cout << "pgm_dynamic_probes " << pgm_probes
              << " pgm_tombstones_found " << tombstones
              << " pgm_bytes_per_delete " << bytes_per_delete
              << "\n";
    return lookup_misses == 0 && identical == positions.size() ? 0 : 1;
}

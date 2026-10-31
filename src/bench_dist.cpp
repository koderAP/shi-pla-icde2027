#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

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

shi::SHIIndex build_index(const std::vector<uint64_t> &keys) {
    return shi::SHIIndex(keys, 16, 0x5A175EEDULL, 4096, shi::PLAMode::OptimalHull, 4);
}

struct Sample {
    size_t rank;
    double e2e_us;
    size_t refit;
};

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    double idx = p * static_cast<double>(v.size() - 1);
    size_t lo = static_cast<size_t>(idx);
    size_t hi = std::min(lo + 1, v.size() - 1);
    double w = idx - static_cast<double>(lo);
    return v[lo] * (1.0 - w) + v[hi] * w;
}

void report_prefix(const std::string &tag, const std::vector<Sample> &all, size_t k) {
    k = std::min(k, all.size());
    if (k == 0) return;
    std::vector<double> us;
    us.reserve(k);
    double sum = 0;
    double refit = 0;
    size_t refit_max = 0;
    for (size_t i = 0; i < k; ++i) {
        us.push_back(all[i].e2e_us);
        sum += all[i].e2e_us;
        refit += static_cast<double>(all[i].refit);
        refit_max = std::max(refit_max, all[i].refit);
    }
    double mean = sum / static_cast<double>(k);
    double var = 0;
    for (double x : us) {
        double d = x - mean;
        var += d * d;
    }
    var /= static_cast<double>(k);
    std::cout << "STATS " << tag
              << " n " << k
              << " mean_us " << mean
              << " stdev_us " << std::sqrt(var)
              << " min_us " << *std::min_element(us.begin(), us.end())
              << " p50_us " << percentile(us, 0.50)
              << " p90_us " << percentile(us, 0.90)
              << " p99_us " << percentile(us, 0.99)
              << " max_us " << *std::max_element(us.begin(), us.end())
              << " refit_mean " << (refit / static_cast<double>(k))
              << " refit_max " << refit_max
              << std::endl;
}

Sample time_erase(const shi::SHIIndex &base, uint64_t key, size_t rank) {
    shi::SHIIndex live = base;
    live.set_zero_tail(true);
    auto t0 = std::chrono::steady_clock::now();
    auto st = live.erase(key);
    auto t1 = std::chrono::steady_clock::now();
    Sample s;
    s.rank = rank;
    s.e2e_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    s.refit = st.segments_refit;
    if (s.e2e_us > 20000) {
        std::cout << "SLOW rank " << rank
                  << " us " << s.e2e_us
                  << " refit " << st.segments_refit
                  << " scanned " << st.keys_scanned
                  << " piece " << base.piece_len(rank)
                  << std::endl;
    }
    return s;
}

bool check_identical(const std::vector<uint64_t> &keys, size_t rank, double &rebuild_us) {
    std::vector<uint64_t> survivors;
    survivors.reserve(keys.size() - 1);
    survivors.insert(survivors.end(), keys.begin(), keys.begin() + static_cast<std::ptrdiff_t>(rank));
    survivors.insert(survivors.end(), keys.begin() + static_cast<std::ptrdiff_t>(rank + 1), keys.end());
    shi::SHIIndex live = build_index(keys);
    live.erase(keys[rank]);
    auto r0 = std::chrono::steady_clock::now();
    shi::SHIIndex fresh = build_index(survivors);
    auto r1 = std::chrono::steady_clock::now();
    rebuild_us = std::chrono::duration<double, std::micro>(r1 - r0).count();
    return live.serialize_bytes() == fresh.serialize_bytes();
}

} // namespace

int main(int argc, char **argv) {
    std::string path;
    size_t max_keys = 0;
    size_t n_random = 512;
    size_t n_checks = 0;
    size_t n_batch = 0;
    bool uint32_keys = false;
    bool do_identity = true;
    bool do_timer = true;
    std::vector<size_t> only_ranks;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char *flag) {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing ") + flag);
            return std::string(argv[++i]);
        };
        if (a == "--file") path = need("--file");
        else if (a == "--n") max_keys = static_cast<size_t>(std::stoull(need("--n")));
        else if (a == "--deletes") n_random = static_cast<size_t>(std::stoull(need("--deletes")));
        else if (a == "--u32") uint32_keys = true;
        else if (a == "--no-identity") do_identity = false;
        else if (a == "--no-timer") do_timer = false;
        else if (a == "--checks") n_checks = static_cast<size_t>(std::stoull(need("--checks")));
        else if (a == "--batch") n_batch = static_cast<size_t>(std::stoull(need("--batch")));
        else if (a == "--ranks") {
            std::string spec = need("--ranks");
            size_t pos = 0;
            while (pos < spec.size()) {
                size_t comma = spec.find(',', pos);
                std::string tok = spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
                if (!tok.empty()) only_ranks.push_back(static_cast<size_t>(std::stoull(tok)));
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
        }
        else throw std::runtime_error("unknown arg " + a);
    }
    if (path.empty()) {
        std::cerr << "usage: bench_dist --file PATH [--n N] [--u32] [--deletes K] [--no-identity]\n";
        return 2;
    }

    auto keys = load_file(path, max_keys, uint32_keys);
    const size_t n = keys.size();
    std::cout << "keys " << n << std::endl;
    if (n < 3) return 1;

    auto b0 = std::chrono::steady_clock::now();
    shi::SHIIndex base = build_index(keys);
    auto b1 = std::chrono::steady_clock::now();
    std::cout << "built_ms " << std::chrono::duration<double, std::milli>(b1 - b0).count()
              << " segments " << base.segment_count()
              << " max_piece " << base.max_piece_keys()
              << std::endl;
    if (!only_ranks.empty()) {
        for (size_t rank : only_ranks) {
            shi::SHIIndex live = base;
            auto t0 = std::chrono::steady_clock::now();
            auto st = live.erase(keys[rank]);
            auto t1 = std::chrono::steady_clock::now();
            const auto &ed = live.last_edit();
            std::cout << "TRACE rank " << rank
                      << " us " << std::chrono::duration<double, std::micro>(t1 - t0).count()
                      << " refit " << st.segments_refit
                      << " scanned " << st.keys_scanned
                      << " fit_us " << (st.fit_ns / 1e3)
                      << " upper_us " << (st.upper_ns / 1e3)
                      << " refit_us " << (live.last_refit_ns() / 1e3)
                      << " piece " << ed.piece_keys
                      << " prev " << ed.prev_keys
                      << " copied " << ed.copied_keys
                      << " merged " << ed.merged
                      << " move_us " << (ed.move_ns / 1e3)
                      << " merge_us " << (ed.merge_ns / 1e3)
                      << " prefix_us " << (ed.prefix_ns / 1e3)
                      << std::endl;
        }
        return 0;
    }

    if (!do_timer) {
        n_random = 0;
    }

    std::vector<size_t> quantiles;
    if (!do_timer) {
        // Identity and batch only.
    } else
    for (double f : {0.0, 0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0}) {
        size_t rank = static_cast<size_t>(f * static_cast<double>(n - 1));
        if (quantiles.empty() || quantiles.back() != rank) quantiles.push_back(rank);
    }

    std::vector<Sample> qsamples;
    for (size_t rank : quantiles) {
        Sample s = time_erase(base, keys[rank], rank);
        qsamples.push_back(s);
        std::cout << "QUANTILE rank " << rank
                  << " frac " << (static_cast<double>(rank) / static_cast<double>(n - 1))
                  << " e2e_us " << s.e2e_us
                  << " refit " << s.refit
                  << std::endl;
    }

    std::mt19937_64 rng(20261022);
    std::uniform_int_distribution<size_t> pick(0, n - 1);
    std::vector<size_t> ranks;
    ranks.reserve(n_random);
    while (ranks.size() < n_random) {
        size_t r = pick(rng);
        if (std::find(ranks.begin(), ranks.end(), r) == ranks.end()) ranks.push_back(r);
    }

    std::vector<Sample> random_samples;
    random_samples.reserve(n_random);
    for (size_t i = 0; i < ranks.size(); ++i) {
        size_t rank = ranks[i];
        Sample s = time_erase(base, keys[rank], rank);
        random_samples.push_back(s);
        if (i + 1 == 8 || i + 1 == 32 || i + 1 == 128 || i + 1 == ranks.size() || (i + 1) % 64 == 0) {
            std::cout << "PROGRESS random " << (i + 1)
                      << " last_rank " << rank
                      << " last_us " << s.e2e_us
                      << " last_refit " << s.refit
                      << std::endl;
        }
    }

    for (size_t k : {size_t{8}, size_t{32}, size_t{128}, size_t{512}}) {
        if (k <= random_samples.size()) report_prefix("random", random_samples, k);
    }
    if (!random_samples.empty() &&
        random_samples.size() != 8 && random_samples.size() != 32 &&
        random_samples.size() != 128 && random_samples.size() != 512) {
        report_prefix("random", random_samples, random_samples.size());
    }

    std::vector<Sample> inner;
    for (const Sample &s : random_samples) {
        double frac = static_cast<double>(s.rank) / static_cast<double>(n - 1);
        if (frac >= 0.05 && frac <= 0.95) inner.push_back(s);
    }
    if (!inner.empty()) report_prefix("inner90", inner, inner.size());

    if (n_checks > 0) {
        std::mt19937_64 crng(20261023);
        std::uniform_int_distribution<size_t> cpick(0, n - 1);
        std::vector<size_t> check_ranks;
        while (check_ranks.size() < n_checks) {
            size_t r = cpick(crng);
            if (std::find(check_ranks.begin(), check_ranks.end(), r) == check_ranks.end()) {
                check_ranks.push_back(r);
            }
        }
        size_t ok = 0;
        size_t refit_max = 0;
        double refit_sum = 0;
        for (size_t rank : check_ranks) {
            std::vector<uint64_t> survivors;
            survivors.reserve(n - 1);
            survivors.insert(survivors.end(), keys.begin(), keys.begin() + static_cast<std::ptrdiff_t>(rank));
            survivors.insert(survivors.end(), keys.begin() + static_cast<std::ptrdiff_t>(rank + 1), keys.end());
            shi::SHIIndex live = base;
            auto st = live.erase(keys[rank]);
            shi::SHIIndex fresh = build_index(survivors);
            bool match = live.serialize_bytes() == fresh.serialize_bytes();
            if (match) ++ok;
            refit_sum += static_cast<double>(st.segments_refit);
            refit_max = std::max(refit_max, st.segments_refit);
        }
        std::cout << "CHECKS n " << check_ranks.size()
                  << " matched " << ok
                  << " refit_mean " << (refit_sum / static_cast<double>(check_ranks.size()))
                  << " refit_max " << refit_max
                  << std::endl;
    }

    if (n_batch > 0) {
        std::mt19937_64 brng(20261024);
        std::uniform_int_distribution<size_t> bpick(0, n - 1);
        std::vector<size_t> batch_ranks;
        while (batch_ranks.size() < n_batch) {
            size_t r = bpick(brng);
            if (std::find(batch_ranks.begin(), batch_ranks.end(), r) == batch_ranks.end()) {
                batch_ranks.push_back(r);
            }
        }
        shi::SHIIndex live = std::move(base);
        auto t0 = std::chrono::steady_clock::now();
        size_t refit_sum = 0;
        for (size_t rank : batch_ranks) refit_sum += live.erase(keys[rank]).segments_refit;
        auto t1 = std::chrono::steady_clock::now();
        double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        std::vector<char> drop(n, 0);
        for (size_t rank : batch_ranks) drop[rank] = 1;
        std::vector<uint64_t> survivors;
        survivors.reserve(n - batch_ranks.size());
        for (size_t i = 0; i < n; ++i) {
            if (!drop[i]) survivors.push_back(keys[i]);
        }
        shi::SHIIndex fresh = build_index(survivors);
        bool match = live.serialize_bytes() == fresh.serialize_bytes();
        std::cout << "BATCH n " << batch_ranks.size()
                  << " total_us " << total_us
                  << " per_us " << (total_us / static_cast<double>(batch_ranks.size()))
                  << " refit_mean " << (static_cast<double>(refit_sum) / static_cast<double>(batch_ranks.size()))
                  << " live " << live.size()
                  << " match " << (match ? 1 : 0)
                  << std::endl;
    }

    if (do_identity && !quantiles.empty()) {
        size_t ok = 0;
        double rebuild_sum = 0;
        for (size_t rank : quantiles) {
            double rebuild_us = 0;
            bool match = check_identical(keys, rank, rebuild_us);
            if (match) ++ok;
            rebuild_sum += rebuild_us;
            std::cout << "IDENTICAL rank " << rank
                      << " match " << (match ? 1 : 0)
                      << " rebuild_us " << rebuild_us
                      << std::endl;
        }
        std::cout << "IDENTICAL_SUMMARY n " << quantiles.size()
                  << " matched " << ok
                  << " rebuild_mean_us " << (rebuild_sum / static_cast<double>(quantiles.size()))
                  << std::endl;
    }

    std::cout << "DIST_EXIT 0" << std::endl;
    return 0;
}

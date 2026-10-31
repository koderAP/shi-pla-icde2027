#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace shi {

inline bool file_exists(const std::string &path) {
    struct stat st{};
    return stat(path.c_str(), &st) == 0 && st.st_size > 8;
}

inline std::vector<uint64_t> read_sosd_binary(const std::string &path, size_t max_keys = 0) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open SOSD binary file: " + path);
    }
    uint64_t count = 0;
    in.read(reinterpret_cast<char *>(&count), sizeof(uint64_t));
    size_t n = static_cast<size_t>(count);
    if (max_keys > 0 && max_keys < n) {
        n = max_keys;
    }
    std::vector<uint64_t> keys(n);
    in.read(reinterpret_cast<char *>(keys.data()), static_cast<std::streamsize>(n * sizeof(uint64_t)));
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    return keys;
}

inline void write_sosd_binary(const std::string &path, const std::vector<uint64_t> &keys) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("Cannot write SOSD binary file: " + path);
    }
    uint64_t count = static_cast<uint64_t>(keys.size());
    out.write(reinterpret_cast<const char *>(&count), sizeof(uint64_t));
    out.write(
        reinterpret_cast<const char *>(keys.data()),
        static_cast<std::streamsize>(keys.size() * sizeof(uint64_t))
    );
}

inline std::vector<uint64_t> generate_dataset(
    const std::string &name,
    size_t n,
    uint64_t seed = 20261022ULL
) {
    std::mt19937_64 rng(seed);
    std::vector<uint64_t> out;
    out.reserve(n);

    if (name == "convex") {
        for (size_t i = 1; i <= n; ++i) {
            out.push_back(static_cast<uint64_t>(i) * static_cast<uint64_t>(i));
        }
        return out;
    }

    if (name == "uniform") {
        std::uniform_int_distribution<uint64_t> dist(1, 1000000000000000ULL);
        out.resize(n + n / 10 + 64);
        for (size_t i = 0; i < out.size(); ++i) out[i] = dist(rng);
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        out.resize(n);
        return out;
    }

    if (name == "lognormal") {
        std::lognormal_distribution<double> dist(0.0, 1.0);
        uint64_t cur = 1000;
        for (size_t i = 0; i < n; ++i) {
            double step = std::min(1e7, std::max(1.0, std::round(dist(rng) * 500.0)));
            cur += static_cast<uint64_t>(step);
            out.push_back(cur);
        }
        return out;
    }

    if (name == "clustered") {
        const size_t clusters = 40;
        std::uniform_int_distribution<uint64_t> center_dist(100000000ULL, 100000000000000ULL);
        std::normal_distribution<double> noise(0.0, 250000.0);
        std::vector<uint64_t> centers(clusters);
        for (size_t c = 0; c < clusters; ++c) centers[c] = center_dist(rng);
        size_t per = (n / clusters) + 256;
        for (uint64_t center : centers) {
            for (size_t j = 0; j < per; ++j) {
                int64_t v = static_cast<int64_t>(center) + static_cast<int64_t>(std::llround(noise(rng)));
                if (v > 0) out.push_back(static_cast<uint64_t>(v));
            }
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        while (out.size() < n) {
            out.push_back(out.back() + 1 + (rng() % 17));
        }
        out.resize(n);
        return out;
    }

    if (name == "books") {
        // SOSD books: smooth CDF with local popularity curvature (Gamma/Weibull steps)
        std::gamma_distribution<double> gamma(1.6, 400.0);
        uint64_t cur = 100000ULL;
        for (size_t i = 0; i < n; ++i) {
            double envelope = 1.0 + 0.6 * std::sin(static_cast<double>(i) / 18000.0);
            uint64_t step = std::max<uint64_t>(1, static_cast<uint64_t>(std::llround(gamma(rng) * envelope)));
            cur += step;
            out.push_back(cur);
        }
        return out;
    }

    if (name == "fb") {
        // SOSD fb: Facebook user IDs with heavy-tailed Pareto outlier gaps + dense ID blocks
        std::exponential_distribution<double> exp_dist(1.5);
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        uint64_t cur = 4ULL;
        for (size_t i = 0; i < n; ++i) {
            uint64_t step = 1;
            double p = u01(rng);
            if (p < 0.78) {
                step = 1 + (rng() % 6);
            } else if (p < 0.985) {
                step = 10 + static_cast<uint64_t>(exp_dist(rng) * 240.0);
            } else {
                // Heavy-tail allocation jump
                double pareto = std::pow(1.0 - u01(rng) * 0.999, -1.65);
                step = static_cast<uint64_t>(std::min(1e8, pareto * 2500.0));
            }
            cur += std::max<uint64_t>(1, step);
            out.push_back(cur);
        }
        return out;
    }

    if (name == "osm") {
        // SOSD osm_cellids: hierarchical S2 Hilbert cell IDs with prefix jumps across regions
        const size_t n_regions = 64;
        std::vector<uint64_t> region_prefixes(n_regions);
        for (size_t r = 0; r < n_regions; ++r) {
            region_prefixes[r] = ((r + 1) << 42) + ((rng() & 0xFFFFULL) << 24);
        }
        std::lognormal_distribution<double> cell_step(2.2, 1.25);
        size_t per_region = n / n_regions;
        for (size_t r = 0; r < n_regions; ++r) {
            uint64_t cur = region_prefixes[r];
            size_t limit = (r + 1 == n_regions) ? (n - out.size()) : per_region;
            for (size_t j = 0; j < limit; ++j) {
                uint64_t delta = std::max<uint64_t>(
                    1,
                    std::min<uint64_t>(1ULL << 20, static_cast<uint64_t>(cell_step(rng)))
                );
                cur += delta;
                out.push_back(cur);
            }
        }
        std::sort(out.begin(), out.end());
        return out;
    }

    if (name == "wiki") {
        // SOSD wiki_ts: Wikipedia edit timestamps with diurnal rates + edit bursts
        uint64_t cur = 979516800000ULL; // 2001 epoch in ms
        std::exponential_distribution<double> arrival(1.0);
        for (size_t i = 0; i < n; ++i) {
            double phase = static_cast<double>(i) / 8640.0;
            double rate = 2.5 + 1.8 * std::sin(phase) + ((i % 4096 < 96) ? 12.0 : 0.0);
            uint64_t dt = std::max<uint64_t>(1, static_cast<uint64_t>(std::llround(1000.0 * arrival(rng) / rate)));
            cur += dt;
            out.push_back(cur);
        }
        return out;
    }

    throw std::invalid_argument("Unknown dataset name: " + name);
}

inline std::vector<uint64_t> load_or_generate_dataset(
    const std::string &name,
    size_t n,
    const std::string &sosd_dir = "data/sosd",
    uint64_t seed = 20261022ULL
) {
    std::string path = sosd_dir + "/" + name + ".bin";
    if (file_exists(path)) {
        auto loaded = read_sosd_binary(path, n);
        if (loaded.size() >= n) {
            loaded.resize(n);
            return loaded;
        }
    }
    return generate_dataset(name, n, seed);
}

} // namespace shi

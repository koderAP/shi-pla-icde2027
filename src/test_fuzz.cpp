#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <random>
#include <vector>

#include "shi/datasets.hpp"
#include "shi/hi_btree.hpp"
#include "shi/shi_pla.hpp"

namespace {

void verify_exact_lookup_and_identity(
    const std::string &dataset_name,
    shi::PLAMode mode,
    uint32_t modulus
) {
    auto keys = shi::generate_dataset(dataset_name, 5000, 42);
    shi::SHIIndex idx(keys, 16, 0x5A175EEDULL, modulus, mode, 4);
    idx.set_compact_limit(4);

    for (size_t rank = 0; rank < keys.size(); ++rank) {
        size_t found = idx.lookup(keys[rank]);
        if (found != rank) {
            std::fprintf(
                stderr,
                "Lookup mismatch on %s (mode=%d, M=%u): key=%llu expected=%zu got=%zu\n",
                dataset_name.c_str(),
                static_cast<int>(mode),
                modulus,
                static_cast<unsigned long long>(keys[rank]),
                rank,
                found
            );
            std::abort();
        }
    }

    std::mt19937_64 rng(1337);
    std::vector<uint64_t> cur = keys;
    for (int step = 0; step < 50; ++step) {
        size_t pos = rng() % cur.size();
        uint64_t del_key = cur[pos];
        cur.erase(cur.begin() + pos);
        idx.erase(del_key);

        shi::SHIIndex fresh(cur, 16, 0x5A175EEDULL, modulus, mode, 4);
        if (idx.serialize_bytes() != fresh.serialize_bytes()) {
            std::fprintf(
                stderr,
                "Byte mismatch after delete on %s step=%d (mode=%d, M=%u)\n",
                dataset_name.c_str(),
                step,
                static_cast<int>(mode),
                modulus
            );
            std::abort();
        }
    }
    for (size_t rank = 0; rank < cur.size(); ++rank) {
        size_t found = idx.lookup(cur[rank]);
        if (found != rank) {
            std::fprintf(
                stderr,
                "Post-delete lookup mismatch on %s (mode=%d, M=%u): expected=%zu got=%zu\n",
                dataset_name.c_str(),
                static_cast<int>(mode),
                modulus,
                rank,
                found
            );
            std::abort();
        }
    }
}

void run_million_op_fuzzer() {
    constexpr size_t TOTAL_OPS = 1'000'000;
    constexpr size_t UNIVERSE_SIZE = 2048;
    constexpr size_t INITIAL_SIZE = 1024;
    constexpr size_t CHECK_EVERY = 250;

    auto universe = shi::generate_dataset("fb", UNIVERSE_SIZE, 20261022ULL);
    std::mt19937_64 rng(9001);

    std::vector<uint8_t> in_set(UNIVERSE_SIZE, 0);
    std::vector<uint64_t> initial_keys;
    initial_keys.reserve(INITIAL_SIZE);
    for (size_t i = 0; i < UNIVERSE_SIZE; i += 2) {
        in_set[i] = 1;
        initial_keys.push_back(universe[i]);
    }

    shi::SHIIndex opt_idx(initial_keys, 16, 0x5A175EEDULL, 64, shi::PLAMode::OptimalHull, 4);
    shi::SHIIndex greedy_idx(initial_keys, 16, 0x5A175EEDULL, 64, shi::PLAMode::GreedyCone, 4);
    opt_idx.set_compact_limit(4);
    greedy_idx.set_compact_limit(4);
    shi::HIBTree btree_idx(initial_keys, 16, 0xB7E151628AED2A6AULL);

    size_t byte_checks = 0;
    size_t mismatches = 0;
    size_t total_refit_opt = 0;
    size_t total_refit_greedy = 0;

    auto t0 = std::chrono::steady_clock::now();

    for (size_t op = 1; op <= TOTAL_OPS; ++op) {
        size_t u_idx = rng() % UNIVERSE_SIZE;
        uint64_t key = universe[u_idx];

        if (in_set[u_idx]) {
            if (opt_idx.size() <= 64) {
                continue;
            }
            in_set[u_idx] = 0;
            total_refit_opt += opt_idx.erase(key).segments_refit;
            total_refit_greedy += greedy_idx.erase(key).segments_refit;
            btree_idx.erase(key);
        } else {
            in_set[u_idx] = 1;
            total_refit_opt += opt_idx.insert(key).segments_refit;
            total_refit_greedy += greedy_idx.insert(key).segments_refit;
            btree_idx.insert(key);
        }

        if (op % CHECK_EVERY == 0) {
            std::vector<uint64_t> expected;
            expected.reserve(opt_idx.size());
            for (size_t i = 0; i < UNIVERSE_SIZE; ++i) {
                if (in_set[i]) expected.push_back(universe[i]);
            }

            shi::SHIIndex fresh_opt(expected, 16, 0x5A175EEDULL, 64, shi::PLAMode::OptimalHull, 4);
            shi::SHIIndex fresh_greedy(expected, 16, 0x5A175EEDULL, 64, shi::PLAMode::GreedyCone, 4);
            shi::HIBTree fresh_btree(expected, 16, 0xB7E151628AED2A6AULL);

            ++byte_checks;
            for (size_t i = 0; i < expected.size(); ++i) {
                if (opt_idx.lookup(expected[i]) != i) {
                    std::fprintf(stderr, "Fuzz lookup mismatch at op=%zu rank=%zu\n", op, i);
                    std::abort();
                }
            }
            if (opt_idx.serialize_bytes() != fresh_opt.serialize_bytes() ||
                greedy_idx.serialize_bytes() != fresh_greedy.serialize_bytes() ||
                btree_idx.serialize_bytes() != fresh_btree.serialize_bytes()) {
                ++mismatches;
                std::fprintf(stderr, "Fuzz mismatch at op=%zu\n", op);
                std::abort();
            }
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    std::printf(
        "[PASS] 1,000,000-op SHI byte-identity fuzzer completed in %.2fs\n"
        "       Full rebuild byte-equality checks: %zu (mismatches: %zu)\n"
        "       Avg segments re-fit/op: OptimalHull=%.2f, GreedyCone=%.2f\n",
        secs,
        byte_checks,
        mismatches,
        static_cast<double>(total_refit_opt) / TOTAL_OPS,
        static_cast<double>(total_refit_greedy) / TOTAL_OPS
    );
}

} // namespace

int main() {
    const std::vector<std::string> datasets = {
        "books", "fb", "osm", "wiki", "lognormal", "uniform", "clustered", "convex"
    };
    for (const auto &ds : datasets) {
        verify_exact_lookup_and_identity(ds, shi::PLAMode::OptimalHull, 0);
        verify_exact_lookup_and_identity(ds, shi::PLAMode::OptimalHull, 512);
        verify_exact_lookup_and_identity(ds, shi::PLAMode::GreedyCone, 0);
        verify_exact_lookup_and_identity(ds, shi::PLAMode::GreedyCone, 512);
    }
    std::printf("[PASS] All 8 datasets x {OptimalHull, GreedyCone} x {Unanchored, Anchored} verified.\n");

    run_million_op_fuzzer();
    return 0;
}

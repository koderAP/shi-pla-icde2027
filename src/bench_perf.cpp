#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "alex.h"
#include "lipp.h"
#include "pgm/pgm_index.hpp"
#include "pgm/pgm_index_dynamic.hpp"

#include "shi/datasets.hpp"
#include "shi/hi_btree.hpp"
#include "shi/shi_pla.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double percentile(std::vector<size_t> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double pos = (v.size() - 1) * q;
    size_t lo = static_cast<size_t>(pos);
    size_t hi = std::min(lo + 1, v.size() - 1);
    double frac = pos - lo;
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

struct VariantPerf {
    std::string variant_name;
    size_t segments = 0;
    double model_bytes_per_key = 0.0;
    double space_ratio_vs_pgm = 0.0;
    double lookup_ns = 0.0;
    double lookup_ratio_vs_pgm = 0.0;
    double refit_mean = 0.0;
    double refit_p99 = 0.0;
    size_t refit_max = 0;
    double local_delete_us = 0.0;
    double rebuild_us = 0.0;
    double speedup_vs_rebuild = 0.0;
    size_t byte_identical_count = 0;
    size_t total_deletes = 0;
    double abs_segments_changed_mean = 0.0;
};

struct DatasetPerfReport {
    std::string dataset;
    size_t n_keys = 0;
    VariantPerf pgm_static;
    VariantPerf pgm_dynamic;
    VariantPerf alex_dynamic;
    VariantPerf lipp_dynamic;
    VariantPerf hi_btree;
    VariantPerf shi_optimal_unanchored;
    VariantPerf shi_optimal_anchored;
    VariantPerf shi_greedy_unanchored;
    VariantPerf shi_greedy_anchored;
};

VariantPerf measure_shi_variant(
    const std::string &label,
    const std::vector<uint64_t> &keys,
    const std::vector<uint64_t> &queries,
    const std::vector<size_t> &del_positions,
    uint32_t modulus,
    shi::PLAMode mode,
    double pgm_lookup_ns,
    double pgm_bytes_per_key
) {
    VariantPerf vp;
    vp.variant_name = label;
    vp.total_deletes = del_positions.size();

    auto t_build0 = Clock::now();
    shi::SHIIndex base(keys, 16, 0x5A175EEDULL, modulus, mode, 4);
    auto t_build1 = Clock::now();
    double single_build_us = std::chrono::duration<double, std::micro>(t_build1 - t_build0).count();

    vp.segments = base.segment_count();
    vp.model_bytes_per_key = static_cast<double>(base.model_bytes()) / static_cast<double>(keys.size());
    vp.space_ratio_vs_pgm = (pgm_bytes_per_key > 0.0) ? (vp.model_bytes_per_key / pgm_bytes_per_key) : 1.0;

    // Measure lookup latency over queries
    volatile size_t sink = 0;
    auto t_q0 = Clock::now();
    for (uint64_t q : queries) {
        sink += base.lookup(q);
    }
    auto t_q1 = Clock::now();
    (void)sink;
    vp.lookup_ns = std::chrono::duration<double, std::nano>(t_q1 - t_q0).count() / queries.size();
    vp.lookup_ratio_vs_pgm = (pgm_lookup_ns > 0.0) ? (vp.lookup_ns / pgm_lookup_ns) : 1.0;

    // Measure local delete re-segmentation vs full rebuild and verify byte-identity
    std::vector<size_t> refits;
    refits.reserve(del_positions.size());
    double total_local_us = 0.0;
    double total_rebuild_us = 0.0;
    double total_abs_changed = 0.0;
    size_t identical = 0;

    for (size_t idx = 0; idx < del_positions.size(); ++idx) {
        size_t pos = del_positions[idx];
        uint64_t del_key = keys[pos];

        std::vector<uint64_t> survivors;
        survivors.reserve(keys.size() - 1);
        survivors.insert(survivors.end(), keys.begin(), keys.begin() + pos);
        survivors.insert(survivors.end(), keys.begin() + pos + 1, keys.end());

        shi::SHIIndex working = base;
        working.set_keys_raw(survivors);
        auto t0 = Clock::now();
        auto st = working.refit_only(del_key);
        auto t1 = Clock::now();
        total_local_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
        refits.push_back(st.segments_refit);

        auto t2 = Clock::now();
        shi::SHIIndex fresh(survivors, 16, 0x5A175EEDULL, modulus, mode, 4);
        auto t3 = Clock::now();
        total_rebuild_us += std::chrono::duration<double, std::micro>(t3 - t2).count();

        if (working.serialize_bytes() == fresh.serialize_bytes()) {
            ++identical;
        }
        total_abs_changed += static_cast<double>(base.absolute_segments_changed_vs(fresh));
    }

    double sum_refit = std::accumulate(refits.begin(), refits.end(), 0.0);
    vp.refit_mean = sum_refit / std::max<size_t>(1, refits.size());
    vp.refit_p99 = percentile(refits, 0.99);
    vp.refit_max = refits.empty() ? 0 : *std::max_element(refits.begin(), refits.end());
    vp.local_delete_us = total_local_us / std::max<size_t>(1, del_positions.size());
    vp.rebuild_us = (!del_positions.empty())
                        ? (total_rebuild_us / del_positions.size())
                        : single_build_us;
    vp.speedup_vs_rebuild = (vp.local_delete_us > 0.0) ? (vp.rebuild_us / vp.local_delete_us) : 0.0;
    vp.byte_identical_count = identical;
    vp.abs_segments_changed_mean = total_abs_changed / std::max<size_t>(1, del_positions.size());
    return vp;
}

DatasetPerfReport benchmark_dataset(
    const std::string &name,
    size_t n_keys,
    size_t n_queries,
    size_t n_deletes
) {
    DatasetPerfReport rep;
    rep.dataset = name;
    auto keys = shi::load_or_generate_dataset(name, n_keys, "data/sosd", 20261022ULL);
    rep.n_keys = keys.size();

    std::mt19937_64 rng(20261022ULL);
    std::vector<uint64_t> queries(n_queries);
    for (size_t i = 0; i < n_queries; ++i) {
        queries[i] = keys[rng() % keys.size()];
    }

    std::vector<size_t> del_positions(keys.size() - 4);
    for (size_t i = 2; i + 2 < keys.size(); ++i) del_positions[i - 2] = i;
    std::shuffle(del_positions.begin(), del_positions.end(), rng);
    del_positions.resize(std::min(n_deletes, del_positions.size()));

    // 1. Static PGMIndex<uint64_t, 16>
    {
        VariantPerf vp;
        vp.variant_name = "PGM-Static";
        auto tb0 = Clock::now();
        pgm::PGMIndex<uint64_t, 16> pgm_idx(keys);
        auto tb1 = Clock::now();
        vp.rebuild_us = std::chrono::duration<double, std::micro>(tb1 - tb0).count();
        vp.local_delete_us = vp.rebuild_us; // Static PGM requires full rebuild to erase cleanly
        vp.speedup_vs_rebuild = 1.0;
        vp.segments = pgm_idx.segments_count();
        vp.model_bytes_per_key = static_cast<double>(pgm_idx.size_in_bytes()) / keys.size();
        vp.space_ratio_vs_pgm = 1.0;

        volatile size_t sink = 0;
        auto tq0 = Clock::now();
        for (uint64_t q : queries) {
            auto approx = pgm_idx.search(q);
            auto it = std::lower_bound(keys.begin() + approx.lo, keys.begin() + approx.hi, q);
            sink += static_cast<size_t>(it - keys.begin());
        }
        auto tq1 = Clock::now();
        (void)sink;
        vp.lookup_ns = std::chrono::duration<double, std::nano>(tq1 - tq0).count() / queries.size();
        vp.lookup_ratio_vs_pgm = 1.0;
        rep.pgm_static = vp;
    }

    // 2. DynamicPGMIndex (tombstone LSM)
    {
        VariantPerf vp;
        vp.variant_name = "PGM-Dynamic";
        std::vector<std::pair<uint64_t, uint64_t>> pairs(keys.size());
        for (size_t i = 0; i < keys.size(); ++i) pairs[i] = {keys[i], i};
        pgm::DynamicPGMIndex<uint64_t, uint64_t, pgm::PGMIndex<uint64_t, 16>> dyn(pairs.begin(), pairs.end());

        vp.model_bytes_per_key = static_cast<double>(dyn.index_size_in_bytes()) / keys.size();
        vp.space_ratio_vs_pgm = vp.model_bytes_per_key / rep.pgm_static.model_bytes_per_key;

        volatile size_t sink = 0;
        auto tq0 = Clock::now();
        for (uint64_t q : queries) {
            auto it = dyn.find(q);
            if (it != dyn.end()) sink += it->second;
        }
        auto tq1 = Clock::now();
        (void)sink;
        vp.lookup_ns = std::chrono::duration<double, std::nano>(tq1 - tq0).count() / queries.size();
        vp.lookup_ratio_vs_pgm = vp.lookup_ns / rep.pgm_static.lookup_ns;

        auto td0 = Clock::now();
        for (size_t pos : del_positions) {
            dyn.erase(keys[pos]);
        }
        auto td1 = Clock::now();
        vp.local_delete_us = std::chrono::duration<double, std::micro>(td1 - td0).count() / del_positions.size();
        vp.rebuild_us = rep.pgm_static.rebuild_us;
        vp.speedup_vs_rebuild = vp.rebuild_us / std::max(1e-6, vp.local_delete_us);
        vp.total_deletes = del_positions.size();
        vp.byte_identical_count = 0; // Retains tombstones
        rep.pgm_dynamic = vp;
    }

    // 3. ALEX Dynamic
    {
        VariantPerf vp;
        vp.variant_name = "ALEX";
        std::vector<std::pair<uint64_t, uint64_t>> bulk(keys.size());
        for (size_t i = 0; i < keys.size(); ++i) bulk[i] = {keys[i], i};
        alex::Alex<uint64_t, uint64_t> alex_idx;
        auto tb0 = Clock::now();
        alex_idx.bulk_load(bulk.data(), static_cast<int>(bulk.size()));
        auto tb1 = Clock::now();
        vp.rebuild_us = std::chrono::duration<double, std::micro>(tb1 - tb0).count();

        vp.segments = alex_idx.stats_.num_data_nodes;
        vp.model_bytes_per_key = static_cast<double>(alex_idx.model_size() + alex_idx.data_size()) / keys.size();
        vp.space_ratio_vs_pgm = vp.model_bytes_per_key / rep.pgm_static.model_bytes_per_key;

        volatile size_t sink = 0;
        auto tq0 = Clock::now();
        for (uint64_t q : queries) {
            auto it = alex_idx.find(q);
            if (!it.is_end()) sink += it.payload();
        }
        auto tq1 = Clock::now();
        (void)sink;
        vp.lookup_ns = std::chrono::duration<double, std::nano>(tq1 - tq0).count() / queries.size();
        vp.lookup_ratio_vs_pgm = vp.lookup_ns / rep.pgm_static.lookup_ns;

        auto td0 = Clock::now();
        for (size_t pos : del_positions) {
            alex_idx.erase(keys[pos]);
        }
        auto td1 = Clock::now();
        vp.local_delete_us = std::chrono::duration<double, std::micro>(td1 - td0).count() / del_positions.size();
        vp.speedup_vs_rebuild = vp.rebuild_us / std::max(1e-6, vp.local_delete_us);
        vp.total_deletes = del_positions.size();
        vp.byte_identical_count = 0; // Leaves gap anomalies & payloads
        rep.alex_dynamic = vp;
    }

    // 4. LIPP Dynamic
    {
        VariantPerf vp;
        vp.variant_name = "LIPP";
        using LippType = LIPP<uint64_t, uint64_t>;
        std::vector<LippType::V> bulk(keys.size());
        for (size_t i = 0; i < keys.size(); ++i) bulk[i] = {keys[i], i};
        LippType lipp_idx(0.0, true, 64);
        auto tb0 = Clock::now();
        lipp_idx.bulk_load(bulk.data(), static_cast<int>(bulk.size()));
        auto tb1 = Clock::now();
        vp.rebuild_us = std::chrono::duration<double, std::micro>(tb1 - tb0).count();
        vp.model_bytes_per_key = static_cast<double>(lipp_idx.index_size(true, false)) / keys.size();
        vp.space_ratio_vs_pgm = vp.model_bytes_per_key / rep.pgm_static.model_bytes_per_key;

        volatile size_t sink = 0;
        auto tq0 = Clock::now();
        for (uint64_t q : queries) {
            if (lipp_idx.exists(q)) sink += 1;
        }
        auto tq1 = Clock::now();
        (void)sink;
        vp.lookup_ns = std::chrono::duration<double, std::nano>(tq1 - tq0).count() / queries.size();
        vp.lookup_ratio_vs_pgm = vp.lookup_ns / rep.pgm_static.lookup_ns;

        auto td0 = Clock::now();
        for (size_t pos : del_positions) {
            lipp_idx.erase(keys[pos], false);
        }
        auto td1 = Clock::now();
        vp.local_delete_us = std::chrono::duration<double, std::micro>(td1 - td0).count() / del_positions.size();
        vp.speedup_vs_rebuild = vp.rebuild_us / std::max(1e-6, vp.local_delete_us);
        vp.total_deletes = del_positions.size();
        vp.byte_identical_count = 0;
        rep.lipp_dynamic = vp;
    }

    // 5. History-Independent B-Tree (Bender et al. 2016 / Canonical B-Treap)
    {
        VariantPerf vp;
        vp.variant_name = "HI-BTree";
        auto tb0 = Clock::now();
        shi::HIBTree btree(keys, 32, 0xB7E151628AED2A6AULL);
        auto tb1 = Clock::now();
        vp.rebuild_us = std::chrono::duration<double, std::micro>(tb1 - tb0).count();
        vp.segments = btree.leaf_node_count();
        vp.model_bytes_per_key = static_cast<double>(btree.model_bytes()) / keys.size();
        vp.space_ratio_vs_pgm = vp.model_bytes_per_key / rep.pgm_static.model_bytes_per_key;

        volatile size_t sink = 0;
        auto tq0 = Clock::now();
        for (uint64_t q : queries) {
            sink += btree.lookup(q);
        }
        auto tq1 = Clock::now();
        (void)sink;
        vp.lookup_ns = std::chrono::duration<double, std::nano>(tq1 - tq0).count() / queries.size();
        vp.lookup_ratio_vs_pgm = vp.lookup_ns / rep.pgm_static.lookup_ns;

        size_t identical = 0;
        double total_del_us = 0.0;
        std::vector<size_t> touched;
        for (size_t pos : del_positions) {
            uint64_t del_key = keys[pos];
            shi::HIBTree working = btree;
            auto td0 = Clock::now();
            size_t t = working.erase(del_key);
            auto td1 = Clock::now();
            total_del_us += std::chrono::duration<double, std::micro>(td1 - td0).count();
            touched.push_back(t);

            std::vector<uint64_t> survivors;
            survivors.reserve(keys.size() - 1);
            survivors.insert(survivors.end(), keys.begin(), keys.begin() + pos);
            survivors.insert(survivors.end(), keys.begin() + pos + 1, keys.end());
            shi::HIBTree fresh(survivors, 32, 0xB7E151628AED2A6AULL);
            if (working.serialize_bytes() == fresh.serialize_bytes()) {
                ++identical;
            }
        }
        vp.local_delete_us = total_del_us / del_positions.size();
        vp.speedup_vs_rebuild = vp.rebuild_us / std::max(1e-6, vp.local_delete_us);
        vp.refit_mean = std::accumulate(touched.begin(), touched.end(), 0.0) / touched.size();
        vp.refit_p99 = percentile(touched, 0.99);
        vp.refit_max = *std::max_element(touched.begin(), touched.end());
        vp.byte_identical_count = identical;
        vp.total_deletes = del_positions.size();
        rep.hi_btree = vp;
    }

    // 6. SHI-PLA variants (OptimalHull & GreedyCone, Unanchored & Anchored M=4096)
    rep.shi_optimal_unanchored = measure_shi_variant(
        "SHI-Opt-Unanchored",
        keys, queries, del_positions, 0, shi::PLAMode::OptimalHull,
        rep.pgm_static.lookup_ns, rep.pgm_static.model_bytes_per_key
    );
    rep.shi_optimal_anchored = measure_shi_variant(
        "SHI-Opt-M4096",
        keys, queries, del_positions, 4096, shi::PLAMode::OptimalHull,
        rep.pgm_static.lookup_ns, rep.pgm_static.model_bytes_per_key
    );
    rep.shi_greedy_unanchored = measure_shi_variant(
        "SHI-Greedy-Unanchored",
        keys, queries, del_positions, 0, shi::PLAMode::GreedyCone,
        rep.pgm_static.lookup_ns, rep.pgm_static.model_bytes_per_key
    );
    rep.shi_greedy_anchored = measure_shi_variant(
        "SHI-Greedy-M4096",
        keys, queries, del_positions, 4096, shi::PLAMode::GreedyCone,
        rep.pgm_static.lookup_ns, rep.pgm_static.model_bytes_per_key
    );

    return rep;
}

void write_variant_json(std::ostream &out, const VariantPerf &vp, int indent = 6) {
    std::string pad(indent, ' ');
    out << pad << "{\n"
        << pad << "  \"variant\": \"" << vp.variant_name << "\",\n"
        << pad << "  \"segments\": " << vp.segments << ",\n"
        << pad << "  \"model_bytes_per_key\": " << vp.model_bytes_per_key << ",\n"
        << pad << "  \"space_ratio_vs_pgm\": " << vp.space_ratio_vs_pgm << ",\n"
        << pad << "  \"lookup_ns\": " << vp.lookup_ns << ",\n"
        << pad << "  \"lookup_ratio_vs_pgm\": " << vp.lookup_ratio_vs_pgm << ",\n"
        << pad << "  \"refit_mean\": " << vp.refit_mean << ",\n"
        << pad << "  \"refit_p99\": " << vp.refit_p99 << ",\n"
        << pad << "  \"refit_max\": " << vp.refit_max << ",\n"
        << pad << "  \"local_delete_us\": " << vp.local_delete_us << ",\n"
        << pad << "  \"rebuild_us\": " << vp.rebuild_us << ",\n"
        << pad << "  \"speedup_vs_rebuild\": " << vp.speedup_vs_rebuild << ",\n"
        << pad << "  \"byte_identical_count\": " << vp.byte_identical_count << ",\n"
        << pad << "  \"total_deletes\": " << vp.total_deletes << ",\n"
        << pad << "  \"abs_segments_changed_mean\": " << vp.abs_segments_changed_mean << "\n"
        << pad << "}";
}

} // namespace

int main(int argc, char **argv) {
    size_t n_keys = 200000;
    size_t n_queries = 100000;
    size_t n_deletes = 200;
    std::string out_path = "results/cpp_perf.json";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--n" && i + 1 < argc) n_keys = std::stoull(argv[++i]);
        else if (arg == "--queries" && i + 1 < argc) n_queries = std::stoull(argv[++i]);
        else if (arg == "--deletes" && i + 1 < argc) n_deletes = std::stoull(argv[++i]);
        else if (arg == "--out" && i + 1 < argc) out_path = argv[++i];
    }

    const std::vector<std::string> datasets = {
        "books", "fb", "osm", "wiki", "lognormal", "uniform", "clustered", "convex"
    };

    std::printf(
        "=== C++ Performance & Overhead Benchmark (n=%zu, queries=%zu, deletes=%zu, eps=16) ===\n",
        n_keys, n_queries, n_deletes
    );
    std::printf(
        "%-10s | %-20s | %6s | %9s | %9s | %13s | %9s | %9s | %7s\n",
        "Dataset", "Variant", "Segs", "Bytes/Key", "Lookup ns", "Re-fit m/p99/M", "Delete us", "Speedup", "SHI-ID"
    );
    std::printf("--------------------------------------------------------------------------------------------------------------------\n");

    std::vector<DatasetPerfReport> reports;
    for (const auto &ds : datasets) {
        auto rep = benchmark_dataset(ds, n_keys, n_queries, n_deletes);
        reports.push_back(rep);

        const std::vector<const VariantPerf *> rows = {
            &rep.pgm_static,
            &rep.pgm_dynamic,
            &rep.alex_dynamic,
            &rep.lipp_dynamic,
            &rep.hi_btree,
            &rep.shi_optimal_unanchored,
            &rep.shi_optimal_anchored,
            &rep.shi_greedy_anchored
        };
        for (const auto *vp : rows) {
            char refit_buf[48];
            std::snprintf(refit_buf, sizeof(refit_buf), "%.1f/%.0f/%zu", vp->refit_mean, vp->refit_p99, vp->refit_max);
            std::printf(
                "%-10s | %-20s | %6zu | %6.3f (%3.1fx) | %5.1f (%3.1fx) | %13s | %9.2f | %8.1fx | %3zu/%zu\n",
                ds.c_str(),
                vp->variant_name.c_str(),
                vp->segments,
                vp->model_bytes_per_key,
                vp->space_ratio_vs_pgm,
                vp->lookup_ns,
                vp->lookup_ratio_vs_pgm,
                refit_buf,
                vp->local_delete_us,
                vp->speedup_vs_rebuild,
                vp->byte_identical_count,
                vp->total_deletes
            );
        }
        std::printf("--------------------------------------------------------------------------------------------------------------------\n");
    }

    // Anchor M sweep on SOSD books and adversarial convex
    const std::vector<uint32_t> moduli = {0, 512, 1024, 2048, 4096, 8192};
    std::vector<std::pair<std::string, std::vector<VariantPerf>>> anchor_sweeps;
    for (const std::string &sweep_ds : {"books", "fb", "convex"}) {
        auto keys = shi::load_or_generate_dataset(sweep_ds, n_keys, "data/sosd", 20261022ULL);
        std::mt19937_64 rng(20261022ULL);
        std::vector<uint64_t> queries(n_queries);
        for (size_t i = 0; i < n_queries; ++i) queries[i] = keys[rng() % keys.size()];
        std::vector<size_t> del_positions(keys.size() - 4);
        for (size_t i = 2; i + 2 < keys.size(); ++i) del_positions[i - 2] = i;
        std::shuffle(del_positions.begin(), del_positions.end(), rng);
        del_positions.resize(std::min(n_deletes, del_positions.size()));

        pgm::PGMIndex<uint64_t, 16> pgm_idx(keys);
        double pgm_bpk = static_cast<double>(pgm_idx.size_in_bytes()) / keys.size();
        volatile size_t sink = 0;
        auto tq0 = Clock::now();
        for (uint64_t q : queries) {
            auto approx = pgm_idx.search(q);
            auto it = std::lower_bound(keys.begin() + approx.lo, keys.begin() + approx.hi, q);
            sink += static_cast<size_t>(it - keys.begin());
        }
        auto tq1 = Clock::now();
        (void)sink;
        double pgm_ns = std::chrono::duration<double, std::nano>(tq1 - tq0).count() / queries.size();

        std::vector<VariantPerf> sweep_rows;
        std::printf("\n=== Anchor Modulus Sweep on %s (n=%zu) ===\n", sweep_ds.c_str(), n_keys);
        for (uint32_t m : moduli) {
            std::string label = (m == 0) ? "M=none" : ("M=" + std::to_string(m));
            auto vp = measure_shi_variant(
                label, keys, queries, del_positions, m, shi::PLAMode::OptimalHull, pgm_ns, pgm_bpk
            );
            sweep_rows.push_back(vp);
            std::printf(
                "  %-10s segs=%5zu space=%.2fx lookup=%.2fx refit=%.2f/%.0f/%zu speedup=%.1fx id=%zu/%zu\n",
                label.c_str(),
                vp.segments,
                vp.space_ratio_vs_pgm,
                vp.lookup_ratio_vs_pgm,
                vp.refit_mean,
                vp.refit_p99,
                vp.refit_max,
                vp.speedup_vs_rebuild,
                vp.byte_identical_count,
                vp.total_deletes
            );
        }
        anchor_sweeps.push_back({sweep_ds, std::move(sweep_rows)});
    }

    // Write JSON
    std::ofstream out(out_path);
    out << std::fixed << std::setprecision(4);
    out << "{\n  \"n_keys\": " << n_keys << ",\n  \"n_queries\": " << n_queries
        << ",\n  \"n_deletes\": " << n_deletes << ",\n  \"datasets\": {\n";
    for (size_t i = 0; i < reports.size(); ++i) {
        const auto &r = reports[i];
        out << "    \"" << r.dataset << "\": [\n";
        const std::vector<const VariantPerf *> all_v = {
            &r.pgm_static, &r.pgm_dynamic, &r.alex_dynamic, &r.lipp_dynamic,
            &r.hi_btree, &r.shi_optimal_unanchored, &r.shi_optimal_anchored,
            &r.shi_greedy_unanchored, &r.shi_greedy_anchored
        };
        for (size_t j = 0; j < all_v.size(); ++j) {
            write_variant_json(out, *all_v[j], 6);
            out << (j + 1 < all_v.size() ? ",\n" : "\n");
        }
        out << "    ]" << (i + 1 < reports.size() ? "," : "") << "\n";
    }
    out << "  },\n  \"anchor_sweeps\": {\n";
    for (size_t i = 0; i < anchor_sweeps.size(); ++i) {
        out << "    \"" << anchor_sweeps[i].first << "\": [\n";
        const auto &rows = anchor_sweeps[i].second;
        for (size_t j = 0; j < rows.size(); ++j) {
            write_variant_json(out, rows[j], 6);
            out << (j + 1 < rows.size() ? ",\n" : "\n");
        }
        out << "    ]" << (i + 1 < anchor_sweeps.size() ? "," : "") << "\n";
    }
    out << "  }\n}\n";
    std::printf("\nWrote performance results to %s\n", out_path.c_str());
    return 0;
}

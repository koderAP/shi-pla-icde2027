#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "pgm/pgm_index.hpp"
#include "pgm/pgm_index_dynamic.hpp"
#include "shi/hi_btree.hpp"
#include "shi/shi_pla.hpp"

#include "alex.h"

namespace {

std::vector<uint64_t> load_file(const std::string &path, size_t max_keys, bool uint32_keys) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    uint64_t count = 0;
    in.read(reinterpret_cast<char *>(&count), sizeof(count));
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
    if (!in) throw std::runtime_error("short payload in " + path);
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    return keys;
}

double ns_per(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b, size_t n) {
    return std::chrono::duration<double, std::nano>(b - a).count() / static_cast<double>(n);
}

void alex_payload(const std::vector<uint64_t> &keys, size_t probes) {
    if (keys.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        std::cout << "alex_skipped keys_exceed_int" << std::endl;
        return;
    }
    std::vector<std::pair<uint64_t, uint64_t>> bulk(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) bulk[i] = {keys[i], keys[i]};
    alex::Alex<uint64_t, uint64_t> index;
    auto b0 = std::chrono::steady_clock::now();
    index.bulk_load(bulk.data(), static_cast<int>(bulk.size()));
    auto b1 = std::chrono::steady_clock::now();
    std::mt19937_64 rng(20261022);
    std::uniform_int_distribution<size_t> pick(keys.size() / 20, keys.size() - keys.size() / 20 - 1);
    size_t payload = 0;
    size_t max_exposed = 0;
    size_t scrub_left = 0;
    size_t done = 0;
    for (size_t i = 0; i < probes; ++i) {
        uint64_t del = keys[pick(rng)];
        auto *before = index.get_leaf(del);
        bool was_max = before && before->max_key_ == del;
        index.erase(del);
        auto *leaf = index.get_leaf(del);
        if (!leaf) continue;
        ++done;
        if (was_max && leaf->max_key_ == del) ++max_exposed;
        bool held = false;
        for (int s = 0; s < leaf->data_capacity_; ++s) {
            if (!leaf->check_exists(s) && leaf->payload_slots_[s] == del) {
                held = true;
                break;
            }
        }
        if (held) ++payload;
        for (int s = 0; s < leaf->data_capacity_; ++s) {
            if (leaf->payload_slots_[s] == del) leaf->payload_slots_[s] = 0;
        }
        bool left = false;
        for (int s = 0; s < leaf->data_capacity_; ++s) {
            if (leaf->key_slots_[s] == del || leaf->payload_slots_[s] == del) left = true;
        }
        if (left) ++scrub_left;
        index.insert(del, del);
    }

    const size_t nq = 1000000;
    std::mt19937_64 qrng(20261022);
    std::uniform_int_distribution<size_t> qpick(0, keys.size() - 1);
    std::vector<uint64_t> queries(nq);
    for (size_t i = 0; i < nq; ++i) queries[i] = keys[qpick(qrng)];
    volatile size_t sink = 0;
    size_t alex_miss = 0;
    auto l0 = std::chrono::steady_clock::now();
    for (uint64_t q : queries) {
        auto *payload_p = index.get_payload(q);
        if (payload_p == nullptr) ++alex_miss;
        else sink += static_cast<size_t>(*payload_p);
    }
    auto l1 = std::chrono::steady_clock::now();

    pgm::PGMIndex<uint64_t, 16> pgm(keys);
    auto p0 = std::chrono::steady_clock::now();
    for (uint64_t q : queries) {
        auto approx = pgm.search(q);
        auto it = std::lower_bound(keys.begin() + approx.lo, keys.begin() + approx.hi, q);
        sink += static_cast<size_t>(it - keys.begin());
    }
    auto p1 = std::chrono::steady_clock::now();
    double alex_ns = ns_per(l0, l1, nq);
    double pgm_ns = ns_per(p0, p1, nq);

    std::cout << "alex_build_ms " << std::chrono::duration<double, std::milli>(b1 - b0).count()
              << " alex_probes " << done
              << " alex_payload_retained " << payload
              << " alex_after_scrub " << scrub_left
              << " alex_max_key_exposed " << max_exposed
              << " alex_lookup_ns " << alex_ns
              << " alex_lookup_miss " << alex_miss
              << " pgm_lookup_ns " << pgm_ns
              << " alex_over_pgm " << (alex_ns / pgm_ns)
              << " sink " << sink
              << std::endl;
}

bool level_holds(const pgm::DynamicPGMIndex<uint64_t, uint64_t> &dyn, uint64_t key, bool &as_tomb) {
    as_tomb = false;
    for (const auto &lvl : dyn.debug_levels()) {
        auto it = std::lower_bound(
            lvl.begin(), lvl.end(), key,
            [](const auto &item, uint64_t k) { return item.first < k; });
        if (it != lvl.end() && it->first == key) {
            as_tomb = as_tomb || it->deleted();
            return true;
        }
    }
    return false;
}

void pgm_full_tombstone(const std::vector<uint64_t> &keys) {
    std::vector<std::pair<uint64_t, uint64_t>> pairs(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) pairs[i] = {keys[i], keys[i]};
    auto b0 = std::chrono::steady_clock::now();
    pgm::DynamicPGMIndex<uint64_t, uint64_t> dyn(pairs.begin(), pairs.end());
    auto b1 = std::chrono::steady_clock::now();
    size_t bytes0 = dyn.size_in_bytes();
    std::mt19937_64 rng(11);
    std::uniform_int_distribution<size_t> pick(keys.size() / 20, keys.size() - keys.size() / 20 - 1);
    std::vector<uint64_t> first;
    size_t retained = 0;
    size_t tomb = 0;
    for (int i = 0; i < 20; ++i) {
        uint64_t del = keys[pick(rng)];
        dyn.erase(del);
        first.push_back(del);
        bool as_tomb = false;
        if (level_holds(dyn, del, as_tomb)) ++retained;
        if (as_tomb) ++tomb;
    }
    size_t bytes20 = dyn.size_in_bytes();
    for (int i = 0; i < 800; ++i) dyn.erase(keys[pick(rng)]);
    size_t after = 0;
    size_t after_tomb = 0;
    for (uint64_t del : first) {
        bool as_tomb = false;
        if (level_holds(dyn, del, as_tomb)) ++after;
        if (as_tomb) ++after_tomb;
    }
    std::cout << "pgm_dyn build_ms " << std::chrono::duration<double, std::milli>(b1 - b0).count()
              << " bytes0 " << bytes0
              << " tombstone_20 " << retained
              << " tomb_flag_20 " << tomb
              << " bytes20 " << bytes20
              << " growth20 " << (bytes20 - bytes0)
              << " after_800_more " << after
              << " tomb_flag_800 " << after_tomb
              << " bytes_final " << dyn.size_in_bytes()
              << std::endl;
}

bool level_has_key(const pgm::DynamicPGMIndex<uint64_t, uint64_t> &dyn, uint64_t key) {
    for (const auto &lvl : dyn.debug_levels()) {
        for (const auto &item : lvl) {
            if (item.first == key) return true;
        }
    }
    return false;
}

void real_probe(const std::vector<uint64_t> &keys) {
    const size_t n = keys.size();
    if (n < 1000) return;
    std::vector<std::pair<uint64_t, uint64_t>> pairs(n);
    for (size_t i = 0; i < n; ++i) pairs[i] = {keys[i], keys[i]};
    pgm::DynamicPGMIndex<uint64_t, uint64_t> dyn(pairs.begin(), pairs.end());
    std::mt19937_64 rng(11);
    std::uniform_int_distribution<size_t> pick(n / 20, n - n / 20 - 1);
    std::vector<uint64_t> first;
    size_t retained = 0;
    for (int i = 0; i < 20; ++i) {
        uint64_t del = keys[pick(rng)];
        dyn.erase(del);
        first.push_back(del);
        if (level_has_key(dyn, del)) ++retained;
    }
    size_t before_merge = retained;
    for (int i = 0; i < 800; ++i) dyn.erase(keys[pick(rng)]);
    size_t after_merge = 0;
    for (uint64_t del : first) {
        if (level_has_key(dyn, del)) ++after_merge;
    }
    std::cout << "pgm_dyn tombstone_20 " << before_merge
              << " after_800_more " << after_merge
              << " bytes " << dyn.size_in_bytes()
              << std::endl;

    pgm::PGMIndex<uint64_t, 16> pgm(keys);
    shi::SHIIndex shi(keys, 16, 0x5A175EEDULL, 4096, shi::PLAMode::OptimalHull, 4);
    std::uniform_int_distribution<size_t> cand(0, n - 1);
    double stale_sum = 0;
    double t2_sum = 0;
    const int probes = 40;
    for (int p = 0; p < probes; ++p) {
        size_t rank = pick(rng);
        uint64_t del = keys[rank];
        auto approx = pgm.search(del);
        size_t lo = std::min(approx.lo, n - 1);
        size_t hi = std::min(approx.hi, n);
        if (hi <= lo) hi = std::min(lo + 1, n);
        uint64_t klo = keys[lo];
        uint64_t khi = keys[hi - 1];
        size_t in_stale = 1;
        for (int c = 0; c < 999; ++c) {
            uint64_t x = keys[cand(rng)];
            if (x >= klo && x <= khi) ++in_stale;
        }
        stale_sum += 1.0 / static_cast<double>(in_stale);

        auto before = shi.leaf_start_keys();
        shi::SHIIndex live = shi;
        live.erase(del);
        auto after = live.leaf_start_keys();
        size_t changed_lo = 0;
        size_t changed_hi = std::max(before.size(), after.size());
        size_t m = std::min(before.size(), after.size());
        while (changed_lo < m && before[changed_lo] == after[changed_lo]) ++changed_lo;
        size_t from_end = 0;
        while (from_end < m - changed_lo &&
               before[before.size() - 1 - from_end] == after[after.size() - 1 - from_end]) {
            ++from_end;
        }
        uint64_t wlo = (changed_lo < before.size()) ? before[changed_lo] : del;
        size_t right = (from_end == 0) ? before.size() : before.size() - from_end;
        uint64_t whi = (right < before.size()) ? before[right] : keys.back();
        if (whi < wlo) std::swap(wlo, whi);
        size_t in_t2 = 1;
        for (int c = 0; c < 999; ++c) {
            uint64_t x = keys[cand(rng)];
            if (x >= wlo && x <= whi && x != del) ++in_t2;
        }
        t2_sum += 1.0 / static_cast<double>(in_t2);
    }
    std::cout << "real_window probes " << probes
              << " stale_top1 " << (stale_sum / probes)
              << " t2_top1 " << (t2_sum / probes)
              << std::endl;
}

void window_only(const std::vector<uint64_t> &keys) {
    const size_t n = keys.size();
    pgm::PGMIndex<uint64_t, 16> pgm(keys);
    shi::SHIIndex shi(keys, 16, 0x5A175EEDULL, 4096, shi::PLAMode::OptimalHull, 4);
    std::mt19937_64 rng(11);
    std::uniform_int_distribution<size_t> pick(n / 20, n - n / 20 - 1);
    std::uniform_int_distribution<size_t> cand(0, n - 1);
    double stale_sum = 0;
    double t2_sum = 0;
    const int probes = 40;
    for (int p = 0; p < probes; ++p) {
        size_t rank = pick(rng);
        uint64_t del = keys[rank];
        auto approx = pgm.search(del);
        size_t lo = std::min(approx.lo, n - 1);
        size_t hi = std::min(approx.hi, n);
        if (hi <= lo) hi = std::min(lo + 1, n);
        uint64_t klo = keys[lo];
        uint64_t khi = keys[hi - 1];
        size_t in_stale = 1;
        for (int c = 0; c < 999; ++c) {
            uint64_t x = keys[cand(rng)];
            if (x >= klo && x <= khi) ++in_stale;
        }
        stale_sum += 1.0 / static_cast<double>(in_stale);
        auto before = shi.leaf_start_keys();
        shi::SHIIndex live = shi;
        live.erase(del);
        auto after = live.leaf_start_keys();
        size_t changed_lo = 0;
        size_t m = std::min(before.size(), after.size());
        while (changed_lo < m && before[changed_lo] == after[changed_lo]) ++changed_lo;
        size_t from_end = 0;
        while (from_end < m - changed_lo &&
               before[before.size() - 1 - from_end] == after[after.size() - 1 - from_end]) {
            ++from_end;
        }
        uint64_t wlo = (changed_lo < before.size()) ? before[changed_lo] : del;
        size_t right = (from_end == 0) ? before.size() : before.size() - from_end;
        uint64_t whi = (right < before.size()) ? before[right] : keys.back();
        if (whi < wlo) std::swap(wlo, whi);
        size_t in_t2 = 1;
        for (int c = 0; c < 999; ++c) {
            uint64_t x = keys[cand(rng)];
            if (x >= wlo && x <= whi && x != del) ++in_t2;
        }
        t2_sum += 1.0 / static_cast<double>(in_t2);
        if ((p + 1) % 10 == 0) {
            std::cout << "window_progress " << (p + 1) << std::endl;
        }
    }
    std::cout << "real_window probes " << probes
              << " stale_top1 " << (stale_sum / probes)
              << " t2_top1 " << (t2_sum / probes)
              << std::endl;
}

} // namespace

int main(int argc, char **argv) {
    std::string path;
    size_t max_keys = 0;
    size_t n_queries = 1000000;
    size_t n_deletes = 4;
    bool uint32_keys = false;
    bool do_alex = false;
    bool do_sweep = true;
    bool do_lookup = true;
    bool do_audit = false;
    bool do_post = false;
    bool do_post_audit = false;
    bool do_real = false;
    bool do_gap_curve = false;
    bool do_pgm_tomb = false;
    bool do_window = false;
    size_t post_extra = 0;
    size_t compact_at = 0;
    size_t audit_after = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char *flag) {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing ") + flag);
            return std::string(argv[++i]);
        };
        if (a == "--file") path = need("--file");
        else if (a == "--n") max_keys = static_cast<size_t>(std::stoull(need("--n")));
        else if (a == "--queries") n_queries = static_cast<size_t>(std::stoull(need("--queries")));
        else if (a == "--deletes") n_deletes = static_cast<size_t>(std::stoull(need("--deletes")));
        else if (a == "--u32") uint32_keys = true;
        else if (a == "--alex") do_alex = true;
        else if (a == "--no-sweep") do_sweep = false;
        else if (a == "--no-lookup") do_lookup = false;
        else if (a == "--audit") do_audit = true;
        else if (a == "--post") do_post = true;
        else if (a == "--post-audit") do_post_audit = true;
        else if (a == "--real") do_real = true;
        else if (a == "--post-extra") post_extra = static_cast<size_t>(std::stoull(need("--post-extra")));
        else if (a == "--compact-at") compact_at = static_cast<size_t>(std::stoull(need("--compact-at")));
        else if (a == "--gap-curve") do_gap_curve = true;
        else if (a == "--pgm-tomb") do_pgm_tomb = true;
        else if (a == "--window") do_window = true;
        else if (a == "--audit-after") audit_after = static_cast<size_t>(std::stoull(need("--audit-after")));
        else throw std::runtime_error("unknown arg " + a);
    }
    if (path.empty()) {
        std::cerr << "usage: bench_open --file PATH [--n N] [--u32] [--queries Q] [--alex]\n";
        return 2;
    }

    auto keys = load_file(path, max_keys, uint32_keys);
    const size_t n = keys.size();
    std::cout << "keys " << n << std::endl;

    // A full-file ALEX bulk load is the measurement. Building the other
    // indexes in the same process reserves a second copy of the key array.
    if (do_window && !do_lookup && !do_sweep && !do_post && !do_audit &&
        !do_post_audit && !do_real && !do_gap_curve && !do_pgm_tomb && !do_alex && audit_after == 0) {
        window_only(keys);
        return 0;
    }
    if (do_alex && !do_lookup && !do_sweep && !do_post && !do_audit &&
        !do_post_audit && !do_real && !do_gap_curve && !do_pgm_tomb && !do_window && audit_after == 0) {
        alex_payload(keys, 20);
        return 0;
    }
    if (do_pgm_tomb && !do_lookup && !do_sweep && !do_post && !do_audit &&
        !do_post_audit && !do_real && !do_gap_curve && !do_alex && audit_after == 0) {
        pgm_full_tombstone(keys);
        return 0;
    }
    if (audit_after > 0 && !do_lookup && !do_sweep && !do_post && !do_audit &&
        !do_post_audit && !do_real && !do_gap_curve && !do_alex && !do_pgm_tomb) {
        shi::SHIIndex only(keys, 16, 0x5A175EEDULL, 4096, shi::PLAMode::OptimalHull, 4);
        std::mt19937_64 arng(20261022);
        std::uniform_int_distribution<size_t> apick(0, n - 1);
        std::vector<size_t> gone;
        while (gone.size() < audit_after && gone.size() < n / 2) {
            size_t r = apick(arng);
            if (std::find(gone.begin(), gone.end(), r) == gone.end()) gone.push_back(r);
        }
        std::sort(gone.begin(), gone.end());
        for (size_t r : gone) only.erase(keys[r]);
        size_t misses = 0;
        size_t seen = 0;
        size_t cursor = 0;
        for (size_t i = 0; i < n; ++i) {
            if (cursor < gone.size() && gone[cursor] == i) {
                ++cursor;
                continue;
            }
            size_t expect = i - cursor;
            if (only.lookup(keys[i]) != expect) ++misses;
            ++seen;
            if (seen % 50000000 == 0) {
                std::cout << "audit_after_progress " << seen << " misses " << misses << std::endl;
            }
        }
        std::cout << "audit_after deletes " << gone.size()
                  << " survivors " << seen
                  << " misses " << misses
                  << std::endl;
        return 0;
    }

    std::mt19937_64 rng(20261022);
    std::uniform_int_distribution<size_t> qpick(0, n - 1);
    std::vector<uint64_t> queries(n_queries);
    for (size_t i = 0; i < n_queries; ++i) queries[i] = keys[qpick(rng)];

    pgm::PGMIndex<uint64_t, 16> pgm(keys);
    shi::SHIIndex shi(keys, 16, 0x5A175EEDULL, 4096, shi::PLAMode::OptimalHull, 4);
    shi.set_compact_limit(compact_at);
    shi::HIBTree btree(keys);
    std::cout << "built shi_segs " << shi.segment_count()
              << " shi_bytes " << shi.model_bytes()
              << " pgm_segs " << pgm.segments_count()
              << " pgm_bytes " << pgm.size_in_bytes()
              << " btree_bytes " << btree.model_bytes()
              << " btree_height " << btree.height()
              << std::endl;

    if (do_post_audit) {
        const size_t cut = n / 2;
        shi.erase(keys[cut]);
        size_t misses = 0;
        size_t seen = 0;
        for (size_t i = 0; i < n; ++i) {
            if (i == cut) continue;
            size_t expect = (i < cut) ? i : (i - 1);
            if (shi.lookup(keys[i]) != expect) ++misses;
            ++seen;
            if (seen % 50000000 == 0) {
                std::cout << "post_audit_progress " << seen << " misses " << misses << std::endl;
            }
        }
        std::cout << "post_audit_keys " << seen << " misses " << misses << std::endl;
    }

    if (do_audit) {
        size_t misses = 0;
        for (size_t i = 0; i < n; ++i) {
            if (shi.lookup(keys[i]) != i) ++misses;
            if ((i + 1) % 50000000 == 0) {
                std::cout << "audit_progress " << (i + 1) << " misses " << misses << std::endl;
            }
        }
        std::cout << "audit_keys " << n << " misses " << misses << std::endl;
    }

    if (do_lookup) {
        volatile size_t sink = 0;
        auto p0 = std::chrono::steady_clock::now();
        for (uint64_t q : queries) {
            auto approx = pgm.search(q);
            auto it = std::lower_bound(keys.begin() + approx.lo, keys.begin() + approx.hi, q);
            sink += static_cast<size_t>(it - keys.begin());
        }
        auto p1 = std::chrono::steady_clock::now();
        auto s0 = std::chrono::steady_clock::now();
        for (uint64_t q : queries) sink += shi.lookup(q);
        auto s1 = std::chrono::steady_clock::now();
        auto t0 = std::chrono::steady_clock::now();
        for (uint64_t q : queries) sink += btree.lookup(q);
        auto t1 = std::chrono::steady_clock::now();
        double pgm_ns = ns_per(p0, p1, n_queries);
        double shi_ns = ns_per(s0, s1, n_queries);
        double bt_ns = ns_per(t0, t1, n_queries);
        std::cout << "lookup_queries " << n_queries
                  << " pgm_ns " << pgm_ns
                  << " shi_ns " << shi_ns
                  << " shi_over_pgm " << (shi_ns / pgm_ns)
                  << " btree_ns " << bt_ns
                  << " btree_over_pgm " << (bt_ns / pgm_ns)
                  << " sink " << sink
                  << std::endl;
    }

    if (do_audit || do_post) {
        std::vector<size_t> gone;
        auto erase_original = [&](size_t original) {
            shi.erase(keys[original]);
            gone.insert(std::lower_bound(gone.begin(), gone.end(), original), original);
        };
        auto fill_probes = [&](std::vector<uint64_t> &post_keys, std::vector<size_t> &post_ranks) {
            std::uniform_int_distribution<size_t> pick(0, n - 1);
            for (size_t t = 0; t < n_queries; ++t) {
                size_t original = pick(rng);
                while (std::binary_search(gone.begin(), gone.end(), original)) original = pick(rng);
                post_keys[t] = keys[original];
                post_ranks[t] = original - static_cast<size_t>(
                    std::lower_bound(gone.begin(), gone.end(), original) - gone.begin());
            }
        };
        auto time_post = [&](const char *tag) {
            std::vector<uint64_t> post_keys(n_queries);
            std::vector<size_t> post_ranks(n_queries);
            fill_probes(post_keys, post_ranks);
            volatile size_t sink = 0;
            auto d0 = std::chrono::steady_clock::now();
            for (size_t t = 0; t < n_queries; ++t) sink += shi.lookup(post_keys[t]);
            auto d1 = std::chrono::steady_clock::now();
            auto b0 = std::chrono::steady_clock::now();
            for (size_t t = 0; t < n_queries; ++t) sink += btree.lookup(post_keys[t]);
            auto b1 = std::chrono::steady_clock::now();
            size_t post_miss = 0;
            for (size_t t = 0; t < n_queries; ++t) {
                if (shi.lookup(post_keys[t]) != post_ranks[t]) ++post_miss;
            }
            std::cout << tag
                      << " deletes " << gone.size()
                      << " probes " << n_queries
                      << " misses " << post_miss
                      << " shi_ns " << ns_per(d0, d1, n_queries)
                      << " btree_ns " << ns_per(b0, b1, n_queries)
                      << " boundaries " << shi.boundary_count()
                      << " compactions " << shi.compaction_count()
                      << " compact_ms " << (static_cast<double>(shi.compact_ns()) / 1e6)
                      << " compact_max_ms " << (static_cast<double>(shi.compact_max_ns()) / 1e6)
                      << " sink " << sink
                      << std::endl;
        };
        std::uniform_int_distribution<size_t> extra(0, n - 1);
        if (do_gap_curve) {
            const size_t steps[] = {1, 8, 32, 128, 512, 1024, 4096};
            for (size_t step : steps) {
                while (gone.size() < step && gone.size() < n) {
                    size_t original = extra(rng);
                    if (std::binary_search(gone.begin(), gone.end(), original)) continue;
                    erase_original(original);
                }
                time_post("gap_curve");
            }
        } else {
        erase_original(n / 2);
        time_post("post_delete");
        size_t added = 0;
        while (added < post_extra && gone.size() < n) {
            size_t original = extra(rng);
            if (std::binary_search(gone.begin(), gone.end(), original)) continue;
            erase_original(original);
            ++added;
        }
        if (post_extra > 0) time_post("post_many");
        }
    }

    if (do_sweep) {
        const uint32_t moduli[] = {0, 1024, 4096, 16384};
        std::uniform_int_distribution<size_t> dpick(n / 20, n - n / 20 - 1);
        for (uint32_t m : moduli) {
            auto b0 = std::chrono::steady_clock::now();
            shi::SHIIndex idx(keys, 16, 0x5A175EEDULL, m, shi::PLAMode::OptimalHull, 4);
            auto b1 = std::chrono::steady_clock::now();
            double refit_sum = 0;
            size_t refit_max = 0;
            size_t identical = 0;
            for (size_t d = 0; d < n_deletes; ++d) {
                uint64_t del = keys[dpick(rng)];
                std::vector<uint64_t> survivors;
                survivors.reserve(n - 1);
                auto it = std::lower_bound(keys.begin(), keys.end(), del);
                survivors.insert(survivors.end(), keys.begin(), it);
                survivors.insert(survivors.end(), it + 1, keys.end());
                shi::SHIIndex live(keys, 16, 0x5A175EEDULL, m, shi::PLAMode::OptimalHull, 4);
                auto st = live.erase(del);
                refit_sum += static_cast<double>(st.segments_refit);
                refit_max = std::max(refit_max, st.segments_refit);
                shi::SHIIndex fresh(survivors, 16, 0x5A175EEDULL, m, shi::PLAMode::OptimalHull, 4);
                if (live.serialize_bytes() == fresh.serialize_bytes()) ++identical;
            }
            std::cout << "sweep M " << m
                      << " segs " << idx.segment_count()
                      << " bytes " << idx.model_bytes()
                      << " space_vs_pgm " << (static_cast<double>(idx.model_bytes()) / static_cast<double>(pgm.size_in_bytes()))
                      << " build_ms " << std::chrono::duration<double, std::milli>(b1 - b0).count()
                      << " deletes " << n_deletes
                      << " identical " << identical
                      << " refit_mean " << (refit_sum / static_cast<double>(n_deletes))
                      << " refit_max " << refit_max
                      << std::endl;
        }
    }

    if (do_alex) alex_payload(keys, 20);
    if (do_real) real_probe(keys);
    return 0;
}

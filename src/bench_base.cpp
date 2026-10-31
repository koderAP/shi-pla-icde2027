#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "shi/bptree.hpp"
#include "shi/shi_pla.hpp"

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

double pct(std::vector<double> v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    double idx = p * static_cast<double>(v.size() - 1);
    size_t i = static_cast<size_t>(idx);
    return v[i];
}

double mean_of(const std::vector<double> &v) {
    if (v.empty()) return 0;
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

void report(const char *name, const std::vector<double> &us) {
    std::cout << name
              << " n " << us.size()
              << " mean_us " << mean_of(us)
              << " p50 " << pct(us, 0.50)
              << " p90 " << pct(us, 0.90)
              << " p99 " << pct(us, 0.99)
              << " max " << pct(us, 1.0)
              << std::endl;
}

size_t zipf_rank(std::mt19937_64 &rng, size_t n) {
    std::uniform_real_distribution<double> dist(std::numeric_limits<double>::min(), 1.0);
    double u = dist(rng);
    size_t raw = static_cast<size_t>(std::pow(u, 100.0) * static_cast<double>(n));
    if (raw >= n) raw = n - 1;
    uint64_t h = static_cast<uint64_t>(raw) * 0x9E3779B185EBCA87ULL;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    return static_cast<size_t>(h % n);
}

template <int FAN>
void check_tree(const char *label) {
    std::mt19937_64 rng(20261022);
    std::uniform_int_distribution<uint64_t> dist(1, 100000000000ULL);
    std::vector<uint64_t> keys(20000);
    for (uint64_t &k : keys) k = dist(rng);
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    shi::U64BPlus<FAN> tree;
    tree.bulk_load(keys.data(), keys.size());
    for (uint64_t k : keys) {
        if (!tree.contains(k)) throw std::runtime_error(std::string(label) + " missing after bulk");
    }
    std::vector<uint64_t> order = keys;
    std::shuffle(order.begin(), order.end(), rng);
    size_t left = keys.size();
    for (uint64_t k : order) {
        if (!tree.erase(k) || tree.contains(k) || tree.size() != --left) {
            throw std::runtime_error(std::string(label) + " erase");
        }
    }
    if (tree.size() != 0) throw std::runtime_error(std::string(label) + " not empty");
    for (uint64_t k : order) {
        if (!tree.insert(k) || tree.contains(k) == false) throw std::runtime_error(std::string(label) + " insert");
    }
    if (tree.size() != keys.size()) throw std::runtime_error(std::string(label) + " size");
    for (uint64_t k : keys) {
        if (!tree.contains(k)) throw std::runtime_error(std::string(label) + " missing after insert");
    }
    std::cout << "self_check " << label << " keys " << keys.size() << " height " << tree.height() << std::endl;
}

void run_self_check() {
    check_tree<4>("fan4");
    check_tree<256>("fan256");
}

void run_btree(const std::vector<uint64_t> &keys) {
    shi::U64BPlus<256> tree;
    auto t0 = std::chrono::steady_clock::now();
    tree.bulk_load(keys.data(), keys.size());
    auto t1 = std::chrono::steady_clock::now();
    std::cout << "btree_build_ms "
              << std::chrono::duration<double, std::milli>(t1 - t0).count()
              << " height " << tree.height() << " n " << tree.size() << std::endl;
    std::mt19937_64 rng(20261022);
    std::uniform_int_distribution<size_t> pick(0, keys.size() - 1);
    size_t misses = 0;
    constexpr size_t kQueries = 1000000;
    auto q0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < kQueries; ++i) {
        if (!tree.contains(keys[pick(rng)])) ++misses;
    }
    auto q1 = std::chrono::steady_clock::now();
    std::cout << "btree_lookup_ns "
              << std::chrono::duration<double, std::nano>(q1 - q0).count() / kQueries
              << " misses " << misses << std::endl;
    if (misses != 0) throw std::runtime_error("btree lookup miss");
    std::unordered_set<size_t> chosen;
    std::vector<uint64_t> victims;
    while (victims.size() < 512) {
        size_t r = pick(rng);
        if (chosen.insert(r).second) victims.push_back(keys[r]);
    }
    std::vector<double> us;
    us.reserve(victims.size());
    for (uint64_t k : victims) {
        auto a = std::chrono::steady_clock::now();
        if (!tree.erase(k)) throw std::runtime_error("btree erase miss");
        auto b = std::chrono::steady_clock::now();
        us.push_back(std::chrono::duration<double, std::micro>(b - a).count());
    }
    report("btree_del", us);
    std::cout << "btree_left " << tree.size() << std::endl;
}

shi::SHIIndex make_index(const std::vector<uint64_t> &keys) {
    shi::SHIIndex idx(keys, 16, 0x5A175EEDULL, 4096, shi::PLAMode::OptimalHull, 4);
    idx.set_compact_limit(32);
    return idx;
}

std::vector<uint64_t> distinct_keys(const std::vector<uint64_t> &keys, size_t want, std::mt19937_64 &rng, bool zipf) {
    std::uniform_int_distribution<size_t> pick(0, keys.size() - 1);
    std::unordered_set<size_t> chosen;
    std::vector<uint64_t> out;
    while (out.size() < want) {
        size_t r = zipf ? zipf_rank(rng, keys.size()) : pick(rng);
        if (chosen.insert(r).second) out.push_back(keys[r]);
    }
    return out;
}

void time_deletes(const char *name, shi::SHIIndex &idx, const std::vector<uint64_t> &victims) {
    std::vector<double> us;
    us.reserve(victims.size());
    for (uint64_t k : victims) {
        auto a = std::chrono::steady_clock::now();
        idx.erase(k);
        auto b = std::chrono::steady_clock::now();
        us.push_back(std::chrono::duration<double, std::micro>(b - a).count());
    }
    report(name, us);
    std::cout << name << "_boundaries " << idx.boundary_count()
              << " compactions " << idx.compaction_count()
              << " compact_ms " << static_cast<double>(idx.compact_ns()) / 1e6
              << std::endl;
}

void run_seq(const std::vector<uint64_t> &keys) {
    {
        auto idx = make_index(keys);
        std::mt19937_64 rng(20261022);
        auto victims = distinct_keys(keys, 512, rng, true);
        time_deletes("zipf_del", idx, victims);
    }
    {
        auto idx = make_index(keys);
        std::vector<uint64_t> victims(keys.begin() + static_cast<std::ptrdiff_t>(keys.size() / 2),
                                      keys.begin() + static_cast<std::ptrdiff_t>(keys.size() / 2 + 512));
        time_deletes("range_del", idx, victims);
    }
}

void run_mix(const std::vector<uint64_t> &keys, size_t ops) {
    auto idx = make_index(keys);
    std::mt19937_64 rng(20261022);
    std::uniform_int_distribution<size_t> pick(0, keys.size() - 1);
    std::uniform_int_distribution<uint64_t> fresh(0, UINT64_MAX);
    std::uniform_int_distribution<int> kind(0, 99);
    std::vector<char> live(keys.size(), 1);
    std::vector<uint64_t> inserted;
    inserted.reserve(ops / 5 + 8);
    double lookup_ns = 0;
    double delete_us = 0;
    double insert_us = 0;
    size_t n_lookup = 0;
    size_t n_delete = 0;
    size_t n_insert = 0;
    size_t lookup_miss = 0;
    size_t guard = 0;
    while (n_lookup + n_delete + n_insert < ops) {
        int k = kind(rng);
        if (k < 80) {
            uint64_t key = keys[pick(rng)];
            auto a = std::chrono::steady_clock::now();
            bool hit = idx.contains(key);
            auto b = std::chrono::steady_clock::now();
            lookup_ns += std::chrono::duration<double, std::nano>(b - a).count();
            if (!hit) ++lookup_miss;
            ++n_lookup;
        } else if (k < 90) {
            uint64_t key = 0;
            bool found = false;
            for (int t = 0; t < 64; ++t) {
                size_t r = pick(rng);
                if (live[r]) {
                    key = keys[r];
                    live[r] = 0;
                    found = true;
                    break;
                }
            }
            if (!found) continue;
            auto a = std::chrono::steady_clock::now();
            idx.erase(key);
            auto b = std::chrono::steady_clock::now();
            delete_us += std::chrono::duration<double, std::micro>(b - a).count();
            ++n_delete;
        } else {
            uint64_t key = 0;
            bool ok = false;
            for (int t = 0; t < 64; ++t) {
                key = fresh(rng);
                if (!idx.contains(key)) {
                    ok = true;
                    break;
                }
            }
            if (!ok) continue;
            auto a = std::chrono::steady_clock::now();
            idx.insert(key);
            auto b = std::chrono::steady_clock::now();
            insert_us += std::chrono::duration<double, std::micro>(b - a).count();
            inserted.push_back(key);
            ++n_insert;
        }
        if (++guard > ops * 4) throw std::runtime_error("mix made no progress");
    }
    std::cout << "mix ops " << ops
              << " lookup_ns " << (n_lookup ? lookup_ns / static_cast<double>(n_lookup) : 0)
              << " n_lookup " << n_lookup
              << " lookup_miss " << lookup_miss
              << " delete_us " << (n_delete ? delete_us / static_cast<double>(n_delete) : 0)
              << " n_delete " << n_delete
              << " insert_us " << (n_insert ? insert_us / static_cast<double>(n_insert) : 0)
              << " n_insert " << n_insert
              << " boundaries " << idx.boundary_count()
              << " compactions " << idx.compaction_count()
              << " compact_ms " << static_cast<double>(idx.compact_ns()) / 1e6
              << std::endl;
    std::vector<uint64_t> survivors;
    survivors.reserve(idx.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        if (live[i]) survivors.push_back(keys[i]);
    }
    survivors.insert(survivors.end(), inserted.begin(), inserted.end());
    std::sort(survivors.begin(), survivors.end());
    shi::SHIIndex rebuilt(survivors, 16, 0x5A175EEDULL, 4096, shi::PLAMode::OptimalHull, 4);
    bool match = idx.serialize_bytes() == rebuilt.serialize_bytes();
    std::cout << "mix_match " << (match ? 1 : 0)
              << " survivors " << survivors.size()
              << " bytes " << idx.serialize_bytes().size()
              << std::endl;
}

}  // namespace

int main(int argc, char **argv) {
    std::ios::sync_with_stdio(false);
    bool self = false;
    bool btree = false;
    bool seq = false;
    bool mix = false;
    bool uint32_keys = false;
    std::string path;
    size_t max_keys = 0;
    size_t ops = 20000;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char *flag) {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing ") + flag);
            return std::string(argv[++i]);
        };
        if (a == "--self-check") self = true;
        else if (a == "--btree") btree = true;
        else if (a == "--seq") seq = true;
        else if (a == "--mix") mix = true;
        else if (a == "--u32") uint32_keys = true;
        else if (a == "--file") path = need("--file");
        else if (a == "--n") max_keys = static_cast<size_t>(std::stoull(need("--n")));
        else if (a == "--ops") ops = static_cast<size_t>(std::stoull(need("--ops")));
        else throw std::runtime_error("unknown arg " + a);
    }
    if (self || path.empty()) {
        run_self_check();
        if (path.empty()) return 0;
    }
    auto keys = load_file(path, max_keys, uint32_keys);
    std::cout << "keys " << keys.size() << std::endl;
    if (btree) run_btree(keys);
    if (seq) run_seq(keys);
    if (mix) run_mix(keys, ops);
    return 0;
}

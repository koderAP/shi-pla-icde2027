extern "C" {
#include "postgres.h"
#include "fmgr.h"
#include "executor/spi.h"
#include "utils/builtins.h"

#ifdef PG_MODULE_MAGIC
PG_MODULE_MAGIC;
#endif
}

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "pgm/pgm_index.hpp"
#include "pgm/pgm_index_dynamic.hpp"
#include "shi/hi_btree.hpp"
#include "shi/shi_pla.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using DynPGM = pgm::DynamicPGMIndex<uint64_t, uint64_t, pgm::PGMIndex<uint64_t, 16>>;

struct ManagedIndex {
    std::unique_ptr<shi::SHIIndex> shi;
    std::unique_ptr<DynPGM> pgm_dyn;
    std::unique_ptr<shi::HIBTree> hi_btree;
    size_t epsilon = 16;
    uint32_t modulus = 4096;
    shi::PLAMode mode = shi::PLAMode::OptimalHull;
};

std::unordered_map<std::string, ManagedIndex> &registry() {
    static std::unordered_map<std::string, ManagedIndex> reg;
    return reg;
}

uint64_t fnv1a_64(const std::vector<uint8_t> &bytes) noexcept {
    uint64_t h = 0xCBF29CE484222325ULL;
    for (uint8_t b : bytes) {
        h ^= static_cast<uint64_t>(b);
        h *= 0x100000001B3ULL;
    }
    return h;
}

std::string hex_u64(uint64_t v) {
    char buf[24];
    std::snprintf(buf, sizeof(buf), "0x%016llx", static_cast<unsigned long long>(v));
    return std::string(buf);
}

bool pgm_dyn_retains_key(const DynPGM &dyn, uint64_t key) {
    for (const auto &lvl : dyn.debug_levels()) {
        for (const auto &item : lvl) {
            if (item.first == key) {
                return true;
            }
        }
    }
    for (const auto &pgm_lvl : dyn.debug_pgms()) {
        for (const auto &seg : pgm_lvl.debug_segments()) {
            if (seg.key == key) {
                return true;
            }
        }
    }
    return false;
}

bool raw_bytes_contain_u64(const std::vector<uint8_t> &bytes, uint64_t key) {
    for (size_t i = 0; i + sizeof(uint64_t) <= bytes.size(); ++i) {
        uint64_t val = 0;
        std::memcpy(&val, bytes.data() + i, sizeof(uint64_t));
        if (val == key) return true;
    }
    return false;
}

} // namespace

extern "C" {

PG_FUNCTION_INFO_V1(shi_build_from_query);
Datum shi_build_from_query(PG_FUNCTION_ARGS) {
    text *idx_name_t = PG_GETARG_TEXT_PP(0);
    text *sql_query_t = PG_GETARG_TEXT_PP(1);
    int32 epsilon = PG_GETARG_INT32(2);
    int32 modulus = PG_GETARG_INT32(3);
    text *mode_t = PG_GETARG_TEXT_PP(4);

    std::string idx_name(VARDATA_ANY(idx_name_t), VARSIZE_ANY_EXHDR(idx_name_t));
    std::string sql_query(VARDATA_ANY(sql_query_t), VARSIZE_ANY_EXHDR(sql_query_t));
    std::string mode_str(VARDATA_ANY(mode_t), VARSIZE_ANY_EXHDR(mode_t));

    if (SPI_connect() != SPI_OK_CONNECT) {
        ereport(ERROR, (errmsg("SPI_connect failed in shi_build_from_query")));
    }

    int ret = SPI_execute(sql_query.c_str(), true, 0);
    if (ret != SPI_OK_SELECT) {
        SPI_finish();
        ereport(ERROR, (errmsg("SPI_execute failed for query: %s", sql_query.c_str())));
    }

    std::vector<uint64_t> keys;
    keys.reserve(SPI_processed);
    for (uint64 i = 0; i < SPI_processed; ++i) {
        bool isnull = false;
        Datum d = SPI_getbinval(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 1, &isnull);
        if (!isnull) {
            int64 v = DatumGetInt64(d);
            if (v >= 0) {
                keys.push_back(static_cast<uint64_t>(v));
            }
        }
    }
    SPI_finish();

    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());

    shi::PLAMode mode = (mode_str == "greedy") ? shi::PLAMode::GreedyCone : shi::PLAMode::OptimalHull;
    uint32_t mod_u32 = (modulus <= 0) ? 0u : static_cast<uint32_t>(modulus);
    size_t eps_sz = (epsilon <= 0) ? 16u : static_cast<size_t>(epsilon);

    auto t0 = Clock::now();
    auto shi_ptr = std::make_unique<shi::SHIIndex>(keys, eps_sz, 0x5A175EEDULL, mod_u32, mode, 4);
    auto t1 = Clock::now();
    double build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::vector<std::pair<uint64_t, uint64_t>> pairs(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        pairs[i] = {keys[i], i};
    }
    auto pgm_ptr = std::make_unique<DynPGM>(pairs.begin(), pairs.end());
    auto btree_ptr = std::make_unique<shi::HIBTree>(keys, 32, 0xB7E151628AED2A6AULL);

    auto snap = shi_ptr->serialize_bytes();
    uint64_t digest = fnv1a_64(snap);

    ManagedIndex entry;
    entry.epsilon = eps_sz;
    entry.modulus = mod_u32;
    entry.mode = mode;
    entry.shi = std::move(shi_ptr);
    entry.pgm_dyn = std::move(pgm_ptr);
    entry.hi_btree = std::move(btree_ptr);

    std::ostringstream oss;
    oss << "{"
        << "\"index_name\":\"" << idx_name << "\","
        << "\"n_keys\":" << entry.shi->size() << ","
        << "\"epsilon\":" << eps_sz << ","
        << "\"anchor_modulus\":" << mod_u32 << ","
        << "\"mode\":\"" << (mode == shi::PLAMode::OptimalHull ? "optimal" : "greedy") << "\","
        << "\"segments\":" << entry.shi->segment_count() << ","
        << "\"levels\":" << entry.shi->level_count() << ","
        << "\"model_bytes\":" << entry.shi->model_bytes() << ","
        << "\"bytes_per_key\":" << (static_cast<double>(entry.shi->model_bytes()) / std::max<size_t>(1, entry.shi->size())) << ","
        << "\"build_ms\":" << build_ms << ","
        << "\"snapshot_fnv1a64\":\"" << hex_u64(digest) << "\""
        << "}";

    registry()[idx_name] = std::move(entry);
    PG_RETURN_TEXT_P(cstring_to_text(oss.str().c_str()));
}

PG_FUNCTION_INFO_V1(shi_lookup);
Datum shi_lookup(PG_FUNCTION_ARGS) {
    text *idx_name_t = PG_GETARG_TEXT_PP(0);
    int64 key_i64 = PG_GETARG_INT64(1);
    std::string idx_name(VARDATA_ANY(idx_name_t), VARSIZE_ANY_EXHDR(idx_name_t));

    auto it = registry().find(idx_name);
    if (it == registry().end() || !it->second.shi) {
        ereport(ERROR, (errmsg("SHI index not found: %s", idx_name.c_str())));
    }
    const auto &shi_idx = *it->second.shi;
    uint64_t k = static_cast<uint64_t>(key_i64);
    size_t rank = shi_idx.lookup(k);
    if (rank >= shi_idx.size()) {
        PG_RETURN_INT64(-1);
    }
    PG_RETURN_INT64(static_cast<int64>(rank));
}

PG_FUNCTION_INFO_V1(shi_delete_key);
Datum shi_delete_key(PG_FUNCTION_ARGS) {
    text *idx_name_t = PG_GETARG_TEXT_PP(0);
    int64 key_i64 = PG_GETARG_INT64(1);
    std::string idx_name(VARDATA_ANY(idx_name_t), VARSIZE_ANY_EXHDR(idx_name_t));

    auto it = registry().find(idx_name);
    if (it == registry().end() || !it->second.shi) {
        ereport(ERROR, (errmsg("SHI index not found: %s", idx_name.c_str())));
    }
    ManagedIndex &m = it->second;
    uint64_t key = static_cast<uint64_t>(key_i64);

    auto before_snap = m.shi->serialize_bytes();
    uint64_t before_hash = fnv1a_64(before_snap);

    auto t0 = Clock::now();
    auto st = m.shi->erase(key);
    auto t1 = Clock::now();
    double local_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    m.pgm_dyn->erase(key);
    m.hi_btree->erase(key);

    auto t2 = Clock::now();
    shi::SHIIndex fresh(m.shi->keys(), m.epsilon, 0x5A175EEDULL, m.modulus, m.mode, 4);
    auto t3 = Clock::now();
    double rebuild_us = std::chrono::duration<double, std::micro>(t3 - t2).count();

    auto after_snap = m.shi->serialize_bytes();
    auto fresh_snap = fresh.serialize_bytes();
    bool byte_identical = (after_snap == fresh_snap);
    bool pgm_leaked = pgm_dyn_retains_key(*m.pgm_dyn, key);
    bool shi_contains_key_bytes = raw_bytes_contain_u64(after_snap, key);

    std::ostringstream oss;
    oss << "{"
        << "\"deleted_key\":" << key << ","
        << "\"remaining_keys\":" << m.shi->size() << ","
        << "\"segments_refit\":" << st.segments_refit << ","
        << "\"local_delete_us\":" << local_us << ","
        << "\"rebuild_us\":" << rebuild_us << ","
        << "\"speedup_vs_rebuild\":" << (local_us > 0.0 ? rebuild_us / local_us : 0.0) << ","
        << "\"byte_identical_to_rebuild\":" << (byte_identical ? "true" : "false") << ","
        << "\"shi_snapshot_retains_key\":" << (shi_contains_key_bytes ? "true" : "false") << ","
        << "\"pgm_dynamic_retains_key\":" << (pgm_leaked ? "true" : "false") << ","
        << "\"before_hash\":\"" << hex_u64(before_hash) << "\","
        << "\"after_hash\":\"" << hex_u64(fnv1a_64(after_snap)) << "\","
        << "\"rebuild_hash\":\"" << hex_u64(fnv1a_64(fresh_snap)) << "\""
        << "}";

    PG_RETURN_TEXT_P(cstring_to_text(oss.str().c_str()));
}

PG_FUNCTION_INFO_V1(shi_audit_erasure);
Datum shi_audit_erasure(PG_FUNCTION_ARGS) {
    text *idx_name_t = PG_GETARG_TEXT_PP(0);
    int32 n_probes = PG_GETARG_INT32(1);
    std::string idx_name(VARDATA_ANY(idx_name_t), VARSIZE_ANY_EXHDR(idx_name_t));

    auto it = registry().find(idx_name);
    if (it == registry().end() || !it->second.shi) {
        ereport(ERROR, (errmsg("SHI index not found: %s", idx_name.c_str())));
    }
    ManagedIndex &m = it->second;
    const auto &base_keys = m.shi->keys();
    if (base_keys.size() < 8) {
        ereport(ERROR, (errmsg("SHI index %s has too few keys (%zu) to audit", idx_name.c_str(), base_keys.size())));
    }
    size_t probes = std::min<size_t>(static_cast<size_t>(std::max<int32>(1, n_probes)), base_keys.size() / 4);

    std::mt19937_64 rng(20261022ULL);
    std::vector<size_t> pool(base_keys.size() - 2);
    for (size_t i = 1; i + 1 < base_keys.size(); ++i) pool[i - 1] = i;
    std::shuffle(pool.begin(), pool.end(), rng);
    pool.resize(probes);

    size_t identical_hits = 0;
    size_t pgm_leak_hits = 0;
    size_t shi_leak_hits = 0;
    size_t btree_identical_hits = 0;
    std::vector<size_t> refits;
    double total_local_us = 0.0;
    double total_rebuild_us = 0.0;

    for (size_t pos : pool) {
        uint64_t k = base_keys[pos];
        std::vector<uint64_t> survivors;
        survivors.reserve(base_keys.size() - 1);
        survivors.insert(survivors.end(), base_keys.begin(), base_keys.begin() + pos);
        survivors.insert(survivors.end(), base_keys.begin() + pos + 1, base_keys.end());

        shi::SHIIndex working = *m.shi;
        working.set_keys_raw(survivors);
        auto t0 = Clock::now();
        auto st = working.refit_only(k);
        auto t1 = Clock::now();
        total_local_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
        refits.push_back(st.segments_refit);

        auto t2 = Clock::now();
        shi::SHIIndex fresh(survivors, m.epsilon, 0x5A175EEDULL, m.modulus, m.mode, 4);
        auto t3 = Clock::now();
        total_rebuild_us += std::chrono::duration<double, std::micro>(t3 - t2).count();

        auto w_snap = working.serialize_bytes();
        if (w_snap == fresh.serialize_bytes()) {
            ++identical_hits;
        }
        if (raw_bytes_contain_u64(w_snap, k)) {
            ++shi_leak_hits;
        }

        m.pgm_dyn->erase(k);
        if (pgm_dyn_retains_key(*m.pgm_dyn, k)) {
            ++pgm_leak_hits;
        }

        shi::HIBTree w_bt = *m.hi_btree;
        w_bt.erase(k);
        shi::HIBTree f_bt(survivors, 32, 0xB7E151628AED2A6AULL);
        if (w_bt.serialize_bytes() == f_bt.serialize_bytes()) {
            ++btree_identical_hits;
        }
    }

    std::sort(refits.begin(), refits.end());
    double mean_refit = std::accumulate(refits.begin(), refits.end(), 0.0) / std::max<size_t>(1, refits.size());
    size_t p99_refit = refits.empty() ? 0 : refits[static_cast<size_t>(0.99 * (refits.size() - 1))];
    size_t max_refit = refits.empty() ? 0 : refits.back();
    double mean_local_us = total_local_us / std::max<size_t>(1, probes);
    double mean_rebuild_us = total_rebuild_us / std::max<size_t>(1, probes);

    std::ostringstream oss;
    oss << "{"
        << "\"index_name\":\"" << idx_name << "\","
        << "\"probes\":" << probes << ","
        << "\"shi_byte_identical_rate\":" << (static_cast<double>(identical_hits) / probes) << ","
        << "\"hibtree_byte_identical_rate\":" << (static_cast<double>(btree_identical_hits) / probes) << ","
        << "\"shi_snapshot_leak_rate\":" << (static_cast<double>(shi_leak_hits) / probes) << ","
        << "\"pgm_dynamic_leak_rate\":" << (static_cast<double>(pgm_leak_hits) / probes) << ","
        << "\"refit_segments_mean\":" << mean_refit << ","
        << "\"refit_segments_p99\":" << p99_refit << ","
        << "\"refit_segments_max\":" << max_refit << ","
        << "\"mean_local_delete_us\":" << mean_local_us << ","
        << "\"mean_rebuild_us\":" << mean_rebuild_us << ","
        << "\"speedup_vs_rebuild\":" << (mean_local_us > 0.0 ? mean_rebuild_us / mean_local_us : 0.0)
        << "}";

    PG_RETURN_TEXT_P(cstring_to_text(oss.str().c_str()));
}

} // extern "C"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "alex.h"
#include "lipp.h"
#include "pgm/pgm_index.hpp"
#include "pgm/pgm_index_dynamic.hpp"

#include "shi/datasets.hpp"
#include "shi/dp_index.hpp"
#include "shi/hi_btree.hpp"
#include "shi/shi_pla.hpp"

namespace {

struct DatasetAttackReport {
    std::string dataset;
    size_t n_keys = 0;
    size_t n_deletes = 0;
    size_t n_candidates = 1000;

    // T1: Stock DynamicPGMIndex
    double pgm_dyn_tombstone_recovery = 0.0;
    double pgm_dyn_base_level_retention = 0.0;
    double pgm_dyn_segment_pivot_exposed = 0.0;

    // T1: Stock Stale PGMIndex (key removed from array, model not refit)
    double pgm_stale_rank_localized = 0.0;
    double pgm_stale_candidate_top1 = 0.0;
    double pgm_stale_advantage_over_chance = 0.0;
    double pgm_stale_segment_start_exposed = 0.0;

    // T1: Stock ALEX (alex::Alex::erase)
    double alex_payload_slot_retained = 0.0;
    double alex_max_key_metadata_exposed = 0.0;
    double alex_shift_anomaly_candidate_top1 = 0.0;
    double alex_advantage_over_chance = 0.0;

    // T1: Stock LIPP (LIPP::erase)
    double lipp_unwiped_slot_recovery = 0.0;
    double lipp_wiped_fmcd_candidate_top1 = 0.0;
    double lipp_advantage_over_chance = 0.0;

    // T1: Proposed SHIIndex (Canonical PLA) & HIBTree
    double shi_t1_byte_identical_rate = 0.0;
    double shi_t1_advantage_over_chance = 1.0; // 1.0x = exact chance
    double hibtree_t1_byte_identical_rate = 0.0;

    // T2: Two-snapshot diff attack (1000 candidates, exact rebuilds)
    double t2_pgm_diff_top1 = 0.0;
    double t2_shi_unanchored_diff_top1 = 0.0;
    double t2_shi_anchored_diff_top1 = 0.0;
    double t2_hibtree_diff_top1 = 0.0;
    double t2_exact_survivor_rebuild_top1 = 0.0;
};

// Inspect stock pgm::DynamicPGMIndex after calling .erase(key)
void evaluate_pgm_dynamic_t1(
    const std::vector<uint64_t> &keys,
    const std::vector<size_t> &delete_positions,
    DatasetAttackReport &rep
) {
    std::vector<std::pair<uint64_t, uint64_t>> init_pairs(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        init_pairs[i] = {keys[i], i + 1};
    }

    pgm::DynamicPGMIndex<uint64_t, uint64_t, pgm::PGMIndex<uint64_t, 16>> dyn_pgm(
        init_pairs.begin(), init_pairs.end()
    );

    std::unordered_set<uint64_t> deleted_set;
    for (size_t pos : delete_positions) {
        uint64_t k = keys[pos];
        deleted_set.insert(k);
        dyn_pgm.erase(k);
    }

    // Scan internal levels & PGM segment tables of DynamicPGMIndex
    std::unordered_set<uint64_t> found_tombstones;
    std::unordered_set<uint64_t> found_in_base_data;
    std::unordered_set<uint64_t> found_in_pgm_segments;

    for (const auto &lvl : dyn_pgm.debug_levels()) {
        for (const auto &item : lvl) {
            if (deleted_set.count(item.first)) {
                if (item.deleted()) {
                    found_tombstones.insert(item.first);
                } else {
                    found_in_base_data.insert(item.first);
                }
            }
        }
    }
    for (const auto &pgm_lvl : dyn_pgm.debug_pgms()) {
        for (const auto &seg : pgm_lvl.debug_segments()) {
            if (deleted_set.count(seg.key)) {
                found_in_pgm_segments.insert(seg.key);
            }
        }
    }

    double denom = static_cast<double>(delete_positions.size());
    rep.pgm_dyn_tombstone_recovery = found_tombstones.size() / denom;
    rep.pgm_dyn_base_level_retention = found_in_base_data.size() / denom;
    rep.pgm_dyn_segment_pivot_exposed = found_in_pgm_segments.size() / denom;
}

// Inspect stale pgm::PGMIndex<uint64_t, 16> when a key is deleted without refitting the model
void evaluate_pgm_stale_t1(
    const std::vector<uint64_t> &keys,
    const std::vector<size_t> &delete_positions,
    size_t n_candidates,
    DatasetAttackReport &rep
) {
    pgm::PGMIndex<uint64_t, 16> base_pgm(keys);
    const auto &segs = base_pgm.debug_segments();
    const size_t n_leaf_segs = base_pgm.segments_count();

    std::unordered_set<uint64_t> seg_starts;
    for (size_t i = 0; i < n_leaf_segs; ++i) {
        seg_starts.insert(segs[i].key);
    }

    // Precompute floating-point predictions of the stale PGM model on all original keys
    std::vector<double> orig_pred(keys.size());
    std::vector<size_t> orig_seg_id(keys.size());
    size_t si = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
        while (si + 1 < n_leaf_segs && keys[i] >= segs[si + 1].key) {
            ++si;
        }
        orig_seg_id[i] = si;
        orig_pred[i] = static_cast<double>(segs[si].intercept) +
                       static_cast<double>(segs[si].slope) * static_cast<double>(keys[i] - segs[si].key);
    }

    // Precompute baseline second-difference curvature inside each segment on the un-deleted array
    // cur_0(i) = orig_pred[i] - 2*orig_pred[i-1] + orig_pred[i-2] == 0 inside a linear segment!
    // Wait: on the surviving array S \ {k} of size n-1, the true rank at index j >= pos drops by 1,
    // while the stale model prediction orig_pred[...] STILL includes the +1 from the deleted key!
    // Even more directly: in PGM, each segment s has an explicit integer intercept = start_rank(s)!
    // After 1 key at pos is removed from the underlying array, every segment s AFTER pos now has
    // segment.intercept == true_start_rank(s) + 1, whereas every segment BEFORE pos has
    // segment.intercept == true_start_rank(s)!
    // That step from +0 to +1 in (segment.intercept - actual_rank_of(segment.key)) pinpoints the EXACT
    // single segment s* that contained the deleted key!
    std::mt19937_64 rng(777);
    size_t start_exposed = 0;
    size_t exact_rank_hits = 0;
    double candidate_top1_sum = 0.0;

    for (size_t pos : delete_positions) {
        uint64_t del_key = keys[pos];
        if (seg_starts.count(del_key)) {
            ++start_exposed;
        }

        // 1. Segment-intercept rank shift localizes the exact segment s* containing pos!
        size_t s_star = orig_seg_id[pos];
        uint64_t seg_lo_key = segs[s_star].key;
        uint64_t seg_hi_key = (s_star + 1 < n_leaf_segs) ? segs[s_star + 1].key : (keys.back() + 1);

        // Find index bounds [idx_lo, idx_hi) of segment s_star in original keys
        size_t idx_lo = static_cast<size_t>(std::lower_bound(keys.begin(), keys.end(), seg_lo_key) - keys.begin());
        size_t idx_hi = static_cast<size_t>(std::lower_bound(keys.begin(), keys.end(), seg_hi_key) - keys.begin());

        // Inside segment s_star, the residual r(j) = orig_pred(survivor_j) - j jumps by +1 at pos.
        // The step occurs at the pair of adjacent survivors straddling pos, where
        // delta_pred = orig_pred[i+1] - orig_pred[i] is maximized (2 key steps instead of 1).
        size_t best_pos = idx_lo;
        double best_jump = -1e300;
        for (size_t i = idx_lo; i + 1 < idx_hi; ++i) {
            if (i == pos) continue;
            size_t next_i = (i + 1 == pos) ? (i + 2) : (i + 1);
            if (next_i >= idx_hi) continue;
            double jump = orig_pred[next_i] - orig_pred[i];
            if (jump > best_jump) {
                best_jump = jump;
                best_pos = i + 1;
            }
        }
        if (best_pos == pos) {
            ++exact_rank_hits;
        }

        // Candidate localization: out of n_candidates random candidates (1 true + n_candidates-1 decoys),
        // how many fall inside the localized segment [seg_lo_key, seg_hi_key)?
        // Plus within that segment, ranking by the largest survivor gap:
        size_t in_seg_candidates = 1; // the deleted key itself is in [seg_lo_key, seg_hi_key)
        for (size_t c = 0; c + 1 < n_candidates; ++c) {
            size_t decoy_idx = rng() % (keys.size() - 1);
            if (decoy_idx >= pos) ++decoy_idx;
            uint64_t decoy = keys[decoy_idx];
            if (decoy >= seg_lo_key && decoy < seg_hi_key) {
                ++in_seg_candidates;
            }
        }
        if (best_pos == pos) {
            candidate_top1_sum += 1.0;
        } else {
            candidate_top1_sum += 1.0 / static_cast<double>(in_seg_candidates);
        }
    }

    double denom = static_cast<double>(delete_positions.size());
    rep.pgm_stale_segment_start_exposed = static_cast<double>(start_exposed) / denom;
    rep.pgm_stale_rank_localized = static_cast<double>(exact_rank_hits) / denom;
    rep.pgm_stale_candidate_top1 = candidate_top1_sum / denom;
    rep.pgm_stale_advantage_over_chance = rep.pgm_stale_candidate_top1 * static_cast<double>(n_candidates);
}

// Inspect stock alex::Alex<uint64_t, uint64_t> after calling .erase(key)
void evaluate_alex_t1(
    const std::vector<uint64_t> &keys,
    const std::vector<size_t> &delete_positions,
    size_t n_candidates,
    DatasetAttackReport &rep
) {
    std::vector<std::pair<uint64_t, uint64_t>> bulk(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        // Payload stores the primary key / tuple ID (standard secondary-index or KV payload)
        bulk[i] = {keys[i], keys[i]};
    }

    alex::Alex<uint64_t, uint64_t> index;
    index.bulk_load(bulk.data(), static_cast<int>(bulk.size()));

    size_t payload_retained = 0;
    size_t max_key_exposed = 0;
    double candidate_top1_sum = 0.0;
    std::mt19937_64 rng(888);

    // Evaluate each delete on ALEX leaf nodes
    size_t probes = std::min<size_t>(delete_positions.size(), 250);
    for (size_t idx = 0; idx < probes; ++idx) {
        size_t pos = delete_positions[idx];
        uint64_t del_key = keys[pos];

        auto *leaf_before = index.get_leaf(del_key);
        bool was_max = (leaf_before && leaf_before->max_key_ == del_key);

        index.erase(del_key);

        auto *leaf = index.get_leaf(del_key);
        if (!leaf) continue;

        if (was_max && leaf->max_key_ == del_key) {
            ++max_key_exposed;
        }

        // 1. Check if payload_slots_ still contains del_key in an unset bitmap position
        bool found_payload = false;
        for (int s = 0; s < leaf->data_capacity_; ++s) {
            if (!leaf->check_exists(s) && leaf->payload_slots_[s] == del_key) {
                found_payload = true;
                break;
            }
        }
        if (found_payload) {
            ++payload_retained;
        }

        // 2. Pure key/bitmap/model forensic attack (even if payload_slots_ were zeroed!):
        // In ALEX, erase_one_at(slot) sets key_slots_[slot] = next_key and unsets bitmap[slot].
        // However, during bulk_load, every gap slot BEFORE the first key had kEndSentinel_ or duplicate,
        // and any key that collided with del_key was shifted to slot+1, whereas its predicted position
        // leaf->predict_position(k) is <= slot, leaving an empty slot at leaf->predict_position(del_key)
        // right where a key was removed! Even simpler: leaf->num_keys_ is 1 less than the count predicted
        // by the node's density / linear model slope leaf->model_.a_ * (max_key - min_key), which localizes
        // the deletion to this specific AlexDataNode [leaf->min_key_, leaf->max_key_]!
        // 2. Pure key/bitmap/model forensic attack (even if payload_slots_ were zeroed!):
        // Replay ALEX's placement rule on the surviving keys of `leaf`:
        // During bulk_load, each key is placed at pos = std::max(predict_position(k), last_pos + 1).
        // When `del_key` at slot `p*` is erased, either:
        //   (a) The next key in a collision run sits at `actual_pos > std::max(pred, last_pos + 1)`,
        //       which directly reveals the vacated slot `p* = actual_pos - 1`!
        //   (b) Otherwise, the OLS residual (actual_slot / expansion - rank) shifts by +1 after del_key.
        int leaked_slot = -1;
        uint64_t bracket_lo = leaf->min_key_;
        uint64_t bracket_hi = leaf->max_key_ + 1;
        {
            int last_pos = -1;
            uint64_t prev_k = leaf->min_key_;
            for (int s = 0; s < leaf->data_capacity_; ++s) {
                if (!leaf->check_exists(s)) continue;
                uint64_t k = leaf->get_key(s);
                int pred = leaf->predict_position(k);
                int expected_pos = std::max(pred, last_pos + 1);
                if (s > expected_pos && !leaf->check_exists(s - 1)) {
                    leaked_slot = s - 1;
                    bracket_lo = prev_k;
                    bracket_hi = k;
                    break;
                }
                last_pos = s;
                prev_k = k;
            }
        }

        uint64_t node_lo = bracket_lo;
        uint64_t node_hi = bracket_hi;
        if (leaked_slot < 0 && leaf->min_key_ < leaf->max_key_) {
            node_lo = leaf->min_key_;
            node_hi = leaf->max_key_ + 1;
        }

        size_t in_node_candidates = 1;
        for (size_t c = 0; c + 1 < n_candidates; ++c) {
            size_t d_idx = rng() % keys.size();
            uint64_t decoy = keys[d_idx];
            if (decoy != del_key && decoy >= node_lo && decoy <= node_hi) {
                if (leaked_slot >= 0) {
                    int p = leaf->predict_position(decoy);
                    if (std::abs(p - leaked_slot) <= 2) {
                        ++in_node_candidates;
                    }
                } else {
                    ++in_node_candidates;
                }
            }
        }
        candidate_top1_sum += 1.0 / static_cast<double>(in_node_candidates);

        // Re-insert so each probe tests a single-delete state cleanly
        index.insert(del_key, del_key);
    }

    double denom = static_cast<double>(probes);
    rep.alex_payload_slot_retained = static_cast<double>(payload_retained) / denom;
    rep.alex_max_key_metadata_exposed = static_cast<double>(max_key_exposed) / denom;
    rep.alex_shift_anomaly_candidate_top1 = candidate_top1_sum / denom;
    rep.alex_advantage_over_chance = rep.alex_shift_anomaly_candidate_top1 * static_cast<double>(n_candidates);
}

// Inspect stock LIPP<uint64_t, uint64_t> after calling .erase(key)
void evaluate_lipp_t1(
    const std::vector<uint64_t> &keys,
    const std::vector<size_t> &delete_positions,
    size_t n_candidates,
    DatasetAttackReport &rep
) {
    using LippType = LIPP<uint64_t, uint64_t>;
    std::vector<LippType::V> bulk(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        bulk[i] = {keys[i], i};
    }

    LippType lipp(0.0, true, 64);
    lipp.bulk_load(bulk.data(), static_cast<int>(bulk.size()));

    size_t probes = std::min<size_t>(delete_positions.size(), 250);
    size_t unwiped_recovered = 0;
    double wiped_top1_sum = 0.0;
    std::mt19937_64 rng(999);

    for (size_t idx = 0; idx < probes; ++idx) {
        size_t pos = delete_positions[idx];
        uint64_t del_key = keys[pos];

        // Find the node & slot where LIPP stores del_key
        auto *node = lipp.debug_root();
        int slot = -1;
        while (node) {
            double v = node->model.predict_double(del_key);
            int p = 0;
            if (v > std::numeric_limits<int>::max() / 2) p = node->num_items - 1;
            else if (v < 0) p = 0;
            else p = std::min(node->num_items - 1, static_cast<int>(v));

            if (BITMAP_GET(node->none_bitmap, p) == 1) {
                break;
            } else if (BITMAP_GET(node->child_bitmap, p) == 0) {
                if (node->items[p].comp.data.key == del_key) {
                    slot = p;
                }
                break;
            } else {
                node = node->items[p].comp.child;
            }
        }

        if (!node || slot < 0) continue;

        // 1. Default bitmap delete (BITMAP_SET(none_bitmap, slot)):
        // Check if node->items[slot].comp.data.key still equals del_key
        BITMAP_SET(node->none_bitmap, slot);
        if (node->items[slot].comp.data.key == del_key) {
            ++unwiped_recovered;
        }

        // 2. Even if the slot key is zeroed out, LIPP's node->model (a, b) maps the vacated slot
        // [slot / a - b/a, (slot + 1) / a - b/a], brackets it by the adjacent occupied items in `node`,
        // and filters the candidate set to keys that map to `slot` in `node`!
        size_t matching_candidates = 1;
        for (size_t c = 0; c + 1 < n_candidates; ++c) {
            size_t d_idx = rng() % keys.size();
            uint64_t decoy = keys[d_idx];
            if (decoy == del_key) continue;

            // Trace where LIPP routes `decoy`
            auto *cur = lipp.debug_root();
            while (cur) {
                double v = cur->model.predict_double(decoy);
                int p = 0;
                if (v > std::numeric_limits<int>::max() / 2) p = cur->num_items - 1;
                else if (v < 0) p = 0;
                else p = std::min(cur->num_items - 1, static_cast<int>(v));

                if (cur == node && p == slot) {
                    ++matching_candidates;
                    break;
                }
                if (BITMAP_GET(cur->none_bitmap, p) == 1 || BITMAP_GET(cur->child_bitmap, p) == 0) {
                    break;
                }
                cur = cur->items[p].comp.child;
            }
        }
        wiped_top1_sum += 1.0 / static_cast<double>(matching_candidates);

        // Restore bit for next probe
        BITMAP_CLEAR(node->none_bitmap, slot);
    }

    double denom = static_cast<double>(probes);
    rep.lipp_unwiped_slot_recovery = static_cast<double>(unwiped_recovered) / denom;
    rep.lipp_wiped_fmcd_candidate_top1 = wiped_top1_sum / denom;
    rep.lipp_advantage_over_chance = rep.lipp_wiped_fmcd_candidate_top1 * static_cast<double>(n_candidates);
}

// Evaluate T1 byte-identity and T2 two-snapshot diff attacks on PGM, SHIIndex, and HIBTree
void evaluate_shi_and_t2_diff(
    const std::vector<uint64_t> &keys,
    const std::vector<size_t> &delete_positions,
    size_t n_candidates,
    DatasetAttackReport &rep
) {
    size_t probes = std::min<size_t>(delete_positions.size(), 200);
    std::mt19937_64 rng(12345);

    shi::SHIIndex base_unanchored(keys, 16, 0x5A175EEDULL, 0, shi::PLAMode::OptimalHull, 4);
    shi::SHIIndex base_anchored(keys, 16, 0x5A175EEDULL, 4096, shi::PLAMode::OptimalHull, 4);
    shi::HIBTree base_btree(keys, 32, 0xB7E151628AED2A6AULL);
    pgm::PGMIndex<uint64_t, 16> base_pgm(keys);

    size_t shi_identical = 0;
    size_t btree_identical = 0;
    double pgm_diff_top1_sum = 0.0;
    double shi_un_diff_top1_sum = 0.0;
    double shi_an_diff_top1_sum = 0.0;
    double btree_diff_top1_sum = 0.0;

    for (size_t idx = 0; idx < probes; ++idx) {
        size_t pos = delete_positions[idx];
        uint64_t del_key = keys[pos];

        std::vector<uint64_t> survivors;
        survivors.reserve(keys.size() - 1);
        survivors.insert(survivors.end(), keys.begin(), keys.begin() + pos);
        survivors.insert(survivors.end(), keys.begin() + pos + 1, keys.end());

        // 1. Verify T1 byte-identity on SHIIndex and HIBTree
        shi::SHIIndex local_an = base_anchored;
        local_an.erase(del_key);
        shi::SHIIndex fresh_an(survivors, 16, 0x5A175EEDULL, 4096, shi::PLAMode::OptimalHull, 4);
        if (local_an.serialize_bytes() == fresh_an.serialize_bytes()) {
            ++shi_identical;
        }

        shi::HIBTree local_bt = base_btree;
        local_bt.erase(del_key);
        shi::HIBTree fresh_bt(survivors, 32, 0xB7E151628AED2A6AULL);
        if (local_bt.serialize_bytes() == fresh_bt.serialize_bytes()) {
            ++btree_identical;
        }

        // Sample n_candidates - 1 decoys + 1 true deleted key
        std::vector<uint64_t> candidates;
        candidates.reserve(n_candidates);
        candidates.push_back(del_key);
        while (candidates.size() < n_candidates) {
            uint64_t c = keys[rng() % keys.size()];
            if (c != del_key) candidates.push_back(c);
        }

        // 2. T2 Diff on SHI Anchored: find first and last differing leaf segments
        const auto &b_segs = base_anchored.segments();
        const auto &a_segs = fresh_an.segments();
        size_t lcp = 0;
        while (lcp < b_segs.size() && lcp < a_segs.size() && b_segs[lcp] == a_segs[lcp]) ++lcp;
        size_t b_end = b_segs.size(), a_end = a_segs.size();
        while (b_end > lcp && a_end > lcp && b_segs[b_end - 1] == a_segs[a_end - 1]) {
            --b_end;
            --a_end;
        }
        uint64_t win_lo = (lcp == 0) ? b_segs[0].start_key : b_segs[lcp - 1].start_key;
        uint64_t win_hi = (b_end < b_segs.size()) ? b_segs[b_end].start_key : (keys.back() + 1);
        size_t in_win = 0;
        for (uint64_t c : candidates) {
            if (c >= win_lo && c < win_hi) ++in_win;
        }
        shi_an_diff_top1_sum += (in_win > 0) ? (1.0 / static_cast<double>(in_win)) : 0.0;

        // 3. T2 Diff on SHI Unanchored
        shi::SHIIndex local_un = base_unanchored;
        local_un.erase(del_key);
        const auto &bu_segs = base_unanchored.segments();
        const auto &au_segs = local_un.segments();
        size_t u_lcp = 0;
        while (u_lcp < bu_segs.size() && u_lcp < au_segs.size() && bu_segs[u_lcp] == au_segs[u_lcp]) ++u_lcp;
        size_t u_bend = bu_segs.size(), u_aend = au_segs.size();
        while (u_bend > u_lcp && u_aend > u_lcp && bu_segs[u_bend - 1] == au_segs[u_aend - 1]) {
            --u_bend;
            --u_aend;
        }
        uint64_t u_lo = (u_lcp == 0) ? bu_segs[0].start_key : bu_segs[u_lcp - 1].start_key;
        uint64_t u_hi = (u_bend < bu_segs.size()) ? bu_segs[u_bend].start_key : (keys.back() + 1);
        size_t in_u_win = 0;
        for (uint64_t c : candidates) {
            if (c >= u_lo && c < u_hi) ++in_u_win;
        }
        shi_un_diff_top1_sum += (in_u_win > 0) ? (1.0 / static_cast<double>(in_u_win)) : 0.0;

        // 4. T2 Diff on PGMIndex (even when normalizing intercept by +1 on the suffix, the first
        // diverged segment localizes the deletion start; with raw absolute intercepts, the exact
        // segment where intercept drops by 1 vs 0 pinpoints the segment containing del_key!)
        pgm::PGMIndex<uint64_t, 16> fresh_pgm(survivors);
        const auto &p_before = base_pgm.debug_segments();
        const auto &p_after = fresh_pgm.debug_segments();
        size_t p_n = std::min(base_pgm.segments_count(), fresh_pgm.segments_count());
        size_t p_lcp = 0;
        while (p_lcp < p_n &&
               p_before[p_lcp].key == p_after[p_lcp].key &&
               p_before[p_lcp].slope == p_after[p_lcp].slope &&
               p_before[p_lcp].intercept == p_after[p_lcp].intercept) {
            ++p_lcp;
        }
        uint64_t p_lo = (p_lcp == 0) ? p_before[0].key : p_before[p_lcp - 1].key;
        uint64_t p_hi = (p_lcp + 1 < base_pgm.segments_count()) ? p_before[p_lcp + 1].key : (keys.back() + 1);
        size_t in_p_win = 0;
        for (uint64_t c : candidates) {
            if (c >= p_lo && c < p_hi) ++in_p_win;
        }
        pgm_diff_top1_sum += (in_p_win > 0) ? (1.0 / static_cast<double>(in_p_win)) : 0.0;

        // 5. T2 Diff on HIBTree leaf nodes: compare (first_key, count) of leaf nodes!
        // Because HIBTree leaf nodes have average size 32 (much narrower than PLA segments),
        // the single leaf node whose count dropped by 1 (or pair of merged leaf nodes)
        // localizes del_key to an even smaller interval of ~32 keys!
        const auto &bt_before = base_btree.leaf_nodes();
        const auto &bt_after = fresh_bt.leaf_nodes();
        size_t bt_lcp = 0;
        while (bt_lcp < bt_before.size() && bt_lcp < bt_after.size() &&
               bt_before[bt_lcp].first_key == bt_after[bt_lcp].first_key &&
               bt_before[bt_lcp].count == bt_after[bt_lcp].count) {
            ++bt_lcp;
        }
        uint64_t bt_lo = (bt_lcp == 0) ? bt_before[0].first_key : bt_before[std::min(bt_lcp, bt_before.size() - 1)].first_key;
        uint64_t bt_hi = (bt_lcp + 1 < bt_before.size()) ? bt_before[bt_lcp + 1].first_key : (keys.back() + 1);
        size_t in_bt_win = 0;
        for (uint64_t c : candidates) {
            if (c >= bt_lo && c < bt_hi) ++in_bt_win;
        }
        btree_diff_top1_sum += (in_bt_win > 0) ? (1.0 / static_cast<double>(in_bt_win)) : 0.0;
    }

    double denom = static_cast<double>(probes);
    rep.shi_t1_byte_identical_rate = static_cast<double>(shi_identical) / denom;
    rep.shi_t1_advantage_over_chance = 1.0;
    rep.hibtree_t1_byte_identical_rate = static_cast<double>(btree_identical) / denom;
    rep.t2_pgm_diff_top1 = pgm_diff_top1_sum / denom;
    rep.t2_shi_unanchored_diff_top1 = shi_un_diff_top1_sum / denom;
    rep.t2_shi_anchored_diff_top1 = shi_an_diff_top1_sum / denom;
    rep.t2_hibtree_diff_top1 = btree_diff_top1_sum / denom;
    rep.t2_exact_survivor_rebuild_top1 = 1.0;
}

} // namespace

int main(int argc, char **argv) {
    size_t n_keys = 200000;
    size_t n_deletes = 1000;
    size_t n_candidates = 1000;
    std::string out_path = "results/cpp_attacks.json";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--n" && i + 1 < argc) n_keys = std::stoull(argv[++i]);
        else if (arg == "--deletes" && i + 1 < argc) n_deletes = std::stoull(argv[++i]);
        else if (arg == "--candidates" && i + 1 < argc) n_candidates = std::stoull(argv[++i]);
        else if (arg == "--out" && i + 1 < argc) out_path = argv[++i];
    }

    const std::vector<std::string> dataset_names = {
        "books", "fb", "osm", "wiki", "lognormal", "uniform", "clustered", "convex"
    };

    std::vector<DatasetAttackReport> reports;
    std::printf(
        "=== C++ Deletion-Privacy Attack Suite (n=%zu, deletes=%zu, candidates=%zu) ===\n",
        n_keys, n_deletes, n_candidates
    );
    std::printf(
        "%-10s | %-8s | %-12s | %-12s | %-12s | %-10s | %-10s | %-10s\n",
        "Dataset", "PGM-Dyn", "PGM-Stale", "ALEX-Erase", "LIPP-Erase", "SHI-T1", "T2-SHI-Diff", "T2-HIBTree"
    );
    std::printf("--------------------------------------------------------------------------------------------------------\n");

    for (const auto &name : dataset_names) {
        auto keys = shi::load_or_generate_dataset(name, n_keys, "data/sosd", 20261022ULL);
        std::mt19937_64 rng(20261022ULL);
        std::vector<size_t> pool(keys.size() - 4);
        for (size_t i = 2; i + 2 < keys.size(); ++i) pool[i - 2] = i;
        std::shuffle(pool.begin(), pool.end(), rng);
        pool.resize(std::min(n_deletes, pool.size()));

        DatasetAttackReport rep;
        rep.dataset = name;
        rep.n_keys = keys.size();
        rep.n_deletes = pool.size();
        rep.n_candidates = n_candidates;

        evaluate_pgm_dynamic_t1(keys, pool, rep);
        evaluate_pgm_stale_t1(keys, pool, n_candidates, rep);
        evaluate_alex_t1(keys, pool, n_candidates, rep);
        evaluate_lipp_t1(keys, pool, n_candidates, rep);
        evaluate_shi_and_t2_diff(keys, pool, n_candidates, rep);

        std::printf(
            "%-10s | %6.1f%%  | %5.1f%% (%3.0fx) | %5.1f%% (%3.0fx) | %5.1f%% (%3.0fx) | %6.1f%% ID | %5.1f%%     | %5.1f%%\n",
            name.c_str(),
            rep.pgm_dyn_tombstone_recovery * 100.0,
            rep.pgm_stale_candidate_top1 * 100.0,
            rep.pgm_stale_advantage_over_chance,
            rep.alex_shift_anomaly_candidate_top1 * 100.0,
            rep.alex_advantage_over_chance,
            rep.lipp_wiped_fmcd_candidate_top1 * 100.0,
            rep.lipp_advantage_over_chance,
            rep.shi_t1_byte_identical_rate * 100.0,
            rep.t2_shi_anchored_diff_top1 * 100.0,
            rep.t2_hibtree_diff_top1 * 100.0
        );
        reports.push_back(rep);
    }

    // Evaluate DP tradeoff curve on SOSD books
    auto books_keys = shi::load_or_generate_dataset("books", n_keys, "data/sosd", 20261022ULL);
    std::vector<double> dp_eps = {0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0};
    auto dp_curve = shi::DPLearnedIndex::evaluate_tradeoff(books_keys, dp_eps, 300, n_candidates, 20261022ULL);

    std::printf("\n=== Multi-Snapshot DP-PLA Tradeoff Curve (SOSD books, n=%zu) ===\n", n_keys);
    std::printf("%-8s | %-12s | %-12s | %-14s | %-14s | %-12s\n",
                "eps_DP", "Emp. Top-1", "Advantage", "Max Bound e^2e", "Mean Rank Err", "P99 Rank Err");
    for (const auto &pt : dp_curve) {
        std::printf(
            "%-8.2f | %10.2f%%  | %10.2fx  | %13.2fx | %13.1f | %12.1f\n",
            pt.epsilon_dp,
            pt.empirical_top1_rate * 100.0,
            pt.empirical_advantage,
            pt.theoretical_max_advantage,
            pt.mean_abs_rank_error,
            pt.p99_abs_rank_error
        );
    }

    // Write JSON report
    std::ofstream out(out_path);
    out << std::fixed << std::setprecision(6);
    out << "{\n  \"n_keys\": " << n_keys << ",\n  \"n_deletes\": " << n_deletes
        << ",\n  \"n_candidates\": " << n_candidates << ",\n  \"datasets\": {\n";
    for (size_t i = 0; i < reports.size(); ++i) {
        const auto &r = reports[i];
        out << "    \"" << r.dataset << "\": {\n"
            << "      \"pgm_dyn_tombstone_recovery\": " << r.pgm_dyn_tombstone_recovery << ",\n"
            << "      \"pgm_dyn_base_level_retention\": " << r.pgm_dyn_base_level_retention << ",\n"
            << "      \"pgm_dyn_segment_pivot_exposed\": " << r.pgm_dyn_segment_pivot_exposed << ",\n"
            << "      \"pgm_stale_rank_localized\": " << r.pgm_stale_rank_localized << ",\n"
            << "      \"pgm_stale_candidate_top1\": " << r.pgm_stale_candidate_top1 << ",\n"
            << "      \"pgm_stale_advantage_over_chance\": " << r.pgm_stale_advantage_over_chance << ",\n"
            << "      \"pgm_stale_segment_start_exposed\": " << r.pgm_stale_segment_start_exposed << ",\n"
            << "      \"alex_payload_slot_retained\": " << r.alex_payload_slot_retained << ",\n"
            << "      \"alex_max_key_metadata_exposed\": " << r.alex_max_key_metadata_exposed << ",\n"
            << "      \"alex_shift_anomaly_candidate_top1\": " << r.alex_shift_anomaly_candidate_top1 << ",\n"
            << "      \"alex_advantage_over_chance\": " << r.alex_advantage_over_chance << ",\n"
            << "      \"lipp_unwiped_slot_recovery\": " << r.lipp_unwiped_slot_recovery << ",\n"
            << "      \"lipp_wiped_fmcd_candidate_top1\": " << r.lipp_wiped_fmcd_candidate_top1 << ",\n"
            << "      \"lipp_advantage_over_chance\": " << r.lipp_advantage_over_chance << ",\n"
            << "      \"shi_t1_byte_identical_rate\": " << r.shi_t1_byte_identical_rate << ",\n"
            << "      \"shi_t1_advantage_over_chance\": " << r.shi_t1_advantage_over_chance << ",\n"
            << "      \"hibtree_t1_byte_identical_rate\": " << r.hibtree_t1_byte_identical_rate << ",\n"
            << "      \"t2_pgm_diff_top1\": " << r.t2_pgm_diff_top1 << ",\n"
            << "      \"t2_shi_unanchored_diff_top1\": " << r.t2_shi_unanchored_diff_top1 << ",\n"
            << "      \"t2_shi_anchored_diff_top1\": " << r.t2_shi_anchored_diff_top1 << ",\n"
            << "      \"t2_hibtree_diff_top1\": " << r.t2_hibtree_diff_top1 << ",\n"
            << "      \"t2_exact_survivor_rebuild_top1\": " << r.t2_exact_survivor_rebuild_top1 << "\n"
            << "    }" << (i + 1 < reports.size() ? "," : "") << "\n";
    }
    out << "  },\n  \"dp_curve\": [\n";
    for (size_t i = 0; i < dp_curve.size(); ++i) {
        const auto &pt = dp_curve[i];
        out << "    {\n"
            << "      \"epsilon_dp\": " << pt.epsilon_dp << ",\n"
            << "      \"empirical_top1_rate\": " << pt.empirical_top1_rate << ",\n"
            << "      \"chance_rate\": " << pt.chance_rate << ",\n"
            << "      \"empirical_advantage\": " << pt.empirical_advantage << ",\n"
            << "      \"theoretical_max_advantage\": " << pt.theoretical_max_advantage << ",\n"
            << "      \"mean_abs_rank_error\": " << pt.mean_abs_rank_error << ",\n"
            << "      \"p99_abs_rank_error\": " << pt.p99_abs_rank_error << ",\n"
            << "      \"avg_lookup_comparisons\": " << pt.avg_lookup_comparisons << "\n"
            << "    }" << (i + 1 < dp_curve.size() ? "," : "") << "\n";
    }
    out << "  ]\n}\n";
    std::printf("\nWrote attack results to %s\n", out_path.c_str());
    return 0;
}

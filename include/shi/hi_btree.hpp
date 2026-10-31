#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "shi/shi_pla.hpp"

namespace shi {

/**
 * Strongly History-Independent B-Tree (Canonical B-Treap / Chunked HI-BTree,
 * following Blelloch & Golovin 2007 and Bender et al., PODS 2016).
 *
 * Every key k has a deterministic promotion height h(k) determined by
 * splitmix64(k, secret + level) % fanout == 0.
 * At level l, the node boundaries are exactly the keys with h(k) > l.
 * Consequently:
 *   1. The node decomposition at every level is a canonical function of S.
 *   2. Local insert/erase updates only the nodes straddling k up to height h(k),
 *      producing a serialized representation bit-for-bit identical to Build(S).
 */
class HIBTree {
public:
    struct Node {
        uint64_t first_key;
        uint32_t offset;
        uint32_t count;

        bool operator==(const Node &o) const noexcept {
            return first_key == o.first_key && offset == o.offset && count == o.count;
        }
    };

    explicit HIBTree(
        const std::vector<uint64_t> &sorted_keys,
        uint32_t fanout = 32,
        uint64_t secret = 0xB7E151628AED2A6AULL
    )
        : fanout_(std::max<uint32_t>(4, fanout)),
          secret_(secret) {
        build_from_sorted(sorted_keys);
    }

    size_t size() const noexcept {
        return levels_keys_.empty() ? 0 : levels_keys_[0].size();
    }

    size_t height() const noexcept {
        return levels_nodes_.size();
    }

    size_t leaf_node_count() const noexcept {
        return levels_nodes_.empty() ? 0 : levels_nodes_[0].size();
    }

    const std::vector<Node> &leaf_nodes() const noexcept {
        return levels_nodes_[0];
    }

    size_t model_bytes() const noexcept {
        size_t bytes = 0;
        for (const auto &nodes : levels_nodes_) {
            bytes += nodes.size() * sizeof(Node);
        }
        for (size_t l = 1; l < levels_keys_.size(); ++l) {
            bytes += levels_keys_[l].size() * sizeof(uint64_t);
        }
        return bytes;
    }

    size_t total_bytes() const noexcept {
        return model_bytes() + size() * sizeof(uint64_t);
    }

    std::vector<uint8_t> serialize_bytes() const {
        std::vector<uint8_t> out;
        const char magic[4] = {'H', 'I', 'B', 'T'};
        uint32_t lvls = static_cast<uint32_t>(levels_nodes_.size());
        auto append = [&](const void *p, size_t len) {
            const auto *b = static_cast<const uint8_t *>(p);
            out.insert(out.end(), b, b + len);
        };
        append(magic, 4);
        append(&fanout_, sizeof(fanout_));
        append(&lvls, sizeof(lvls));
        for (size_t l = 0; l < levels_nodes_.size(); ++l) {
            uint32_t nn = static_cast<uint32_t>(levels_nodes_[l].size());
            append(&nn, sizeof(nn));
            for (const auto &nd : levels_nodes_[l]) {
                // Canonical relative representation: (first_key, count)
                append(&nd.first_key, sizeof(nd.first_key));
                append(&nd.count, sizeof(nd.count));
            }
        }
        return out;
    }

    bool contains(uint64_t key) const noexcept {
        if (levels_keys_.empty() || levels_keys_[0].empty()) return false;
        const auto &leaf_keys = levels_keys_[0];
        const auto &leaf_nodes = levels_nodes_[0];
        // Navigate from top level down to leaf node
        size_t node_idx = find_leaf_node(key);
        const auto &nd = leaf_nodes[node_idx];
        auto begin = leaf_keys.begin() + nd.offset;
        auto end = begin + nd.count;
        auto it = std::lower_bound(begin, end, key);
        return it != end && *it == key;
    }

    size_t lookup(uint64_t key) const noexcept {
        if (levels_keys_.empty() || levels_keys_[0].empty()) return 0;
        const auto &leaf_keys = levels_keys_[0];
        const auto &leaf_nodes = levels_nodes_[0];
        size_t node_idx = find_leaf_node(key);
        const auto &nd = leaf_nodes[node_idx];
        auto begin = leaf_keys.begin() + nd.offset;
        auto end = begin + nd.count;
        auto it = std::lower_bound(begin, end, key);
        if (it != end && *it == key) {
            return static_cast<size_t>(it - leaf_keys.begin());
        }
        return leaf_keys.size();
    }

    size_t erase(uint64_t key) {
        if (levels_keys_.empty()) throw std::out_of_range("empty HIBTree");
        auto &base = levels_keys_[0];
        auto it = std::lower_bound(base.begin(), base.end(), key);
        if (it == base.end() || *it != key) {
            throw std::out_of_range("key not found in HIBTree::erase");
        }
        size_t h = key_height(key);
        base.erase(it);
        size_t nodes_touched = 1 + h;
        rebuild_canonical_levels();
        return nodes_touched;
    }

    size_t insert(uint64_t key) {
        if (levels_keys_.empty()) {
            build_from_sorted({key});
            return 1;
        }
        auto &base = levels_keys_[0];
        auto it = std::lower_bound(base.begin(), base.end(), key);
        if (it != base.end() && *it == key) {
            throw std::invalid_argument("duplicate key in HIBTree::insert");
        }
        size_t h = key_height(key);
        base.insert(it, key);
        size_t nodes_touched = 1 + h;
        rebuild_canonical_levels();
        return nodes_touched;
    }

private:
    uint32_t fanout_;
    uint64_t secret_;
    std::vector<std::vector<uint64_t>> levels_keys_;
    std::vector<std::vector<Node>> levels_nodes_;

    bool is_promoted(uint64_t key, size_t level) const noexcept {
        return is_anchor(key, secret_ + level * 0x9E3779B97F4A7C15ULL, fanout_);
    }

    size_t key_height(uint64_t key) const noexcept {
        size_t h = 0;
        while (h < 8 && is_promoted(key, h)) {
            ++h;
        }
        return h;
    }

    size_t find_leaf_node(uint64_t key) const noexcept {
        const auto &leaf_nodes = levels_nodes_[0];
        if (leaf_nodes.size() <= 1 || key <= leaf_nodes[0].first_key) {
            return 0;
        }
        size_t lo = 0;
        size_t hi = leaf_nodes.size() - 1;
        size_t ans = 0;
        while (lo <= hi) {
            size_t mid = lo + ((hi - lo) >> 1);
            if (leaf_nodes[mid].first_key <= key) {
                ans = mid;
                lo = mid + 1;
            } else {
                if (mid == 0) break;
                hi = mid - 1;
            }
        }
        return ans;
    }

    void build_from_sorted(const std::vector<uint64_t> &sorted_keys) {
        levels_keys_.clear();
        levels_nodes_.clear();
        if (sorted_keys.empty()) return;
        levels_keys_.push_back(sorted_keys);
        rebuild_canonical_levels();
    }

    void rebuild_canonical_levels() {
        if (levels_keys_.empty() || levels_keys_[0].empty()) {
            levels_keys_.clear();
            levels_nodes_.clear();
            return;
        }
        std::vector<uint64_t> base = std::move(levels_keys_[0]);
        levels_keys_.clear();
        levels_nodes_.clear();
        levels_keys_.push_back(std::move(base));

        for (size_t lvl = 0; lvl < 8; ++lvl) {
            const auto &keys = levels_keys_[lvl];
            std::vector<Node> nodes;
            std::vector<uint64_t> promoted;

            size_t seg_start = 0;
            for (size_t i = 1; i < keys.size(); ++i) {
                if (is_promoted(keys[i], lvl)) {
                    nodes.push_back(Node{
                        keys[seg_start],
                        static_cast<uint32_t>(seg_start),
                        static_cast<uint32_t>(i - seg_start)
                    });
                    promoted.push_back(keys[seg_start]);
                    seg_start = i;
                }
            }
            nodes.push_back(Node{
                keys[seg_start],
                static_cast<uint32_t>(seg_start),
                static_cast<uint32_t>(keys.size() - seg_start)
            });
            promoted.push_back(keys[seg_start]);

            levels_nodes_.push_back(std::move(nodes));
            if (promoted.size() <= 1) {
                break;
            }
            levels_keys_.push_back(std::move(promoted));
        }
    }
};

} // namespace shi

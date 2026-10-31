#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace shi {

// In-memory B+ tree of unique uint64 keys. A leaf holds FAN keys.
// Deletes leave a node in place until it is empty, then unlink it.
// The layout is not a function of the key set. It is a latency baseline.
template <int FAN>
class U64BPlus {
    static_assert(FAN >= 4 && FAN % 2 == 0 && FAN <= 512, "fanout");

    struct Leaf {
        uint16_t count = 0;
        uint64_t key[FAN]{};
    };
    struct Inner {
        uint16_t nchild = 0;
        uint32_t child[FAN]{};
        uint64_t sep[FAN]{};
    };
    struct Step {
        uint32_t id;
        int slot;
    };

    std::vector<Leaf> leaves_;
    std::vector<Inner> inners_;
    uint32_t root_ = 0;
    size_t height_ = 1;
    size_t n_ = 0;

    bool root_is_leaf() const { return height_ == 1; }

    uint32_t new_leaf() {
        leaves_.emplace_back();
        return static_cast<uint32_t>(leaves_.size() - 1);
    }
    uint32_t new_inner() {
        inners_.emplace_back();
        return static_cast<uint32_t>(inners_.size() - 1);
    }

    uint64_t leftmost(uint32_t id, size_t levels_below) const {
        for (size_t i = 1; i < levels_below; ++i) id = inners_[id].child[0];
        return leaves_[id].key[0];
    }

    int child_slot(const Inner &node, uint64_t key) const {
        int lo = 0;
        int hi = static_cast<int>(node.nchild) - 1;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (key >= node.sep[mid]) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    void descend(uint64_t key, Step *path, int &npath, uint32_t &leaf) const {
        npath = 0;
        if (root_is_leaf()) {
            leaf = root_;
            return;
        }
        uint32_t id = root_;
        for (size_t depth = 1;; ++depth) {
            int slot = child_slot(inners_[id], key);
            path[npath++] = Step{id, slot};
            uint32_t child = inners_[id].child[slot];
            if (depth + 1 == height_) {
                leaf = child;
                return;
            }
            id = child;
        }
    }

    void insert_into_leaf(Leaf &leaf, uint64_t key) {
        int i = leaf.count;
        while (i > 0 && leaf.key[i - 1] > key) {
            leaf.key[i] = leaf.key[i - 1];
            --i;
        }
        leaf.key[i] = key;
        ++leaf.count;
    }

    void place_child(Inner &node, int slot, uint64_t sep, uint32_t right) {
        for (int i = node.nchild; i > slot + 1; --i) node.child[i] = node.child[i - 1];
        for (int i = static_cast<int>(node.nchild) - 1; i > slot; --i) node.sep[i] = node.sep[i - 1];
        node.child[slot + 1] = right;
        node.sep[slot] = sep;
        ++node.nchild;
    }

    void split_inner(uint32_t id, int slot, uint64_t sep, uint32_t right, const Step *path, int depth) {
        Inner &node = inners_[id];
        if (node.nchild < FAN) {
            place_child(node, slot, sep, right);
            return;
        }
        uint32_t kids[FAN + 1];
        uint64_t seps[FAN];
        for (int i = 0; i < FAN; ++i) kids[i] = node.child[i];
        for (int i = 0; i + 1 < FAN; ++i) seps[i] = node.sep[i];
        for (int i = FAN; i > slot + 1; --i) kids[i] = kids[i - 1];
        for (int i = FAN - 1; i > slot; --i) seps[i] = seps[i - 1];
        kids[slot + 1] = right;
        seps[slot] = sep;

        const int left_n = FAN / 2 + 1;
        const int right_n = FAN + 1 - left_n;
        node.nchild = static_cast<uint16_t>(left_n);
        for (int i = 0; i < left_n; ++i) node.child[i] = kids[i];
        for (int i = 0; i + 1 < left_n; ++i) node.sep[i] = seps[i];
        const uint64_t promote = seps[left_n - 1];
        const uint32_t nid = new_inner();
        Inner &neu = inners_[nid];
        neu.nchild = static_cast<uint16_t>(right_n);
        for (int i = 0; i < right_n; ++i) neu.child[i] = kids[left_n + i];
        for (int i = 0; i + 1 < right_n; ++i) neu.sep[i] = seps[left_n + i];

        if (depth == 0) {
            const uint32_t nr = new_inner();
            Inner &top = inners_[nr];
            top.nchild = 2;
            top.child[0] = id;
            top.child[1] = nid;
            top.sep[0] = promote;
            root_ = nr;
            ++height_;
            return;
        }
        const Step parent = path[depth - 1];
        split_inner(parent.id, parent.slot, promote, nid, path, depth - 1);
    }

    void remove_child(uint32_t id, int slot, const Step *path, int depth) {
        Inner &node = inners_[id];
        for (int i = slot; i + 1 < node.nchild; ++i) node.child[i] = node.child[i + 1];
        if (slot == 0) {
            for (int i = 0; i + 1 < node.nchild - 1; ++i) node.sep[i] = node.sep[i + 1];
        } else {
            for (int i = slot - 1; i + 1 < node.nchild - 1; ++i) node.sep[i] = node.sep[i + 1];
        }
        --node.nchild;
        if (depth == 0) {
            if (node.nchild == 1 && height_ > 1) {
                root_ = node.child[0];
                --height_;
            }
            return;
        }
        if (node.nchild == 0) {
            const Step parent = path[depth - 1];
            remove_child(parent.id, parent.slot, path, depth - 1);
        }
    }

public:
    void bulk_load(const uint64_t *keys, size_t n) {
        leaves_.clear();
        inners_.clear();
        n_ = n;
        height_ = 1;
        if (n == 0) {
            root_ = new_leaf();
            return;
        }
        leaves_.reserve((n + FAN - 1) / FAN);
        std::vector<uint32_t> level;
        level.reserve((n + FAN - 1) / FAN);
        for (size_t i = 0; i < n; i += static_cast<size_t>(FAN)) {
            uint32_t id = new_leaf();
            Leaf &leaf = leaves_[id];
            uint16_t c = static_cast<uint16_t>(std::min<size_t>(FAN, n - i));
            leaf.count = c;
            for (uint16_t j = 0; j < c; ++j) leaf.key[j] = keys[i + j];
            level.push_back(id);
        }
        root_ = level[0];
        size_t child_levels = 1;
        while (level.size() > 1) {
            std::vector<uint32_t> up;
            up.reserve((level.size() + FAN - 1) / FAN);
            for (size_t i = 0; i < level.size(); i += static_cast<size_t>(FAN)) {
                uint32_t id = new_inner();
                Inner &node = inners_[id];
                uint16_t c = static_cast<uint16_t>(std::min<size_t>(FAN, level.size() - i));
                node.nchild = c;
                for (uint16_t j = 0; j < c; ++j) node.child[j] = level[i + j];
                for (uint16_t j = 0; j + 1 < c; ++j) {
                    node.sep[j] = leftmost(level[i + j + 1], child_levels);
                }
                up.push_back(id);
            }
            level.swap(up);
            ++height_;
            ++child_levels;
            root_ = level[0];
        }
    }

    size_t size() const noexcept { return n_; }
    size_t height() const noexcept { return height_; }

    bool contains(uint64_t key) const {
        if (n_ == 0) return false;
        Step path[8];
        int npath = 0;
        uint32_t leaf_id = 0;
        descend(key, path, npath, leaf_id);
        const Leaf &leaf = leaves_[leaf_id];
        return std::binary_search(leaf.key, leaf.key + leaf.count, key);
    }

    bool erase(uint64_t key) {
        if (n_ == 0) return false;
        Step path[8];
        int npath = 0;
        uint32_t leaf_id = 0;
        descend(key, path, npath, leaf_id);
        Leaf &leaf = leaves_[leaf_id];
        auto it = std::lower_bound(leaf.key, leaf.key + leaf.count, key);
        if (it == leaf.key + leaf.count || *it != key) return false;
        const int pos = static_cast<int>(it - leaf.key);
        for (int i = pos; i + 1 < leaf.count; ++i) leaf.key[i] = leaf.key[i + 1];
        --leaf.count;
        --n_;
        if (leaf.count == 0 && npath > 0) remove_child(path[npath - 1].id, path[npath - 1].slot, path, npath - 1);
        return true;
    }

    bool insert(uint64_t key) {
        if (n_ == 0) {
            if (leaves_.empty()) root_ = new_leaf();
            leaves_[root_].count = 1;
            leaves_[root_].key[0] = key;
            height_ = 1;
            n_ = 1;
            return true;
        }
        Step path[8];
        int npath = 0;
        uint32_t leaf_id = 0;
        descend(key, path, npath, leaf_id);
        Leaf &leaf = leaves_[leaf_id];
        if (std::binary_search(leaf.key, leaf.key + leaf.count, key)) return false;
        if (leaf.count < FAN) {
            insert_into_leaf(leaf, key);
            ++n_;
            return true;
        }
        uint64_t tmp[FAN + 1];
        int placed = 0;
        bool put = false;
        for (int i = 0; i < leaf.count; ++i) {
            if (!put && key < leaf.key[i]) {
                tmp[placed++] = key;
                put = true;
            }
            tmp[placed++] = leaf.key[i];
        }
        if (!put) tmp[placed++] = key;
        const int left_n = placed / 2;
        leaf.count = static_cast<uint16_t>(left_n);
        for (int i = 0; i < left_n; ++i) leaf.key[i] = tmp[i];
        const uint32_t rid = new_leaf();
        Leaf &right = leaves_[rid];
        right.count = static_cast<uint16_t>(placed - left_n);
        for (int i = 0; i < right.count; ++i) right.key[i] = tmp[left_n + i];
        const uint64_t promote = right.key[0];
        if (npath == 0) {
            const uint32_t nr = new_inner();
            Inner &top = inners_[nr];
            top.nchild = 2;
            top.child[0] = leaf_id;
            top.child[1] = rid;
            top.sep[0] = promote;
            root_ = nr;
            height_ = 2;
        } else {
            const Step parent = path[npath - 1];
            split_inner(parent.id, parent.slot, promote, rid, path, npath - 1);
        }
        ++n_;
        return true;
    }
};

}  // namespace shi
